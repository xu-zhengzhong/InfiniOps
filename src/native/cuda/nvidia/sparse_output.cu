#include <thrust/iterator/counting_iterator.h>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>

#include "native/cuda/nvidia/ops/coalesce/kernel.h"
#include "native/cuda/nvidia/ops/to_sparse_csr/kernel.h"
#include "native/cuda/nvidia/sparse.cuh"

namespace infini::ops {
namespace {
using namespace native::nvidia::sparse;
using Key = std::uint64_t;
using Index = std::int64_t;

bool NeedsSort(const SparseTensor& input) {
  bool coalesced = false;
  (void)input.is_coalesced(coalesced);
  return !coalesced && input.nnz() > 1;
}

__global__ void MakeKeys(const Index* coords, std::size_t n, std::size_t width,
                         Key* keys) {
  // RT checked rows*width fits size_t and the validator checked each
  // coordinate.
  for (auto p = Thread(); p < n; p += Step())
    keys[p] = Key(coords[p]) * width + coords[n + p];
}

struct IsGroupStart {
  const Key* keys;
  __device__ bool operator()(int i) const {
    return !i || keys[i] != keys[i - 1];
  }
};

template <typename V>
__global__ void WriteGroups(const Key* keys, const V* values, const int* starts,
                            int n, int groups, std::size_t width, Index* rows,
                            Index* cols, V* out) {
  for (auto g = Thread(); g < static_cast<std::size_t>(groups); g += Step()) {
    const int begin = starts[g];
    const int end = g + 1 == groups ? n : starts[g + 1];
    V sum = values[begin];
    for (int p = begin + 1; p < end; ++p) sum += values[p];
    const Key key = keys[begin];
    if (rows) rows[g] = key / width;
    cols[g] = key % width;
    out[g] = sum;
  }
}

__global__ void RowPointers(const Index* coo_rows, const Key* keys,
                            const int* starts, std::size_t count,
                            std::size_t rows, std::size_t width, Index* crow) {
  for (auto r = Thread(); r <= rows; r += Step()) {
    std::size_t lo = 0, hi = count;
    while (lo < hi) {
      const auto mid = lo + (hi - lo) / 2;
      const auto row = starts ? keys[starts[mid]] / width : Key(coo_rows[mid]);
      if (row < r)
        lo = mid + 1;
      else
        hi = mid;
    }
    crow[r] = lo;
  }
}

void WriteCrow(const SparseOutputBuffer& out, const Index* coords,
               const Key* keys, const int* starts, std::size_t actual,
               cudaStream_t stream) {
  RowPointers<<<Blocks(out.shape[0] + 1), kThreads, 0, stream>>>(
      coords, keys, starts, actual, out.shape[0], out.shape[1],
      static_cast<Index*>(out.indices.data));
  Check(cudaGetLastError());
}

// The first part of workspace holds the keys, sorted values, group starts and
// count. Sort and select run consecutively and share the aligned CUB storage.
constexpr std::size_t kCubAlignment = 256;
template <typename V>
std::size_t CubOffset(int n) {
  const auto prefix =
      std::size_t(n) * (2 * sizeof(Key) + sizeof(V) + sizeof(int)) +
      sizeof(int);
  return (prefix + kCubAlignment - 1) / kCubAlignment * kCubAlignment;
}

template <typename V>
std::size_t SortWorkspace(int n, cudaStream_t stream) {
  std::size_t sort = 0, select = 0;
  Key* keys = nullptr;
  V* values = nullptr;
  int* positions = nullptr;
  Check(cub::DeviceRadixSort::SortPairs(nullptr, sort, keys, keys, values,
                                        values, n, 0, 64, stream));
  Check(cub::DeviceSelect::If(nullptr, select,
                              thrust::counting_iterator<int>(0), positions,
                              positions, n, IsGroupStart{nullptr}, stream));
  return CubOffset<V>(n) + std::max(sort, select);
}

ValidationResult Inspect(void* stream, const SparseTensor& input,
                         SparseBufferRequirement& req) {
  // DeviceSelect::If in CUB 2.7 takes an int item count. The sparse index
  // width is independent of this bound on sorting work.
  if (NeedsSort(input) && input.nnz() > std::numeric_limits<int>::max())
    return {ValidationCode::kUnsupportedCapability,
            ValidationComponent::kIndices,
            {},
            0};
  return WithInput(stream, input, nullptr, [&](cudaStream_t cuda_stream) {
    if (NeedsSort(input)) {
      const int n = static_cast<int>(input.nnz());
      req = {input.dtype() == DataType::kFloat32
                 ? SortWorkspace<float>(n, cuda_stream)
                 : SortWorkspace<double>(n, cuda_stream),
             kCubAlignment};
    } else if (input.nnz()) {
      req = {sizeof(Diagnostic), alignof(Diagnostic)};
    }
    return ValidationResult{};
  });
}

template <typename V>
ValidationResult SortAndWrite(const SparseTensor& input,
                              const SparseOutputBuffer& out,
                              SparseBufferSpan workspace, std::size_t& actual,
                              cudaStream_t stream) {
  const detail::SparseComponents src(input);
  const int n = static_cast<int>(input.nnz());
  auto* keys = static_cast<Key*>(workspace.data);
  auto* sorted = keys + n;
  auto* entries = reinterpret_cast<V*>(sorted + n);
  auto* starts = reinterpret_cast<int*>(entries + n);
  auto* count = starts + n;
  const auto offset = CubOffset<V>(n);
  if (workspace.bytes <= offset) return {ValidationCode::kInvalidWorkspace};
  auto* temp = static_cast<char*>(workspace.data) + offset;
  auto sort_bytes = workspace.bytes - offset;
  auto select_bytes = sort_bytes;
  MakeKeys<<<Blocks(n), kThreads, 0, stream>>>(Read<Index>(*src.indices), n,
                                               out.shape[1], keys);
  Check(cudaGetLastError());
  Check(cub::DeviceRadixSort::SortPairs(temp, sort_bytes, keys, sorted,
                                        Read<V>(*src.values), entries, n, 0, 64,
                                        stream));
  Check(cub::DeviceSelect::If(temp, select_bytes,
                              thrust::counting_iterator<int>(0), starts, count,
                              n, IsGroupStart{sorted}, stream));
  int groups{};
  Check(cudaMemcpyAsync(&groups, count, sizeof(groups), cudaMemcpyDeviceToHost,
                        stream));
  Check(cudaStreamSynchronize(stream));
  if (groups < 1 || groups > n) return {ValidationCode::kExecutionFailure};

  actual = static_cast<std::size_t>(groups);
  const bool csr = out.layout == TensorLayout::kCsr;
  auto* rows = csr ? nullptr : static_cast<Index*>(out.indices.data);
  auto* cols = csr ? static_cast<Index*>(out.col_indices.data) : rows + actual;
  WriteGroups<<<Blocks(actual), kThreads, 0, stream>>>(
      sorted, entries, starts, n, groups, out.shape[1], rows, cols,
      static_cast<V*>(out.values.data));
  Check(cudaGetLastError());
  if (csr) WriteCrow(out, nullptr, sorted, starts, actual, stream);
  return {};
}

ValidationResult Execute(void* raw_stream, const SparseTensor& input,
                         const SparseOutputBuffer& out,
                         SparseBufferSpan workspace, std::size_t& actual) {
  return WithInput(
      raw_stream, input, static_cast<Diagnostic*>(workspace.data),
      [&](cudaStream_t stream) {
        if (NeedsSort(input)) {
          return input.dtype() == DataType::kFloat32
                     ? SortAndWrite<float>(input, out, workspace, actual,
                                           stream)
                     : SortAndWrite<double>(input, out, workspace, actual,
                                            stream);
        }
        const detail::SparseComponents src(input);
        const auto* coords = Read<Index>(*src.indices);
        actual = input.nnz();
        if (out.layout == TensorLayout::kCsr) {
          WriteCrow(out, coords, nullptr, nullptr, actual, stream);
          if (actual)
            Check(cudaMemcpyAsync(out.col_indices.data, coords + actual,
                                  actual * sizeof(Index),
                                  cudaMemcpyDeviceToDevice, stream));
        } else if (actual) {
          Check(cudaMemcpyAsync(out.indices.data, coords,
                                2 * actual * sizeof(Index),
                                cudaMemcpyDeviceToDevice, stream));
        }
        if (actual)
          Check(cudaMemcpyAsync(out.values.data, src.values->data(),
                                actual * src.values->element_size(),
                                cudaMemcpyDeviceToDevice, stream));
        return ValidationResult{};
      });
}
}  // namespace

ValidationResult Operator<Coalesce, Device::Type::kNvidia, 0>::checked_call(
    const Handle& handle, const Config&, const SparseTensor& input,
    SparseOutputPlan* plan) {
  return detail::SparseOutputProvider::Bind(input, TensorLayout::kCoo, plan,
                                            Execute, handle.stream(), {}, false,
                                            Inspect);
}

ValidationResult Operator<ToSparseCsr, Device::Type::kNvidia, 0>::checked_call(
    const Handle& handle, const Config&, const SparseTensor& input,
    SparseOutputPlan* plan) {
  return detail::SparseOutputProvider::Bind(input, TensorLayout::kCsr, plan,
                                            Execute, handle.stream(), {}, false,
                                            Inspect);
}

}  // namespace infini::ops
