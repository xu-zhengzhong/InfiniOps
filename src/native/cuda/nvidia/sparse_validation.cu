#include "native/cuda/nvidia/sparse.cuh"

namespace infini::ops::native::nvidia::sparse {
namespace {

__device__ void Report(Diagnostic* error, ValidationCode code,
                       ValidationComponent component, std::size_t position) {
  if (atomicCAS(&error->code, 0, static_cast<int>(code)) == 0) {
    error->component = component;
    error->position = position;
  }
}

__global__ void CheckCoo(const std::int64_t* coords, std::size_t n,
                         std::size_t rows, std::size_t cols, bool coalesced,
                         Diagnostic* error) {
  for (auto p = Thread(); p < n; p += Step()) {
    const auto row = coords[p], col = coords[n + p];
    if (static_cast<std::size_t>(row) >= rows)
      Report(error, ValidationCode::kIndexOutOfBounds,
             ValidationComponent::kIndices, p);
    else if (static_cast<std::size_t>(col) >= cols)
      Report(error, ValidationCode::kIndexOutOfBounds,
             ValidationComponent::kIndices, n + p);
    else if (coalesced && p &&
             (row < coords[p - 1] ||
              (row == coords[p - 1] && col <= coords[n + p - 1])))
      Report(error, ValidationCode::kInvalidCoalescedClaim,
             ValidationComponent::kIndices, p);
  }
}

template <typename I>
__global__ void CheckCsr(const I* crow, const I* col, std::size_t n,
                         std::size_t rows, std::size_t cols,
                         Diagnostic* error) {
  // Check each interval before following it, even if another row is invalid.
  for (auto r = Thread(); r <= rows; r += Step()) {
    const auto begin = static_cast<std::size_t>(crow[r]);
    if (begin > n || (r == 0 && begin != 0) || (r == rows && begin != n)) {
      Report(error, ValidationCode::kInvalidStructure,
             ValidationComponent::kCrowIndices, r);
      continue;
    }
    if (r == rows) continue;
    const auto end = static_cast<std::size_t>(crow[r + 1]);
    if (end < begin || end > n) {
      Report(error, ValidationCode::kInvalidStructure,
             ValidationComponent::kCrowIndices, r + 1);
      continue;
    }
    for (auto p = begin; p < end; ++p) {
      if (static_cast<std::size_t>(col[p]) >= cols)
        Report(error, ValidationCode::kIndexOutOfBounds,
               ValidationComponent::kColIndices, p);
      else if (p > begin && col[p - 1] >= col[p])
        Report(error, ValidationCode::kInvalidStructure,
               ValidationComponent::kColIndices, p);
    }
  }
}

}  // namespace

ValidationResult Validate(const SparseTensor& input, cudaStream_t stream,
                          Diagnostic* scratch) {
  const auto n = input.nnz();
  if (input.layout() == TensorLayout::kCoo && n == 0) return {};
  const detail::SparseComponents src(input);
  const auto shape = input.shape();
  Diagnostic host{};
  struct Temporary {
    Diagnostic* data{};
    ~Temporary() {
      if (data) (void)cudaFree(data);
    }
  } temporary;
  if (!scratch) {
    Check(cudaMalloc(&temporary.data, sizeof(Diagnostic)));
    scratch = temporary.data;
  }
  Check(cudaMemsetAsync(scratch, 0, sizeof(Diagnostic), stream));
  if (!src.columns) {
    bool coalesced = false;
    (void)input.is_coalesced(coalesced);
    CheckCoo<<<Blocks(n), kThreads, 0, stream>>>(
        Read<std::int64_t>(*src.indices), n, shape[0], shape[1], coalesced,
        scratch);
  } else {
    const auto launch = [&](auto index) {
      using I = decltype(index);
      CheckCsr<<<Blocks(shape[0] + 1), kThreads, 0, stream>>>(
          Read<I>(*src.indices), Read<I>(*src.columns), n, shape[0], shape[1],
          scratch);
    };
    if (input.index_dtype() == DataType::kInt32)
      launch(std::int32_t{});
    else
      launch(std::int64_t{});
  }
  Check(cudaGetLastError());
  Check(cudaMemcpyAsync(&host, scratch, sizeof(host), cudaMemcpyDeviceToHost,
                        stream));
  Check(cudaStreamSynchronize(stream));
  if (host.code)
    return {static_cast<ValidationCode>(host.code), host.component,
            host.position, 0};
  return {};
}

}  // namespace infini::ops::native::nvidia::sparse
