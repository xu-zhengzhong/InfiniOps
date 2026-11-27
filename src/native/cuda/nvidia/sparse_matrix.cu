#include "native/cuda/nvidia/ops/sparse_mm/kernel.h"
#include "native/cuda/nvidia/ops/sparse_sampled_addmm/kernel.h"
#include "native/cuda/nvidia/ops/to_dense/kernel.h"
#include "native/cuda/nvidia/ops/to_sparse_coo/kernel.h"
#include "native/cuda/nvidia/sparse.cuh"

namespace infini::ops {
namespace {
using namespace native::nvidia::sparse;

template <typename V>
__device__ void Add(V* dest, V value) {
  atomicAdd(dest, value);
}

// Global float atomicAdd flushes subnormals, including an existing partial
// sum. Preserve stored float values with ordinary round-to-nearest addition.
template <>
__device__ void Add(float* dest, float value) {
  auto* bits = reinterpret_cast<unsigned int*>(dest);
  unsigned int expected = 0;
  for (;;) {
    const auto sum = __fadd_rn(__uint_as_float(expected), value);
    const auto observed = atomicCAS(bits, expected, __float_as_uint(sum));
    if (observed == expected) return;
    expected = observed;
  }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
template <>
__device__ void Add(double* dest, double value) {
  auto* bits = reinterpret_cast<unsigned long long*>(dest);
  unsigned long long old = 0;
  unsigned long long expected;
  do {
    expected = old;
    old =
        atomicCAS(bits, expected,
                  __double_as_longlong(__longlong_as_double(expected) + value));
  } while (old != expected);
}
#endif

// Unsorted COO can contain duplicate coordinates. Only stored entries
// participate, including explicit zeros paired with a nonfinite rhs.
template <typename V>
__global__ void AccumulateCoo(const std::int64_t* coords, const V* values,
                              std::size_t n, std::size_t width, const V* rhs,
                              V* out) {
  for (auto p = Thread(); p < n; p += Step()) {
    const auto row = static_cast<std::size_t>(coords[p]);
    const auto col = static_cast<std::size_t>(coords[n + p]);
    if (rhs) {
      for (std::size_t j = 0; j < width; ++j)
        Add(out + row * width + j, values[p] * rhs[col * width + j]);
    } else {
      Add(out + row * width + col, values[p]);
    }
  }
}

template <typename I, typename V>
__global__ void CsrMm(const I* crow, const I* col, const V* values,
                      const V* rhs, std::size_t width, std::size_t count,
                      V* out) {
  for (auto i = Thread(); i < count; i += Step()) {
    const auto row = i / width;
    V sum = 0;
    for (auto p = crow[row]; p < crow[row + 1]; ++p)
      sum +=
          values[p] * rhs[static_cast<std::size_t>(col[p]) * width + i % width];
    out[i] = sum;
  }
}

template <typename I, typename V>
__global__ void CsrToDense(const I* crow, const I* col, const V* values,
                           std::size_t rows, std::size_t width, V* out) {
  for (auto r = Thread(); r < rows; r += Step())
    for (auto p = crow[r]; p < crow[r + 1]; ++p)
      out[r * width + col[p]] = values[p];
}

ValidationResult DenseOutput(const Handle& handle, const SparseTensor& input,
                             const Tensor* rhs, const Tensor& out) {
  auto status = rhs ? detail::CheckSparseWrites(input, {out}, {*rhs})
                    : detail::CheckSparseWrites(input, {out});
  if (!status) return status;
  return WithInput(handle.stream(), input, nullptr, [&](cudaStream_t stream) {
    const auto count = out.numel();
    if (!count) return ValidationResult{};
    const detail::SparseComponents src(input);
    if (!src.columns || !rhs || !input.nnz())
      Check(cudaMemsetAsync(Write<void>(out), 0, count * out.element_size(),
                            stream));
    if (input.nnz()) {
      VisitTypes(input, [&](auto index, auto value) {
        using I = decltype(index);
        using V = decltype(value);
        auto* dst = Write<V>(out);
        const auto* entries = Read<V>(*src.values);
        if (!src.columns) {
          AccumulateCoo<<<Blocks(input.nnz()), kThreads, 0, stream>>>(
              Read<std::int64_t>(*src.indices), entries, input.nnz(),
              out.size(1), rhs ? Read<V>(*rhs) : nullptr, dst);
        } else if (rhs) {
          CsrMm<<<Blocks(count), kThreads, 0, stream>>>(
              Read<I>(*src.indices), Read<I>(*src.columns), entries,
              Read<V>(*rhs), out.size(1), count, dst);
        } else {
          CsrToDense<<<Blocks(out.size(0)), kThreads, 0, stream>>>(
              Read<I>(*src.indices), Read<I>(*src.columns), entries,
              out.size(0), out.size(1), dst);
        }
      });
      Check(cudaGetLastError());
    }
    return ValidationResult{};
  });
}

template <typename I>
__global__ void ExpandCsr(const I* crow, const I* col, std::size_t rows,
                          std::size_t n, std::int64_t* coords) {
  for (auto r = Thread(); r < rows; r += Step()) {
    for (auto p = crow[r]; p < crow[r + 1]; ++p) {
      coords[p] = r;
      coords[n + p] = col[p];
    }
  }
}

template <typename I, typename V>
__global__ void Sample(const I* crow, const I* col, const V* values,
                       const V* lhs, const V* rhs, std::size_t rows,
                       std::size_t n, std::size_t inner, std::size_t width,
                       V alpha, V beta, V* out) {
  for (auto p = Thread(); p < n; p += Step()) {
    const V scaled = beta * values[p];
    if (inner == 0) {
      out[p] = scaled;
      continue;
    }
    std::size_t lo = 0, hi = rows;
    while (lo < hi) {
      const auto mid = lo + (hi - lo) / 2;
      if (static_cast<std::size_t>(crow[mid + 1]) <= p)
        lo = mid + 1;
      else
        hi = mid;
    }
    V dot = 0;
    for (std::size_t k = 0; k < inner; ++k)
      dot += lhs[lo * inner + k] * rhs[k * width + col[p]];
    out[p] = alpha * dot + scaled;
  }
}

}  // namespace

ValidationResult Operator<ToDense, Device::Type::kNvidia, 0>::checked_call(
    const Handle& handle, const Config&, const SparseTensor& input,
    const Tensor& out) {
  return DenseOutput(handle, input, nullptr, out);
}

ValidationResult Operator<SparseMm, Device::Type::kNvidia, 0>::checked_call(
    const Handle& handle, const Config&, const SparseTensor& input,
    const Tensor& rhs, const std::string&, const Tensor& out) {
  return DenseOutput(handle, input, &rhs, out);
}

ValidationResult Operator<ToSparseCoo, Device::Type::kNvidia, 0>::checked_call(
    const Handle& handle, const Config&, const SparseTensor& input,
    const SparseTensor& out) {
  const detail::SparseComponents src(input), dst(out);
  auto status =
      detail::CheckSparseWrites(input, {*dst.indices, *dst.values}, {}, 1);
  if (!status) return status;
  return WithInput(handle.stream(), input, nullptr, [&](cudaStream_t stream) {
    if (input.nnz()) {
      auto launch = [&](auto index) {
        using I = decltype(index);
        ExpandCsr<<<Blocks(input.shape()[0]), kThreads, 0, stream>>>(
            Read<I>(*src.indices), Read<I>(*src.columns), input.shape()[0],
            input.nnz(), Write<std::int64_t>(*dst.indices));
      };
      if (input.index_dtype() == DataType::kInt32)
        launch(std::int32_t{});
      else
        launch(std::int64_t{});
      Check(cudaGetLastError());
      Copy(*src.values, *dst.values, stream);
    }
    return ValidationResult{};
  });
}

ValidationResult Operator<SparseSampledAddmm, Device::Type::kNvidia,
                          0>::checked_call(const Handle& handle, const Config&,
                                           const SparseTensor& input,
                                           const Tensor& lhs, const Tensor& rhs,
                                           double beta, double alpha,
                                           const SparseTensor& out) {
  const detail::SparseComponents src(input), dst(out);
  auto status = detail::CheckSparseWrites(
      input, {*dst.indices, *dst.columns, *dst.values}, {lhs, rhs}, 5);
  if (!status) return status;
  return WithInput(handle.stream(), input, nullptr, [&](cudaStream_t stream) {
    Copy(*src.indices, *dst.indices, stream);
    Copy(*src.columns, *dst.columns, stream);
    if (input.nnz()) {
      VisitTypes(input, [&](auto index, auto value) {
        using I = decltype(index);
        using V = decltype(value);
        Sample<<<Blocks(input.nnz()), kThreads, 0, stream>>>(
            Read<I>(*src.indices), Read<I>(*src.columns), Read<V>(*src.values),
            Read<V>(lhs), Read<V>(rhs), input.shape()[0], input.nnz(),
            lhs.size(1), rhs.size(1), static_cast<V>(alpha),
            static_cast<V>(beta), Write<V>(*dst.values));
      });
      Check(cudaGetLastError());
    }
    return ValidationResult{};
  });
}
}  // namespace infini::ops
