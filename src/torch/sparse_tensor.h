#ifndef INFINI_OPS_TORCH_SPARSE_TENSOR_H_
#define INFINI_OPS_TORCH_SPARSE_TENSOR_H_

#include <c10/core/impl/LocalDispatchKeySet.h>

#include <array>
#include <cstdint>
#include <limits>
#include <new>

#include "common/sparse_buffers.h"
#include "native/cpu/sparse_validation.h"
#include "torch/stream_guard.h"
#include "torch/tensor_.h"

namespace infini::ops::aten {

inline at::Tensor BorrowCpuTensor(const Tensor& tensor) {
  return ToAtenTensor<Device::Type::kCpu>(
      const_cast<void*>(tensor.data()), tensor.shape(), tensor.strides(),
      tensor.dtype(), tensor.device().index());
}

inline std::array<std::int64_t, 2> AtenSize(const SparseTensor& tensor) {
  // RT checks that each axis fits the index dtype.
  const auto shape = tensor.shape();
  return {static_cast<std::int64_t>(shape[0]),
          static_cast<std::int64_t>(shape[1])};
}

// The RT shape product fits `size_t`. ATen requires a signed 64-bit count.
inline ValidationResult CheckAtenSize(const SparseTensor& input) {
  if (input.numel() > std::numeric_limits<std::int64_t>::max())
    return {
        ValidationCode::kIntegerOverflow, ValidationComponent::kShape, {}, 0};
  return {};
}

inline ValidationResult CheckAtenSize(const Tensor& tensor, std::size_t arg) {
  for (auto size : tensor.shape()) {
    if (size > std::numeric_limits<std::int64_t>::max())
      return {ValidationCode::kIntegerOverflow,
              ValidationComponent::kShape,
              {},
              arg};
  }
  return {};
}

// The input has passed both metadata and CPU index validation. The wrappers
// borrow each component until the synchronous ATen call and copy complete.
inline at::Tensor ToAtenSparseTensor(const SparseTensor& input) {
  const auto [indices, columns, entries] = detail::SparseComponents(input);
  auto first = BorrowCpuTensor(*indices);
  auto values = BorrowCpuTensor(*entries);
  if (columns) {
    return at::sparse_csr_tensor(first, BorrowCpuTensor(*columns), values,
                                 AtenSize(input),
                                 values.options().layout(at::kSparseCsr));
  }
  bool coalesced = false;
  (void)input.is_coalesced(coalesced);
  return at::sparse_coo_tensor(first, values, AtenSize(input),
                               values.options().layout(at::kSparse), coalesced);
}

template <typename Compute>
ValidationResult WithSparseInput(void* stream, const SparseTensor& input,
                                 const Compute& compute) {
  auto status = CheckAtenSize(input);
  if (!status) return status;
  status = native::cpu::ValidateSparseIndices(input);
  if (!status) {
    status.argument = 0;
    return status;
  }
  try {
    const TorchStreamGuard<Device::Type::kCpu> guard{stream,
                                                     input.device().index()};
    const at::NoGradGuard no_grad;
    const c10::impl::ExcludeDispatchKeyGuard no_autocast{
        c10::DispatchKey::AutocastCPU};
    return compute(ToAtenSparseTensor(input));
  } catch (const c10::OutOfMemoryError&) {
    return {ValidationCode::kAllocationFailure};
  } catch (const std::bad_alloc&) {
    return {ValidationCode::kAllocationFailure};
  } catch (const c10::NotImplementedError&) {
    return {ValidationCode::kUnsupportedCapability};
  } catch (...) {
    return {ValidationCode::kExecutionFailure};
  }
}

inline ValidationResult CopyDenseResult(const at::Tensor& result,
                                        const Tensor& out) {
  auto target = BorrowCpuTensor(out);
  if (result.layout() != at::kStrided || result.device() != target.device() ||
      result.scalar_type() != target.scalar_type() ||
      result.sizes() != target.sizes())
    return {ValidationCode::kExecutionFailure};
  target.copy_(result);
  return {};
}

// Sparse `_out` kernels may resize their component tensors. Only ATen-owned
// temporaries are passed to those kernels. Copy into the exact caller views
// after checking the result, including a potentially strided COO index matrix.
inline ValidationResult CopySparseResult(const at::Tensor& result,
                                         const SparseTensor& out) {
  const auto [indices, columns, values] = detail::SparseComponents(out);
  if (result.layout() != (columns ? at::kSparseCsr : at::kSparse) ||
      !result.device().is_cpu() ||
      result.sizes() != at::IntArrayRef(AtenSize(out)) ||
      result.sparse_dim() != 2 || result.dense_dim() != 0 ||
      result.scalar_type() != ToAtenDataType(out.dtype()) ||
      result._nnz() != static_cast<std::int64_t>(out.nnz()) ||
      (!columns && !result.is_coalesced()))
    return {ValidationCode::kExecutionFailure};
  const auto first = columns ? result.crow_indices() : result.indices();
  const auto index_type = ToAtenDataType(out.index_dtype());
  if (first.scalar_type() != index_type ||
      (columns && result.col_indices().scalar_type() != index_type))
    return {ValidationCode::kExecutionFailure};
  BorrowCpuTensor(*indices).copy_(first);
  if (columns) BorrowCpuTensor(*columns).copy_(result.col_indices());
  BorrowCpuTensor(*values).copy_(result.values());
  return {};
}

}  // namespace infini::ops::aten

#endif  // INFINI_OPS_TORCH_SPARSE_TENSOR_H_
