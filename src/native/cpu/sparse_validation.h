#ifndef INFINI_OPS_NATIVE_CPU_SPARSE_VALIDATION_H_
#define INFINI_OPS_NATIVE_CPU_SPARSE_VALIDATION_H_

#include "sparse.h"

namespace infini::ops::native::cpu {

namespace detail {

template <typename Index>
ValidationResult ValidateCsrIndices(const Index* crow, const Index* col,
                                    const SparseTensor::Shape& shape,
                                    std::size_t nnz) {
  // Establish safe row intervals before using any pointer as an offset.
  if (crow[0] != 0) {
    return {ValidationCode::kInvalidStructure,
            ValidationComponent::kCrowIndices, 0};
  }
  const auto rows = shape[0];
  for (std::size_t row = 1; row <= rows; ++row) {
    if (crow[row] < crow[row - 1] ||
        static_cast<std::uintmax_t>(crow[row]) > nnz) {
      return {ValidationCode::kInvalidStructure,
              ValidationComponent::kCrowIndices, row};
    }
  }
  if (static_cast<std::size_t>(crow[rows]) != nnz) {
    return {ValidationCode::kInvalidStructure,
            ValidationComponent::kCrowIndices, rows};
  }
  for (std::size_t row = 0; row < rows; ++row) {
    for (auto p = crow[row]; p < crow[row + 1]; ++p) {
      if (col[p] < 0 || static_cast<std::uintmax_t>(col[p]) >= shape[1]) {
        return {ValidationCode::kIndexOutOfBounds,
                ValidationComponent::kColIndices, static_cast<std::size_t>(p)};
      }
      if (p > crow[row] && col[p] <= col[p - 1]) {
        return {ValidationCode::kInvalidStructure,
                ValidationComponent::kColIndices, static_cast<std::size_t>(p)};
      }
    }
  }
  return {};
}

}  // namespace detail

// RT factories have checked component geometry, contiguity and address ranges.
// The caller owns the buffers and keeps them stable throughout this scan.
inline ValidationResult ValidateSparseIndices(const SparseTensor& tensor) {
  if (tensor.device().type() != Device::Type::kCpu) {
    return {ValidationCode::kUnsupportedCapability};
  }
  const auto shape = tensor.shape();
  const auto nnz = tensor.nnz();
  const Tensor* indices = nullptr;
  if (tensor.layout() == TensorLayout::kCoo) {
    (void)tensor.raw_indices(indices);
    bool coalesced = false;
    (void)tensor.is_coalesced(coalesced);
    const auto* coords = static_cast<const std::int64_t*>(indices->data());
    for (std::size_t p = 0; p < nnz; ++p) {
      const auto row = coords[p];
      const auto col = coords[nnz + p];
      if (row < 0 || static_cast<std::uintmax_t>(row) >= shape[0]) {
        return {ValidationCode::kIndexOutOfBounds,
                ValidationComponent::kIndices, p};
      }
      if (col < 0 || static_cast<std::uintmax_t>(col) >= shape[1]) {
        return {ValidationCode::kIndexOutOfBounds,
                ValidationComponent::kIndices, nnz + p};
      }
      if (coalesced && p > 0 &&
          (row < coords[p - 1] ||
           (row == coords[p - 1] && col <= coords[nnz + p - 1]))) {
        return {ValidationCode::kInvalidCoalescedClaim,
                ValidationComponent::kIndices, p};
      }
    }
    return {};
  }
  if (tensor.layout() != TensorLayout::kCsr) {
    return {ValidationCode::kUnsupportedLayout, ValidationComponent::kLayout};
  }
  const Tensor* columns = nullptr;
  (void)tensor.crow_indices(indices);
  (void)tensor.col_indices(columns);
  if (tensor.index_dtype() == DataType::kInt32) {
    return detail::ValidateCsrIndices(
        static_cast<const std::int32_t*>(indices->data()),
        static_cast<const std::int32_t*>(columns->data()), shape, nnz);
  }
  return detail::ValidateCsrIndices(
      static_cast<const std::int64_t*>(indices->data()),
      static_cast<const std::int64_t*>(columns->data()), shape, nnz);
}

}  // namespace infini::ops::native::cpu

#endif
