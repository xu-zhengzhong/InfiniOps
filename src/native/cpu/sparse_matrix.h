#ifndef INFINI_OPS_NATIVE_CPU_SPARSE_MATRIX_H_
#define INFINI_OPS_NATIVE_CPU_SPARSE_MATRIX_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "common/sparse_buffers.h"
#include "dispatcher.h"
#include "native/cpu/sparse_validation.h"

namespace infini::ops::native::cpu {

inline bool SupportsSparseMatrix(const SparseTensor& input) {
  return input.device().type() == Device::Type::kCpu &&
         (input.layout() == TensorLayout::kCoo ||
          input.layout() == TensorLayout::kCsr);
}

// Common metadata validation has checked contiguity, byte counts and address
// endpoints for every operand. Empty ranges can have arbitrary addresses.
inline bool StorageOverlaps(const Tensor& a, const Tensor& b) {
  return infini::ops::detail::Overlaps(a, b);
}

namespace detail {

// Visit (row, column, storage position) in input order after index validation.
template <typename Visit>
void VisitSparseEntries(const Tensor& indices, const Tensor* columns,
                        const Visit& visit) {
  if (!columns) {
    const auto nnz = indices.size(1);
    const auto* coords = static_cast<const std::int64_t*>(indices.data());
    for (std::size_t p = 0; p < nnz; ++p) {
      visit(coords[p], coords[nnz + p], p);
    }
    return;
  }
  DispatchFunc<Device::Type::kCpu, List<DataType::kInt32, DataType::kInt64>>(
      indices.dtype(),
      [&](auto tag) {
        using Index = typename decltype(tag)::type;
        const auto* crow = static_cast<const Index*>(indices.data());
        const auto* col = static_cast<const Index*>(columns->data());
        for (std::size_t row = 0; row < indices.size(0) - 1; ++row) {
          for (auto p = crow[row]; p < crow[row + 1]; ++p) {
            visit(row, col[p], p);
          }
        }
      },
      "CPU CSR indices");
}

}  // namespace detail

// The sparse input is argument zero in both operations. Check all of its
// ranges and indices before the first output write, even for zero output width.
template <typename Accumulate>
ValidationResult AccumulateSparseEntries(const SparseTensor& input, Tensor out,
                                         const Accumulate& accumulate) {
  const Tensor* indices = nullptr;
  const Tensor* columns = nullptr;
  const Tensor* values = nullptr;
  if (input.layout() == TensorLayout::kCoo) {
    (void)input.raw_indices(indices);
  } else {
    (void)input.crow_indices(indices);
    (void)input.col_indices(columns);
  }
  (void)input.raw_values(values);
  if (StorageOverlaps(*indices, out)) {
    return {ValidationCode::kAliasing,
            columns ? ValidationComponent::kCrowIndices
                    : ValidationComponent::kIndices,
            {},
            0};
  }
  if (columns && StorageOverlaps(*columns, out)) {
    return {ValidationCode::kAliasing, ValidationComponent::kColIndices, {}, 0};
  }
  if (StorageOverlaps(*values, out)) {
    return {ValidationCode::kAliasing, ValidationComponent::kValues, {}, 0};
  }

  auto result = ValidateSparseIndices(input);
  if (!result) {
    result.argument = 0;
    return result;
  }
  const auto count = out.numel();
  if (count == 0) return {};
  DispatchFunc<Device::Type::kCpu, FloatTypes>(
      input.dtype(),
      [&](auto tag) {
        using Value = typename decltype(tag)::type;
        auto* dst = static_cast<Value*>(out.data());
        const auto* src = static_cast<const Value*>(values->data());
        std::fill_n(dst, count, Value{0});
        detail::VisitSparseEntries(
            *indices, columns,
            [&](std::size_t row, std::size_t col, std::size_t p) {
              accumulate(dst, row, col, src[p]);
            });
      },
      "CPU sparse matrix values");
  return {};
}

}  // namespace infini::ops::native::cpu

#endif  // INFINI_OPS_NATIVE_CPU_SPARSE_MATRIX_H_
