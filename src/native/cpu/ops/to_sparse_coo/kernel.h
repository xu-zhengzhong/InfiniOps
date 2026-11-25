#ifndef INFINI_OPS_CPU_TO_SPARSE_COO_KERNEL_H_
#define INFINI_OPS_CPU_TO_SPARSE_COO_KERNEL_H_

#include <cstring>

#include "base/to_sparse_coo.h"
#include "common/sparse_buffers.h"
#include "native/cpu/sparse_matrix.h"

namespace infini::ops {

template <>
class Operator<ToSparseCoo, Device::Type::kCpu, 0> : public ToSparseCoo {
 public:
  using ToSparseCoo::ToSparseCoo;

  static bool supports(const SparseTensor& input, const SparseTensor& out) {
    return input.layout() == TensorLayout::kCsr &&
           out.layout() == TensorLayout::kCoo;
  }

  static ValidationResult checked_call(const Handle&, const Config&,
                                       const SparseTensor& input,
                                       const SparseTensor& out) {
    const Tensor *indices, *values;
    (void)out.indices(indices);
    (void)out.values(values);
    auto status = detail::CheckSparseWrites(input, {*indices, *values});
    if (!status) return status;
    status = native::cpu::ValidateSparseIndices(input);
    if (!status) {
      status.argument = 0;
      return status;
    }
    if (!input.nnz()) return {};
    const Tensor *crow, *col, *entries;
    (void)input.crow_indices(crow);
    (void)input.col_indices(col);
    (void)input.raw_values(entries);
    auto writable_indices = *indices;
    auto writable_values = *values;
    auto* coords = static_cast<std::int64_t*>(writable_indices.data());
    native::cpu::detail::VisitSparseEntries(
        *crow, col, [&](std::size_t row, std::size_t column, std::size_t p) {
          coords[p] = row;
          coords[input.nnz() + p] = column;
        });
    std::memcpy(writable_values.data(), entries->data(),
                entries->numel() * entries->element_size());
    return {};
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CPU_TO_SPARSE_COO_KERNEL_H_
