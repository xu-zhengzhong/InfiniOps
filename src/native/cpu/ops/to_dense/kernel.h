#ifndef INFINI_OPS_CPU_TO_DENSE_KERNEL_H_
#define INFINI_OPS_CPU_TO_DENSE_KERNEL_H_

#include "base/to_dense.h"
#include "native/cpu/sparse_matrix.h"

namespace infini::ops {

template <>
class Operator<ToDense, Device::Type::kCpu, 0> : public ToDense {
 public:
  using ToDense::ToDense;

  static bool supports(const SparseTensor& input, const Tensor&) {
    return native::cpu::SupportsSparseMatrix(input);
  }

  static ValidationResult checked_call(const Handle&, const Config&,
                                       const SparseTensor& input, Tensor out) {
    const auto width = input.shape()[1];
    return native::cpu::AccumulateSparseEntries(
        input, out,
        [width](auto* dst, std::size_t row, std::size_t col, auto value) {
          dst[row * width + col] += value;
        });
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CPU_TO_DENSE_KERNEL_H_
