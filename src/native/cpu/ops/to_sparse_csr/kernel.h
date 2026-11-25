#ifndef INFINI_OPS_CPU_TO_SPARSE_CSR_KERNEL_H_
#define INFINI_OPS_CPU_TO_SPARSE_CSR_KERNEL_H_

#include "base/to_sparse_csr.h"
#include "native/cpu/sparse_output_provider.h"

namespace infini::ops {

template <>
class Operator<ToSparseCsr, Device::Type::kCpu, 0> : public ToSparseCsr {
 public:
  static bool supports(const SparseTensor& input, SparseOutputPlan*) {
    return input.layout() == TensorLayout::kCoo;
  }

  static ValidationResult checked_call(const Handle&, const Config&,
                                       const SparseTensor& input,
                                       SparseOutputPlan* plan) {
    return native::cpu::SparseOutputProvider::Prepare(input, TensorLayout::kCsr,
                                                      plan);
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CPU_TO_SPARSE_CSR_KERNEL_H_
