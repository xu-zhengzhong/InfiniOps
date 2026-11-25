#ifndef INFINI_OPS_CPU_COALESCE_KERNEL_H_
#define INFINI_OPS_CPU_COALESCE_KERNEL_H_

#include "base/coalesce.h"
#include "native/cpu/sparse_output_provider.h"

namespace infini::ops {

template <>
class Operator<Coalesce, Device::Type::kCpu, 0> : public Coalesce {
 public:
  static bool supports(const SparseTensor& input, SparseOutputPlan*) {
    return input.layout() == TensorLayout::kCoo;
  }

  static ValidationResult checked_call(const Handle&, const Config&,
                                       const SparseTensor& input,
                                       SparseOutputPlan* plan) {
    return native::cpu::SparseOutputProvider::Prepare(input, TensorLayout::kCoo,
                                                      plan);
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CPU_COALESCE_KERNEL_H_
