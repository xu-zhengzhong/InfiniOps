#ifndef INFINI_OPS_NVIDIA_COALESCE_KERNEL_H_
#define INFINI_OPS_NVIDIA_COALESCE_KERNEL_H_

#include "base/coalesce.h"

namespace infini::ops {

template <>
class Operator<Coalesce, Device::Type::kNvidia, 0> : public Coalesce {
 public:
  using Coalesce::Coalesce;

  static bool supports(const SparseTensor& input, SparseOutputPlan*) {
    return input.layout() == TensorLayout::kCoo;
  }

  static ValidationResult checked_call(const Handle& handle, const Config&,
                                       const SparseTensor& input,
                                       SparseOutputPlan* plan);
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_COALESCE_KERNEL_H_
