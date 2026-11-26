#ifndef INFINI_OPS_TORCH_COALESCE_H_
#define INFINI_OPS_TORCH_COALESCE_H_

#include "base/coalesce.h"

namespace infini::ops {

template <>
class Operator<Coalesce, Device::Type::kCpu, 8> : public Coalesce {
 public:
  static bool supports(const SparseTensor& input, SparseOutputPlan*) {
    return input.device().index() == 0 && input.layout() == TensorLayout::kCoo;
  }

  static ValidationResult checked_call(const Handle& handle, const Config&,
                                       const SparseTensor& input,
                                       SparseOutputPlan* plan);
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_COALESCE_H_
