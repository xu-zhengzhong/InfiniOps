#ifndef INFINI_OPS_TORCH_TO_DENSE_H_
#define INFINI_OPS_TORCH_TO_DENSE_H_

#include "base/to_dense.h"

namespace infini::ops {

template <>
class Operator<ToDense, Device::Type::kCpu, 8> : public ToDense {
 public:
  using ToDense::ToDense;

  static bool supports(const SparseTensor& input, const Tensor&) {
    return input.device().index() == 0 &&
           (input.layout() == TensorLayout::kCoo ||
            input.layout() == TensorLayout::kCsr);
  }

  static ValidationResult checked_call(const Handle& handle, const Config&,
                                       const SparseTensor& input,
                                       const Tensor& out);
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_TO_DENSE_H_
