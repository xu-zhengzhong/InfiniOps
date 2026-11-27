#ifndef INFINI_OPS_NVIDIA_TO_DENSE_KERNEL_H_
#define INFINI_OPS_NVIDIA_TO_DENSE_KERNEL_H_

#include "base/to_dense.h"

namespace infini::ops {

template <>
class Operator<ToDense, Device::Type::kNvidia, 0> : public ToDense {
 public:
  using ToDense::ToDense;

  static bool supports(const SparseTensor& input, const Tensor&) {
    return input.layout() == TensorLayout::kCsr ||
           input.layout() == TensorLayout::kCoo;
  }

  static ValidationResult checked_call(const Handle& handle, const Config&,
                                       const SparseTensor& input,
                                       const Tensor& out);
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_TO_DENSE_KERNEL_H_
