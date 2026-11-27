#ifndef INFINI_OPS_NVIDIA_TO_SPARSE_COO_KERNEL_H_
#define INFINI_OPS_NVIDIA_TO_SPARSE_COO_KERNEL_H_

#include "base/to_sparse_coo.h"

namespace infini::ops {

template <>
class Operator<ToSparseCoo, Device::Type::kNvidia, 0> : public ToSparseCoo {
 public:
  using ToSparseCoo::ToSparseCoo;

  static bool supports(const SparseTensor& input, const SparseTensor&) {
    return input.layout() == TensorLayout::kCsr;
  }

  static ValidationResult checked_call(const Handle& handle, const Config&,
                                       const SparseTensor& input,
                                       const SparseTensor& out);
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_TO_SPARSE_COO_KERNEL_H_
