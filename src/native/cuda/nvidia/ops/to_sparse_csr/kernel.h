#ifndef INFINI_OPS_NVIDIA_TO_SPARSE_CSR_KERNEL_H_
#define INFINI_OPS_NVIDIA_TO_SPARSE_CSR_KERNEL_H_

#include "base/to_sparse_csr.h"

namespace infini::ops {

template <>
class Operator<ToSparseCsr, Device::Type::kNvidia, 0> : public ToSparseCsr {
 public:
  using ToSparseCsr::ToSparseCsr;

  static bool supports(const SparseTensor& input, SparseOutputPlan*) {
    return input.layout() == TensorLayout::kCoo;
  }

  static ValidationResult checked_call(const Handle& handle, const Config&,
                                       const SparseTensor& input,
                                       SparseOutputPlan* plan);
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_TO_SPARSE_CSR_KERNEL_H_
