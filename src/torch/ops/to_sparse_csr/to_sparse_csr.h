#ifndef INFINI_OPS_TORCH_TO_SPARSE_CSR_H_
#define INFINI_OPS_TORCH_TO_SPARSE_CSR_H_

#include "base/to_sparse_csr.h"

namespace infini::ops {

template <>
class Operator<ToSparseCsr, Device::Type::kCpu, 8> : public ToSparseCsr {
 public:
  static bool supports(const SparseTensor& input, SparseOutputPlan*) {
    return input.device().index() == 0 && input.layout() == TensorLayout::kCoo;
  }

  static ValidationResult checked_call(const Handle& handle, const Config&,
                                       const SparseTensor& input,
                                       SparseOutputPlan* plan);
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_TO_SPARSE_CSR_H_
