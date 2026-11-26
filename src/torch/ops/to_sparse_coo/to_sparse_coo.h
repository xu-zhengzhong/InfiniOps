#ifndef INFINI_OPS_TORCH_TO_SPARSE_COO_H_
#define INFINI_OPS_TORCH_TO_SPARSE_COO_H_

#include "base/to_sparse_coo.h"

namespace infini::ops {

template <>
class Operator<ToSparseCoo, Device::Type::kCpu, 8> : public ToSparseCoo {
 public:
  using ToSparseCoo::ToSparseCoo;

  static bool supports(const SparseTensor& input, const SparseTensor& out) {
    return input.device().index() == 0 &&
           input.layout() == TensorLayout::kCsr &&
           out.layout() == TensorLayout::kCoo;
  }

  static ValidationResult checked_call(const Handle& handle, const Config&,
                                       const SparseTensor& input,
                                       const SparseTensor& out);
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_TO_SPARSE_COO_H_
