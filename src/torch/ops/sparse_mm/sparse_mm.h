#ifndef INFINI_OPS_TORCH_SPARSE_MM_H_
#define INFINI_OPS_TORCH_SPARSE_MM_H_

#include "base/sparse_mm.h"

namespace infini::ops {

template <>
class Operator<SparseMm, Device::Type::kCpu, 8> : public SparseMm {
 public:
  using SparseMm::SparseMm;

  static bool supports(const SparseTensor& input, const Tensor&,
                       const std::string&, const Tensor&) {
    return input.device().index() == 0 &&
           (input.layout() == TensorLayout::kCoo ||
            input.layout() == TensorLayout::kCsr);
  }

  static ValidationResult checked_call(const Handle& handle, const Config&,
                                       const SparseTensor& input,
                                       const Tensor& mat2,
                                       const std::string& reduce,
                                       const Tensor& out);
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_SPARSE_MM_H_
