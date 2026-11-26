#ifndef INFINI_OPS_TORCH_SPARSE_SAMPLED_ADDMM_H_
#define INFINI_OPS_TORCH_SPARSE_SAMPLED_ADDMM_H_

#include "base/sparse_sampled_addmm.h"

namespace infini::ops {

template <>
class Operator<SparseSampledAddmm, Device::Type::kCpu, 8>
    : public SparseSampledAddmm {
 public:
  using SparseSampledAddmm::SparseSampledAddmm;

  static bool supports(const SparseTensor& input, const Tensor&, const Tensor&,
                       double, double, const SparseTensor& out) {
    return input.device().index() == 0 &&
           input.layout() == TensorLayout::kCsr &&
           out.layout() == TensorLayout::kCsr;
  }

  static ValidationResult checked_call(const Handle& handle, const Config&,
                                       const SparseTensor& input,
                                       const Tensor& mat1, const Tensor& mat2,
                                       double beta, double alpha,
                                       const SparseTensor& out);
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_SPARSE_SAMPLED_ADDMM_H_
