#ifndef INFINI_OPS_NVIDIA_SPARSE_MM_KERNEL_H_
#define INFINI_OPS_NVIDIA_SPARSE_MM_KERNEL_H_

#include "base/sparse_mm.h"

namespace infini::ops {

template <>
class Operator<SparseMm, Device::Type::kNvidia, 0> : public SparseMm {
 public:
  using SparseMm::SparseMm;

  static bool supports(const SparseTensor& input, const Tensor&,
                       const std::string&, const Tensor&) {
    return input.layout() == TensorLayout::kCsr ||
           input.layout() == TensorLayout::kCoo;
  }

  static ValidationResult checked_call(const Handle& handle, const Config&,
                                       const SparseTensor& input,
                                       const Tensor& mat2,
                                       const std::string& reduce,
                                       const Tensor& out);
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_SPARSE_MM_KERNEL_H_
