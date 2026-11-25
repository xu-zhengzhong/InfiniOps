#ifndef INFINI_OPS_BASE_SPARSE_MM_H_
#define INFINI_OPS_BASE_SPARSE_MM_H_

#include <string>

#include "operator.h"

namespace infini::ops {

class SparseMm : public Operator<SparseMm> {
 public:
  SparseMm(const SparseTensor& mat1, const Tensor& mat2,
           const std::string& reduce, const Tensor& out) {
    (void)mat1;
    (void)mat2;
    (void)reduce;
    (void)out;
  }

  void operator()(const SparseTensor& mat1, const Tensor& mat2,
                  const std::string& reduce, Tensor out) const = delete;

  // Expose the checked capability diagnostic for sparse right operands.
  SparseMm(const SparseTensor& mat1, const SparseTensor& mat2,
           const std::string& reduce, const Tensor& out) = delete;

  void operator()(const SparseTensor& mat1, const SparseTensor& mat2,
                  const std::string& reduce, Tensor out) const = delete;

  static ValidationResult ValidateSparseSignature(const SparseTensor& mat1,
                                                  const Tensor& mat2,
                                                  const std::string& reduce,
                                                  const Tensor& out) {
    if (reduce != "sum") {
      return {ValidationCode::kUnsupportedCapability,
              ValidationComponent::kNone,
              {},
              2};
    }
    if (mat2.ndim() != 2) {
      return {ValidationCode::kUnsupportedCapability,
              ValidationComponent::kShape,
              {},
              1};
    }
    const auto shape = mat1.shape();
    if (mat2.size(0) != shape[1]) {
      return {
          ValidationCode::kInvalidMetadata, ValidationComponent::kShape, {}, 1};
    }
    if (out.ndim() != 2 || out.size(0) != shape[0] ||
        out.size(1) != mat2.size(1)) {
      return {
          ValidationCode::kInvalidMetadata, ValidationComponent::kShape, {}, 3};
    }
    return {};
  }

  static ValidationResult ValidateSparseSignature(const SparseTensor&,
                                                  const SparseTensor&,
                                                  const std::string&,
                                                  const Tensor&) {
    return {ValidationCode::kUnsupportedCapability,
            ValidationComponent::kLayout,
            {},
            1};
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_SPARSE_MM_H_
