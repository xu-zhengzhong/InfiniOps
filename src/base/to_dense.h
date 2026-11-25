#ifndef INFINI_OPS_BASE_TO_DENSE_H_
#define INFINI_OPS_BASE_TO_DENSE_H_

#include "operator.h"

namespace infini::ops {

class ToDense : public Operator<ToDense> {
 public:
  ToDense(const SparseTensor& input, const Tensor& out) {
    (void)input;
    (void)out;
  }

  // Sparse execution goes through CheckedCall. The signature also defines the
  // generated Python function and the installed C++ factory overload.
  void operator()(const SparseTensor& input, Tensor out) const = delete;

  static ValidationResult ValidateSparseSignature(const SparseTensor& input,
                                                  const Tensor& out) {
    if (out.shape() != input.shape()) {
      return {
          ValidationCode::kInvalidMetadata, ValidationComponent::kShape, {}, 1};
    }
    return {};
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_TO_DENSE_H_
