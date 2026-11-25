#ifndef INFINI_OPS_BASE_TO_SPARSE_COO_H_
#define INFINI_OPS_BASE_TO_SPARSE_COO_H_

#include "operator.h"

namespace infini::ops {

class ToSparseCoo : public Operator<ToSparseCoo> {
 public:
  ToSparseCoo(const SparseTensor& input, const SparseTensor& out) {
    (void)input;
    (void)out;
  }

  void operator()(const SparseTensor& input,
                  const SparseTensor& out) const = delete;

  static ValidationResult ValidateSparseSignature(const SparseTensor& input,
                                                  const SparseTensor& out) {
    if (input.layout() != TensorLayout::kCsr)
      return {ValidationCode::kUnsupportedCapability,
              ValidationComponent::kLayout,
              {},
              0};
    if (out.layout() != TensorLayout::kCoo)
      return {ValidationCode::kUnsupportedCapability,
              ValidationComponent::kLayout,
              {},
              1};
    if (out.shape() != input.shape() || out.nnz() != input.nnz())
      return {
          ValidationCode::kInvalidMetadata, ValidationComponent::kShape, {}, 1};
    bool coalesced = false;
    (void)out.is_coalesced(coalesced);
    if (!coalesced)
      return {ValidationCode::kInvalidMetadata,
              ValidationComponent::kCoalesced,
              {},
              1};
    return {};
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_TO_SPARSE_COO_H_
