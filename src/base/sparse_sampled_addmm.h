#ifndef INFINI_OPS_BASE_SPARSE_SAMPLED_ADDMM_H_
#define INFINI_OPS_BASE_SPARSE_SAMPLED_ADDMM_H_

#include <tuple>

#include "operator.h"

namespace infini::ops {

class SparseSampledAddmm;

template <>
inline constexpr const char* detail::kDenseMigrationMessage<
    SparseSampledAddmm> =
    "The dense SparseSampledAddmm overload is deprecated. Use CSR input and "
    "out with SparseSampledAddmm::CheckedCall (infini.ops.sparse_sampled_addmm "
    "in Python).";

class SparseSampledAddmm : public Operator<SparseSampledAddmm> {
 public:
  SparseSampledAddmm(const SparseTensor& input, const Tensor& mat1,
                     const Tensor& mat2, double beta, double alpha,
                     const SparseTensor& out) {
    // Bindings use these argument names. The checked provider owns execution.
    std::ignore = std::tie(input, mat1, mat2, beta, alpha, out);
  }

  void operator()(const SparseTensor& input, const Tensor& mat1,
                  const Tensor& mat2, double beta, double alpha,
                  const SparseTensor& out) const = delete;

  static ValidationResult ValidateSparseSignature(const SparseTensor& input,
                                                  const Tensor& mat1,
                                                  const Tensor& mat2, double,
                                                  double,
                                                  const SparseTensor& out) {
    using Code = ValidationCode;
    using Component = ValidationComponent;
    if (input.layout() != TensorLayout::kCsr)
      return {Code::kUnsupportedCapability, Component::kLayout, {}, 0};
    if (out.layout() != TensorLayout::kCsr)
      return {Code::kUnsupportedCapability, Component::kLayout, {}, 5};
    const auto shape = input.shape();
    if (out.shape() != shape || out.nnz() != input.nnz())
      return {Code::kInvalidMetadata, Component::kShape, {}, 5};
    if (out.index_dtype() != input.index_dtype())
      return {Code::kInvalidIndexType, Component::kCrowIndices, {}, 5};
    if (mat1.ndim() != 2)
      return {Code::kUnsupportedCapability, Component::kShape, {}, 1};
    if (mat2.ndim() != 2)
      return {Code::kUnsupportedCapability, Component::kShape, {}, 2};
    if (mat1.size(0) != shape[0])
      return {Code::kInvalidMetadata, Component::kShape, {}, 1};
    if (mat2.size(0) != mat1.size(1) || mat2.size(1) != shape[1])
      return {Code::kInvalidMetadata, Component::kShape, {}, 2};
    return {};
  }

  /// \deprecated Removal of the dense overload requires a breaking change.
  [[deprecated("Use SparseSampledAddmm::CheckedCall with CSR input and out.")]]
  SparseSampledAddmm(const Tensor input, const Tensor mat1, const Tensor mat2,
                     const double beta, const double alpha, Tensor out) {
    std::ignore = std::tie(input, mat1, mat2, beta, alpha, out);
    detail::CheckDenseMigration<SparseSampledAddmm>();
  }

  [[deprecated("Use SparseSampledAddmm::CheckedCall with CSR input and out.")]]
  virtual void operator()(const Tensor input, const Tensor mat1,
                          const Tensor mat2, const double beta,
                          const double alpha, Tensor out) const {
    std::ignore = std::tie(input, mat1, mat2, beta, alpha, out);
    detail::CheckDenseMigration<SparseSampledAddmm>();
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_SPARSE_SAMPLED_ADDMM_H_
