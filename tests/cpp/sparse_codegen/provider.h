#pragma once

#include "base/sparse_probe.h"

namespace infini::ops {

template <>
class Operator<SparseProbe, Device::Type::kCpu, 0> : public SparseProbe {
 public:
  using SparseProbe::SparseProbe;

  template <typename... Args>
  static bool supports(const Args&...) {
    return true;
  }

  static ValidationResult checked_call(
      const Handle&, const Config&,
      const std::optional<std::vector<SparseTensor>>& masks, const Tensor&) {
    if (!masks) return {ValidationCode::kInvalidMetadata};
    if (masks->empty()) return {ValidationCode::kInvalidStructure};
    return {};
  }

  static ValidationResult checked_call(const Handle&, const Config&,
                                       const SparseTensor&, double scale) {
    return scale == 2 ? ValidationResult{}
                      : ValidationResult{ValidationCode::kInvalidMetadata};
  }

  static ValidationResult checked_call(
      const Handle&, const Config&, const Tensor&,
      const std::vector<std::optional<SparseTensor>>& masks) {
    return masks.size() == 2 && !masks[0] && masks[1]
               ? ValidationResult{}
               : ValidationResult{ValidationCode::kInvalidStructure};
  }
};

}  // namespace infini::ops
