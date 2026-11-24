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

  static ValidationResult checked_call(const Handle&, const Config&,
                                       const SparseTensor& input,
                                       const std::vector<int64_t>& addresses,
                                       const Tensor& out) {
    std::vector<int64_t> actual;
    const auto append = [&](const Tensor* tensor) {
      actual.push_back(reinterpret_cast<std::intptr_t>(tensor->data()));
    };
    const Tensor* component = nullptr;
    int state = -1;
    if (input.layout() == TensorLayout::kCoo) {
      bool coalesced = false;
      (void)input.is_coalesced(coalesced);
      state = coalesced;
      (void)input.raw_indices(component);
      append(component);
    } else {
      (void)input.crow_indices(component);
      append(component);
      (void)input.col_indices(component);
      append(component);
    }
    (void)input.raw_values(component);
    append(component);
    if (addresses != actual || out.numel() != 4) {
      return {ValidationCode::kInvalidMetadata};
    }
    const auto write = [&](auto* result) {
      result[0] = input.shape()[0];
      result[1] = input.shape()[1];
      result[2] = input.nnz();
      result[3] = state;
    };
    if (out.dtype() == DataType::kFloat64) {
      write(static_cast<double*>(const_cast<void*>(out.data())));
    } else {
      write(static_cast<float*>(const_cast<void*>(out.data())));
    }
    return {};
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
