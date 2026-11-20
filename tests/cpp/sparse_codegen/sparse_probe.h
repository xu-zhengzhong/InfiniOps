#pragma once

#include "operator.h"

namespace infini::ops {

class SparseProbe : public Operator<SparseProbe> {
 public:
  SparseProbe(const Tensor& input, double scale) {}
  SparseProbe(const ::infini::rt::SparseTensorView& input, double scale) {}
  SparseProbe(const Tensor& input,
              const std::vector<std::optional<SparseTensor>>& masks) {}

  SparseProbe(const std::optional<std::vector<SparseTensor>>& masks,
              const Tensor& input) {}

  void operator()(const std::optional<std::vector<SparseTensor>>& masks,
                  const Tensor& input) const {}
  void operator()(const Tensor& input, double scale) const {}
  void operator()(const ::infini::rt::SparseTensorView& input,
                  double scale) const {}
  void operator()(const Tensor& input,
                  const std::vector<std::optional<SparseTensor>>& masks) const {
  }
};

}  // namespace infini::ops
