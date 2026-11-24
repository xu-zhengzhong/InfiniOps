#pragma once

#include "operator.h"

namespace infini::ops {

class SparseProbe : public Operator<SparseProbe> {
 public:
  SparseProbe(const Tensor& input, double scale) {}
  SparseProbe(const ::infini::rt::SparseTensorView& input, double scale) {}
  SparseProbe(const SparseTensor& input, const std::vector<int64_t>& addresses,
              Tensor out) {}
  SparseProbe(const SparseTensor& input, const SparseTensor& out) {}
  SparseProbe(const Tensor& input,
              const std::vector<std::optional<SparseTensor>>& masks) {}
  SparseProbe(const Tensor& input,
              const std::vector<std::optional<Tensor>>& masks) {}

  SparseProbe(const std::optional<std::vector<SparseTensor>>& masks,
              const Tensor& input) {}

  void operator()(const std::optional<std::vector<SparseTensor>>& masks,
                  const Tensor& input) const {}
  void operator()(const Tensor& input, double scale) const {}
  void operator()(const ::infini::rt::SparseTensorView& input,
                  double scale) const {}
  void operator()(const SparseTensor& input,
                  const std::vector<int64_t>& addresses, Tensor out) const {}
  void operator()(const SparseTensor& input, const SparseTensor& out) const {}
  void operator()(const Tensor& input,
                  const std::vector<std::optional<SparseTensor>>& masks) const {
  }
  void operator()(const Tensor& input,
                  const std::vector<std::optional<Tensor>>& masks) const {}
};

}  // namespace infini::ops
