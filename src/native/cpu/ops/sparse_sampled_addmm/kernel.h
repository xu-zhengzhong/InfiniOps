#ifndef INFINI_OPS_CPU_SPARSE_SAMPLED_ADDMM_KERNEL_H_
#define INFINI_OPS_CPU_SPARSE_SAMPLED_ADDMM_KERNEL_H_

#include <cstring>

#include "base/sparse_sampled_addmm.h"
#include "native/cpu/sparse_matrix.h"

namespace infini::ops {

template <>
class Operator<SparseSampledAddmm, Device::Type::kCpu, 0>
    : public SparseSampledAddmm {
 public:
  using SparseSampledAddmm::SparseSampledAddmm;

  static bool supports(const SparseTensor& input, const Tensor&, const Tensor&,
                       double, double, const SparseTensor& out) {
    return input.layout() == TensorLayout::kCsr &&
           out.layout() == TensorLayout::kCsr;
  }

  static ValidationResult checked_call(const Handle&, const Config&,
                                       const SparseTensor& input,
                                       const Tensor& mat1, const Tensor& mat2,
                                       double beta, double alpha,
                                       const SparseTensor& out) {
    const Tensor *out_crow, *out_col, *out_values;
    (void)out.crow_indices(out_crow);
    (void)out.col_indices(out_col);
    (void)out.values(out_values);
    auto status = detail::CheckSparseWrites(
        input, {*out_crow, *out_col, *out_values}, {mat1, mat2}, 5);
    if (!status) return status;
    status = native::cpu::ValidateSparseIndices(input);
    if (!status) {
      status.argument = 0;
      return status;
    }

    const Tensor *crow, *col, *values;
    (void)input.crow_indices(crow);
    (void)input.col_indices(col);
    (void)input.values(values);
    // Even an empty CSR pattern has a row pointer of length M + 1.
    std::memcpy(Tensor{*out_crow}.data(), crow->data(),
                crow->numel() * crow->element_size());
    if (input.nnz() == 0) return {};
    std::memcpy(Tensor{*out_col}.data(), col->data(),
                col->numel() * col->element_size());

    const auto inner = mat1.size(1);
    const auto width = mat2.size(1);
    DispatchFunc<Device::Type::kCpu, FloatTypes>(
        input.dtype(),
        [&](auto tag) {
          using Value = typename decltype(tag)::type;
          const auto b = static_cast<Value>(beta);
          const auto* src = static_cast<const Value*>(values->data());
          auto* dst = static_cast<Value*>(Tensor{*out_values}.data());
          if (inner == 0) {
            for (std::size_t p = 0; p < input.nnz(); ++p) dst[p] = b * src[p];
            return;
          }
          const auto a = static_cast<Value>(alpha);
          const auto* lhs = static_cast<const Value*>(mat1.data());
          const auto* rhs = static_cast<const Value*>(mat2.data());
          native::cpu::detail::VisitSparseEntries(
              *crow, col,
              [&](std::size_t row, std::size_t column, std::size_t p) {
                Value dot = 0;
                for (std::size_t k = 0; k < inner; ++k) {
                  dot += lhs[row * inner + k] * rhs[k * width + column];
                }
                dst[p] = a * dot + b * src[p];
              });
        },
        "CPU CSR sampled addmm values");
    return {};
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CPU_SPARSE_SAMPLED_ADDMM_KERNEL_H_
