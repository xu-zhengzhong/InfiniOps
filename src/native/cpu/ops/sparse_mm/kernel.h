#ifndef INFINI_OPS_CPU_SPARSE_MM_KERNEL_H_
#define INFINI_OPS_CPU_SPARSE_MM_KERNEL_H_

#include "base/sparse_mm.h"
#include "native/cpu/sparse_matrix.h"

namespace infini::ops {

template <>
class Operator<SparseMm, Device::Type::kCpu, 0> : public SparseMm {
 public:
  using SparseMm::SparseMm;

  static bool supports(const SparseTensor& mat1, const Tensor&,
                       const std::string&, const Tensor&) {
    return native::cpu::SupportsSparseMatrix(mat1);
  }

  static ValidationResult checked_call(const Handle&, const Config&,
                                       const SparseTensor& mat1,
                                       const Tensor& mat2, const std::string&,
                                       Tensor out) {
    if (native::cpu::StorageOverlaps(mat2, out)) {
      return {ValidationCode::kAliasing, ValidationComponent::kValues, {}, 1};
    }
    const auto width = mat2.size(1);
    const auto* rhs = mat2.data();
    return native::cpu::AccumulateSparseEntries(
        mat1, out,
        [rhs, width](auto* dst, std::size_t row, std::size_t col, auto value) {
          const auto* rhs_row =
              static_cast<const decltype(value)*>(rhs) + col * width;
          auto* out_row = dst + row * width;
          for (std::size_t j = 0; j < width; ++j) {
            out_row[j] += value * rhs_row[j];
          }
        });
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CPU_SPARSE_MM_KERNEL_H_
