#ifndef INFINI_OPS_COMMON_SPARSE_BUFFERS_H_
#define INFINI_OPS_COMMON_SPARSE_BUFFERS_H_

#include "sparse_output.h"

namespace infini::ops::detail {

struct SparseOutputProvider {
  using Inspect = ValidationResult (*)(void*, const SparseTensor&,
                                       SparseBufferRequirement&);
  static ValidationResult Bind(const SparseTensor&, TensorLayout,
                               SparseOutputPlan*, SparseOutputPlan::Kernel,
                               void* stream, SparseBufferRequirement workspace,
                               bool copies_result, Inspect inspect);
};

// The caller has checked the component capacities for this many entries.
ValidationResult MakeSparseOutputView(const SparseOutputBuffer& out,
                                      std::size_t nnz,
                                      std::optional<SparseTensor>& view);

struct SparseComponents {
  explicit SparseComponents(const SparseTensor& tensor) {
    if (tensor.layout() == TensorLayout::kCoo) {
      (void)tensor.raw_indices(indices);
    } else {
      (void)tensor.crow_indices(indices);
      (void)tensor.col_indices(columns);
    }
    (void)tensor.raw_values(values);
  }

  const Tensor* indices{};
  const Tensor* columns{};
  const Tensor* values{};
};

// All ranges passed to the alias check already have validated endpoints.
struct SparseByteRange {
  SparseByteRange(const Tensor& t)
      : data(t.data()), bytes(t.numel() * t.element_size()) {}
  SparseByteRange(SparseBufferSpan span) : data(span.data), bytes(span.bytes) {}
  const void* data;
  std::size_t bytes;
};

inline bool Overlaps(SparseByteRange a, SparseByteRange b) {
  const auto first = reinterpret_cast<std::uintptr_t>(a.data);
  const auto second = reinterpret_cast<std::uintptr_t>(b.data);
  return a.bytes && b.bytes && first < second + b.bytes &&
         second < first + a.bytes;
}

inline ValidationResult CheckSparseWrites(
    const SparseTensor& input, std::initializer_list<SparseByteRange> writes,
    std::initializer_list<SparseByteRange> dense_inputs = {},
    std::optional<std::size_t> out_argument = {}) {
  const auto [indices, columns, values] = SparseComponents(input);
  for (auto w = writes.begin(); w != writes.end(); ++w) {
    if (Overlaps(*w, *indices))
      return {ValidationCode::kAliasing,
              columns ? ValidationComponent::kCrowIndices
                      : ValidationComponent::kIndices,
              {},
              0};
    if (columns && Overlaps(*w, *columns))
      return {
          ValidationCode::kAliasing, ValidationComponent::kColIndices, {}, 0};
    if (Overlaps(*w, *values))
      return {ValidationCode::kAliasing, ValidationComponent::kValues, {}, 0};
    std::size_t argument = 1;  // Dense inputs follow the sparse input.
    for (auto dense : dense_inputs) {
      if (Overlaps(*w, dense))
        return {ValidationCode::kAliasing,
                ValidationComponent::kValues,
                {},
                argument};
      ++argument;
    }
    for (auto previous = writes.begin(); previous != w; ++previous) {
      if (Overlaps(*w, *previous))
        return {ValidationCode::kAliasing,
                ValidationComponent::kNone,
                {},
                out_argument};
    }
  }
  return {};
}

}  // namespace infini::ops::detail

#endif  // INFINI_OPS_COMMON_SPARSE_BUFFERS_H_
