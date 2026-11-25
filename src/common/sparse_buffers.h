#ifndef INFINI_OPS_COMMON_SPARSE_BUFFERS_H_
#define INFINI_OPS_COMMON_SPARSE_BUFFERS_H_

#include "sparse_output.h"

namespace infini::ops::detail {

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
    const SparseTensor& input, std::initializer_list<SparseByteRange> writes) {
  const Tensor* indices = nullptr;
  const Tensor* columns = nullptr;
  const Tensor* values = nullptr;
  if (input.layout() == TensorLayout::kCoo) {
    (void)input.raw_indices(indices);
  } else {
    (void)input.crow_indices(indices);
    (void)input.col_indices(columns);
  }
  (void)input.raw_values(values);
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
    for (auto previous = writes.begin(); previous != w; ++previous) {
      if (Overlaps(*w, *previous)) return {ValidationCode::kAliasing};
    }
  }
  return {};
}

}  // namespace infini::ops::detail

#endif  // INFINI_OPS_COMMON_SPARSE_BUFFERS_H_
