#include <algorithm>
#include <numeric>
#include <tuple>

#include "native/cpu/sparse_output_provider.h"
#include "native/cpu/sparse_validation.h"

namespace infini::ops::native::cpu {
namespace {

using Code = ValidationCode;
using Component = ValidationComponent;
using Index = std::int64_t;

bool NeedsSort(const SparseTensor& input) {
  bool coalesced = false;
  (void)input.is_coalesced(coalesced);
  return !coalesced && input.nnz() > 1;
}

template <typename Value>
ValidationResult WriteCanonical(const SparseTensor& input,
                                const SparseOutputBuffer& out, void* workspace,
                                std::size_t& actual) {
  const Tensor *indices, *entries;
  (void)input.raw_indices(indices);
  (void)input.raw_values(entries);
  const auto* coords = static_cast<const Index*>(indices->data());
  const auto* values = static_cast<const Value*>(entries->data());
  const auto n = input.nnz();
  const auto key = [&](std::size_t p) {
    return std::tie(coords[p], coords[n + p]);
  };
  std::size_t* order = nullptr;
  if (NeedsSort(input)) {
    order = static_cast<std::size_t*>(workspace);
    std::iota(order, order + n, std::size_t{0});
    std::sort(order, order + n, [&](std::size_t a, std::size_t b) {
      // The position tie-break preserves input order within each sum.
      return std::tie(coords[a], coords[n + a], a) <
             std::tie(coords[b], coords[n + b], b);
    });
  }
  const auto position = [&](std::size_t i) { return order ? order[i] : i; };
  actual = n;
  if (order) {
    for (std::size_t i = 1; i < n; ++i) {
      if (key(order[i]) == key(order[i - 1])) --actual;
    }
  }
  const bool csr = out.layout == TensorLayout::kCsr;
  const auto indices_capacity = csr ? out.col_indices.bytes / sizeof(Index)
                                    : out.indices.bytes / (2 * sizeof(Index));
  if (actual > indices_capacity || actual > out.values.bytes / sizeof(Value))
    return {Code::kInsufficientCapacity};

  auto* first = static_cast<Index*>(out.indices.data);
  auto* cols = static_cast<Index*>(out.col_indices.data);
  auto* dst = static_cast<Value*>(out.values.data);
  if (csr) {
    std::fill_n(first, out.shape[0] + 1, Index{0});
  } else if (actual) {
    cols = first + actual;
  }
  std::size_t count = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto p = position(i);
    if (i && key(p) == key(position(i - 1))) {
      dst[count - 1] += values[p];
      continue;
    }
    if (csr) {
      ++first[coords[p] + 1];
    } else {
      first[count] = coords[p];
    }
    cols[count] = coords[n + p];
    dst[count++] = values[p];
  }
  if (csr) std::partial_sum(first, first + out.shape[0] + 1, first);
  return {};
}

ValidationResult ExecuteCpu(const SparseTensor& input,
                            const SparseOutputBuffer& out, void* workspace,
                            std::size_t& actual) {
  auto status = ValidateSparseIndices(input);
  if (!status) {
    status.argument = 0;
    return status;
  }
  if (input.dtype() == DataType::kFloat32)
    return WriteCanonical<float>(input, out, workspace, actual);
  return WriteCanonical<double>(input, out, workspace, actual);
}

}  // namespace

ValidationResult SparseOutputProvider::Prepare(const SparseTensor& input,
                                               TensorLayout layout,
                                               SparseOutputPlan* plan) {
  if (!plan) return {Code::kNullData};
  const auto n = input.nnz();
  const bool csr = layout == TensorLayout::kCsr;
  const auto rows = input.shape()[0];
  if (csr && rows >= std::numeric_limits<std::size_t>::max() / sizeof(Index))
    return {Code::kIntegerOverflow, Component::kCrowIndices};
  auto status = ValidateSparseIndices(input);
  if (!status) {
    status.argument = 0;
    return status;
  }
  // RT already checked the COO input's index and value byte counts.
  const auto value_size =
      input.dtype() == DataType::kFloat32 ? sizeof(float) : sizeof(double);
  const auto value_align =
      input.dtype() == DataType::kFloat32 ? alignof(float) : alignof(double);
  plan->requirements_ = {
      n,
      {(csr ? rows + 1 : 2 * n) * sizeof(Index), alignof(Index)},
      {csr ? n * sizeof(Index) : 0, alignof(Index)},
      {n * value_size, value_align},
      {NeedsSort(input) ? n * sizeof(std::size_t) : 0, alignof(std::size_t)},
      {sizeof(SparseResultMetadata), alignof(SparseResultMetadata)}};
  plan->input_.emplace(input);
  plan->layout_ = layout;
  plan->kernel_ = ExecuteCpu;
  return {};
}

}  // namespace infini::ops::native::cpu
