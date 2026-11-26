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

ValidationResult ExecuteCpu(void*, const SparseTensor& input,
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
  const SparseBufferRequirement workspace{
      NeedsSort(input) ? input.nnz() * sizeof(std::size_t) : 0,
      alignof(std::size_t)};
  return PrepareWithKernel(input, layout, plan, ExecuteCpu, nullptr, workspace,
                           false);
}

ValidationResult SparseOutputProvider::PrepareWithKernel(
    const SparseTensor& input, TensorLayout layout, SparseOutputPlan* plan,
    SparseOutputPlan::Kernel kernel, void* stream,
    SparseBufferRequirement workspace, bool copies_result) {
  if (!plan) return {Code::kNullData};
  const auto n = input.nnz();
  const bool csr = layout == TensorLayout::kCsr;
  const auto rows = input.shape()[0];
  if (csr && rows >= std::numeric_limits<std::size_t>::max() / sizeof(Index))
    return {Code::kIntegerOverflow, Component::kCrowIndices};
  // RT already checked the COO input's index and value byte counts.
  const auto value_size =
      input.dtype() == DataType::kFloat32 ? sizeof(float) : sizeof(double);
  const auto value_align =
      input.dtype() == DataType::kFloat32 ? alignof(float) : alignof(double);
  SparseOutputRequirements requirements{
      n,
      {(csr ? rows + 1 : 2 * n) * sizeof(Index), alignof(Index)},
      {csr ? n * sizeof(Index) : 0, alignof(Index)},
      {n * value_size, value_align},
      workspace,
      {sizeof(SparseResultMetadata), alignof(SparseResultMetadata)}};
  if (copies_result) {
    for (auto part : {requirements.indices, requirements.col_indices,
                      requirements.values}) {
      if (part.bytes > std::numeric_limits<std::size_t>::max() -
                           requirements.result_copy_bytes_upper_bound)
        return {Code::kIntegerOverflow};
      requirements.result_copy_bytes_upper_bound += part.bytes;
    }
  }
  auto status = ValidateSparseIndices(input);
  if (!status) {
    status.argument = 0;
    return status;
  }
  plan->input_.emplace(input);
  plan->requirements_ = requirements;
  plan->layout_ = layout;
  plan->stream_ = stream;
  plan->kernel_ = kernel;
  return {};
}

}  // namespace infini::ops::native::cpu
