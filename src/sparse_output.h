#ifndef INFINI_OPS_SPARSE_OUTPUT_H_
#define INFINI_OPS_SPARSE_OUTPUT_H_

#include <atomic>

#include "operator.h"

namespace infini::ops {

struct SparseBufferSpan {
  void* data{};
  std::size_t bytes{};
};

struct SparseBufferRequirement {
  std::size_t bytes{};
  std::size_t alignment{1};
};

struct SparseOutputRequirements {
  std::size_t nnz_upper_bound{};
  SparseBufferRequirement indices, col_indices, values, workspace, result;
};

// The first index buffer holds COO coordinates or CSR row pointers.
struct SparseOutputBuffer {
  TensorLayout layout;
  SparseTensor::Shape shape;
  DataType dtype;
  DataType index_dtype;
  Device device;
  SparseBufferSpan indices, col_indices, values;
};

struct SparseWorkspace {
  SparseBufferSpan storage;
  Device device;
};

enum class SparseCapacityMode { kUpperBound, kCountFirst };

struct SparsePrepareOptions {
  SparseCapacityMode capacity_mode{SparseCapacityMode::kUpperBound};
  bool capture{false};
};

enum class SparseCompletion { kPending, kSuccess, kFailed };

struct [[nodiscard]] SparseFinalizeResult {
  SparseCompletion state;
  ValidationResult error;

  constexpr explicit operator bool() const {
    return state == SparseCompletion::kSuccess;
  }
};

// Caller-owned, naturally aligned host object. Inspect the payload only after
// completion and finish consuming it before allowing another execution.
struct SparseResultMetadata {
  std::atomic<SparseCompletion> state{SparseCompletion::kFailed};
  ValidationResult error{ValidationCode::kInvalidState};
  std::size_t actual_nnz{};
  std::size_t generation{};
};

namespace native::cpu {

struct SparseOutputProvider;

}  // namespace native::cpu

// Borrows the input until Execute completes. Each Execute invalidates this
// plan's previous publication. Serialize calls on a plan and keep its bound
// output buffers and result slot alive through Finalize and consumption.
class SparseOutputPlan {
 public:
  SparseOutputPlan() = default;

  SparseOutputPlan(const SparseOutputPlan&) = delete;

  SparseOutputPlan& operator=(const SparseOutputPlan&) = delete;

  const SparseOutputRequirements& requirements() const { return requirements_; }

  TensorLayout output_layout() const { return layout_; }

  ValidationResult Execute(
      const SparseOutputBuffer& out, SparseWorkspace workspace,
      SparseResultMetadata* result,
      std::size_t result_bytes = sizeof(SparseResultMetadata)) noexcept;

  SparseFinalizeResult Finalize(std::optional<SparseTensor>& view) const;

 private:
  friend class Coalesce;
  friend class ToSparseCsr;
  friend struct native::cpu::SparseOutputProvider;

  template <typename Op>
  ValidationResult Prepare(const Handle& handle, const Config& config,
                           const SparseTensor& input,
                           const SparsePrepareOptions& options) {
    input_.reset();
    output_.reset();
    result_ = nullptr;
    kernel_ = nullptr;
    requirements_ = {};
    submission_ = {ValidationCode::kInvalidState};
    if (options.capture ||
        options.capacity_mode != SparseCapacityMode::kUpperBound)
      return {ValidationCode::kUnsupportedCapability};
    if (input.layout() != TensorLayout::kCoo)
      return {ValidationCode::kUnsupportedCapability,
              ValidationComponent::kLayout,
              {},
              0};
    return Op::CheckedCall(handle, config, input, this);
  }

  using Kernel = ValidationResult (*)(const SparseTensor&,
                                      const SparseOutputBuffer&, void*,
                                      std::size_t&);
  Kernel kernel_{};
  std::optional<SparseTensor> input_;
  TensorLayout layout_{TensorLayout::kCoo};
  SparseOutputRequirements requirements_{};
  std::optional<SparseOutputBuffer> output_;
  SparseResultMetadata* result_{};
  std::size_t generation_{};
  ValidationResult submission_{ValidationCode::kInvalidState};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_SPARSE_OUTPUT_H_
