#include "sparse_output.h"

#include "common/sparse_buffers.h"

namespace infini::ops {
namespace {

using Code = ValidationCode;
using Component = ValidationComponent;

ValidationResult CheckBuffer(SparseBufferSpan buffer,
                             SparseBufferRequirement required,
                             Component component = Component::kNone,
                             bool workspace = false) {
  if (buffer.bytes < required.bytes)
    return {workspace ? Code::kInvalidWorkspace : Code::kInsufficientCapacity,
            component};
  if (!buffer.bytes) return {};
  if (!buffer.data)
    return {workspace ? Code::kInvalidWorkspace : Code::kNullData, component};
  const auto address = reinterpret_cast<std::uintptr_t>(buffer.data);
  if (buffer.bytes > std::numeric_limits<std::uintptr_t>::max() - address)
    return {Code::kIntegerOverflow, component};
  if (address % required.alignment)
    return {workspace ? Code::kInvalidWorkspace : Code::kInvalidMetadata,
            component};
  return {};
}

ValidationResult CheckResources(const SparseTensor& input, TensorLayout layout,
                                const SparseOutputRequirements& needs,
                                const SparseOutputBuffer& out,
                                SparseWorkspace workspace,
                                SparseBufferSpan slot) {
  if (out.layout != layout)
    return {Code::kUnsupportedLayout, Component::kLayout};
  if (out.shape != input.shape())
    return {Code::kInvalidMetadata, Component::kShape};
  if (out.dtype != input.dtype())
    return {Code::kInvalidMetadata, Component::kValues};
  if (out.device != input.device()) return {Code::kDeviceMismatch};
  if (out.index_dtype != DataType::kInt64) {
    return {layout == TensorLayout::kCsr && out.index_dtype == DataType::kInt32
                ? Code::kUnsupportedCapability
                : Code::kInvalidIndexType,
            layout == TensorLayout::kCsr ? Component::kCrowIndices
                                         : Component::kIndices};
  }
  if (layout == TensorLayout::kCoo && out.col_indices.bytes)
    return {Code::kInvalidMetadata, Component::kColIndices};
  if (workspace.device != out.device) return {Code::kInvalidWorkspace};
  auto status =
      CheckBuffer(out.indices, needs.indices,
                  layout == TensorLayout::kCoo ? Component::kIndices
                                               : Component::kCrowIndices);
  if (status)
    status =
        CheckBuffer(out.col_indices, needs.col_indices, Component::kColIndices);
  if (status)
    status = CheckBuffer(out.values, needs.values, Component::kValues);
  if (status)
    status =
        CheckBuffer(workspace.storage, needs.workspace, Component::kNone, true);
  if (status) status = CheckBuffer(slot, needs.result);
  if (!status) return status;
  return detail::CheckSparseWrites(
      input,
      {out.indices, out.col_indices, out.values, workspace.storage, slot});
}

}  // namespace

ValidationResult SparseOutputPlan::Execute(const SparseOutputBuffer& out,
                                           SparseWorkspace workspace,
                                           SparseResultMetadata* result,
                                           std::size_t result_bytes) noexcept {
  result_ = nullptr;
  output_.reset();
  if (!kernel_) return submission_ = {Code::kInvalidState};
  submission_ = CheckResources(*input_, layout_, requirements_, out, workspace,
                               {result, result_bytes});
  if (!submission_) return submission_;

  auto state = result->state.load();
  do {
    if (state == SparseCompletion::kPending)
      return submission_ = {Code::kInvalidState};
  } while (
      !result->state.compare_exchange_weak(state, SparseCompletion::kPending));

  generation_ = ++result->generation;
  result_ = result;
  output_ = out;
  std::size_t actual = 0;
  ValidationResult status;
  try {
    status = kernel_(stream_, *input_, out, workspace.storage.data, actual);
  } catch (const std::bad_alloc&) {
    status = {Code::kAllocationFailure};
  } catch (...) {
    status = {Code::kExecutionFailure};
  }
  if (status && actual > requirements_.nnz_upper_bound)
    status = {Code::kInsufficientCapacity};
  result->actual_nnz = status ? actual : 0;
  result->error = status;
  result->state.store(status ? SparseCompletion::kSuccess
                             : SparseCompletion::kFailed);
  return {};  // Submission succeeded. Finalize reports the completion status.
}

SparseFinalizeResult SparseOutputPlan::Finalize(
    std::optional<SparseTensor>& view) const {
  using State = SparseCompletion;
  view.reset();
  if (!submission_) return {State::kFailed, submission_};
  if (!result_ || !output_) return {State::kFailed, {Code::kInvalidState}};
  const auto state = result_->state.load();
  if (state == State::kPending) return {State::kPending, {}};
  if (generation_ != result_->generation)
    return {State::kFailed, {Code::kInvalidState}};
  if (state == State::kFailed) return {state, result_->error};

  const auto& out = *output_;
  const auto n = result_->actual_nnz;
  if (n > requirements_.nnz_upper_bound)
    return {State::kFailed, {Code::kInsufficientCapacity}};
  auto status = detail::MakeSparseOutputView(out, n, view);
  return {status ? State::kSuccess : State::kFailed, status};
}

ValidationResult detail::MakeSparseOutputView(
    const SparseOutputBuffer& out, std::size_t n,
    std::optional<SparseTensor>& view) {
  const Tensor values{out.values.data, std::array{n}, out.dtype, out.device};
  if (out.layout == TensorLayout::kCoo) {
    const Tensor indices{out.indices.data, std::array<std::size_t, 2>{2, n},
                         out.index_dtype, out.device};
    return FromRtValidation(
        SparseTensor::CreateCoo(out.shape, indices, values, view, true));
  }
  const Tensor crow{out.indices.data, std::array{out.shape[0] + 1},
                    out.index_dtype, out.device};
  const Tensor col{out.col_indices.data, std::array{n}, out.index_dtype,
                   out.device};
  return FromRtValidation(
      SparseTensor::CreateCsr(out.shape, crow, col, values, view));
}

}  // namespace infini::ops
