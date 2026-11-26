#include <infini/ops.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>

namespace {

using namespace infini::ops;
using Code = ValidationCode;
using Index = std::int64_t;
const Device cpu{Device::Type::kCpu};
const Handle handle;

#define CHECK(...)                                           \
  do {                                                       \
    if (!(__VA_ARGS__)) {                                    \
      std::cerr << __LINE__ << ": " << #__VA_ARGS__ << '\n'; \
      return false;                                          \
    }                                                        \
  } while (false)

bool CheckDynamic(const Config& config) {
  Index coords[]{2, 1, 0, 3, 1, 0, 2, 2, 1, 0, 1, 1, 3, 2, 0, 1, 1, 1, 0, 3};
  double values[]{1, 2, 3, 4, 5, 6, -1, 0, 2, 1};
  const Tensor indices{coords, {2, 10}, DataType::kInt64, cpu};
  const Tensor entries{values, {10}, DataType::kFloat64, cpu};
  std::optional<SparseTensor> input, view;
  CHECK(SparseTensor::CreateCoo({5, 6}, indices, entries, input));
  std::array<Index, 22> coord_out;
  std::array<double, 12> value_out;
  std::array<std::size_t, 10> work;
  SparseWorkspace workspace{{work.data(), sizeof(work)}, cpu};
  SparseOutputBuffer out{TensorLayout::kCoo,
                         {5, 6},
                         DataType::kFloat64,
                         DataType::kInt64,
                         cpu,
                         {coord_out.data() + 1, 20 * sizeof(Index)},
                         {},
                         {value_out.data() + 1, 10 * sizeof(double)}};
  SparseResultMetadata slot;
  SparseOutputPlan plan;
  CHECK(plan.Execute(out, workspace, &slot).code == Code::kInvalidState);
  CHECK(plan.Finalize(view).error.code == Code::kInvalidState && !view);
  CHECK(Coalesce::Prepare(handle, config, *input, {}, plan));
  const auto& req = plan.requirements();
  CHECK(req.nnz_upper_bound == 10 && req.indices.bytes == 160 &&
        req.values.bytes == 80);
  const bool aten = config.implementation_index() == 8;
  CHECK(req.result_copy_bytes_upper_bound == (aten ? 240 : 0));
  if (aten) {
    CHECK(req.workspace.bytes == 0);
  } else {
    CHECK(req.workspace.bytes == sizeof(work) &&
          req.workspace.alignment == alignof(std::size_t));
  }
  CHECK(req.result.bytes == sizeof(slot) &&
        req.result.alignment == alignof(SparseResultMetadata));
  coord_out.fill(-7);
  value_out.fill(-7);
  CHECK(plan.Execute(out, workspace, &slot));
  CHECK(slot.state == SparseCompletion::kSuccess && slot.actual_nnz == 6);
  CHECK(plan.Finalize(view) && view->nnz() == 6);
  const Tensor *result_indices, *result_values;
  CHECK(view->indices(result_indices) && view->values(result_values));
  CHECK(result_indices->data() == coord_out.data() + 1 &&
        result_indices->stride(0) == 6);
  CHECK(result_values->data() == value_out.data() + 1);
  const Index packed[]{0, 0, 1, 1, 2, 3, 1, 3, 0, 1, 1, 2};
  const double sums[]{6, 4, 7, 2, 0, 4};
  CHECK(
      std::equal(std::begin(packed), std::end(packed), coord_out.begin() + 1));
  CHECK(std::equal(std::begin(sums), std::end(sums), value_out.begin() + 1));
  CHECK(coord_out.front() == -7 &&
        std::all_of(coord_out.begin() + 13, coord_out.end(),
                    [](auto x) { return x == -7; }));
  CHECK(value_out.front() == -7 &&
        std::all_of(value_out.begin() + 7, value_out.end(),
                    [](auto x) { return x == -7; }));
  CHECK(plan.Finalize(view) && view->nnz() == 6);
  slot.actual_nnz = 11;
  CHECK(plan.Finalize(view).error.code == Code::kInsufficientCapacity && !view);
  slot.actual_nnz = 6;
  slot.state = SparseCompletion::kPending;
  const auto pending = plan.Finalize(view);
  CHECK(pending.state == SparseCompletion::kPending && pending.error.ok() &&
        !pending && !view);
  CHECK(plan.Execute(out, workspace, &slot).code == Code::kInvalidState);
  slot.state = SparseCompletion::kSuccess;

  coord_out.fill(19);
  value_out.fill(19);
  const auto rejected = [&](SparseOutputBuffer dst, SparseWorkspace ws,
                            SparseResultMetadata* result, std::size_t size,
                            Code code) {
    return plan.Execute(dst, ws, result, size).code == code &&
           plan.Finalize(view).error.code == code && !view &&
           std::all_of(coord_out.begin(), coord_out.end(),
                       [](auto x) { return x == 19; }) &&
           std::all_of(value_out.begin(), value_out.end(),
                       [](auto x) { return x == 19; });
  };
  auto bad = out;
  bad.values.bytes -= sizeof(double);
  CHECK(rejected(bad, workspace, &slot, sizeof(slot),
                 Code::kInsufficientCapacity));
  bad.indices = {};
  CHECK(rejected(bad, workspace, &slot, sizeof(slot),
                 Code::kInsufficientCapacity));
  bad = out;
  bad.indices.data = nullptr;
  CHECK(rejected(bad, workspace, &slot, sizeof(slot), Code::kNullData));
  bad.indices.data = reinterpret_cast<char*>(coord_out.data()) + 1;
  CHECK(rejected(bad, workspace, &slot, sizeof(slot), Code::kInvalidMetadata));
  bad.indices.data =
      reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - 7);
  CHECK(rejected(bad, workspace, &slot, sizeof(slot), Code::kIntegerOverflow));
  bad = out;
  bad.values.data = coords;
  CHECK(rejected(bad, workspace, &slot, sizeof(slot), Code::kAliasing));
  bad.values.data = coord_out.data() + 12;
  CHECK(rejected(bad, workspace, &slot, sizeof(slot), Code::kAliasing));
  CHECK(rejected(out, {{values, sizeof(values)}, cpu}, &slot, sizeof(slot),
                 Code::kAliasing));
  CHECK(rejected(out, {{out.values.data, out.values.bytes}, cpu}, &slot,
                 sizeof(slot), Code::kAliasing));
  CHECK(rejected(out, workspace,
                 reinterpret_cast<SparseResultMetadata*>(out.indices.data),
                 sizeof(slot), Code::kAliasing));
  CHECK(rejected(out, workspace,
                 reinterpret_cast<SparseResultMetadata*>(work.data()),
                 sizeof(slot), Code::kAliasing));
  CHECK(rejected(out, workspace, nullptr, sizeof(slot), Code::kNullData));
  CHECK(rejected(out, workspace, &slot, sizeof(slot) - 1,
                 Code::kInsufficientCapacity));
  auto* offset = reinterpret_cast<char*>(coord_out.data()) + 1;
  CHECK(rejected(out, workspace,
                 reinterpret_cast<SparseResultMetadata*>(offset), sizeof(slot),
                 Code::kInvalidMetadata));
  if (req.workspace.bytes) {
    CHECK(rejected(out, {{offset, sizeof(work)}, cpu}, &slot, sizeof(slot),
                   Code::kInvalidWorkspace));
    CHECK(
        rejected(out, {{}, cpu}, &slot, sizeof(slot), Code::kInvalidWorkspace));
  }
  CHECK(rejected(out, {workspace.storage, Device{Device::Type::kCpu, 1}}, &slot,
                 sizeof(slot), Code::kInvalidWorkspace));
  bad = out;
  bad.device = Device{Device::Type::kCpu, 1};
  CHECK(rejected(bad, workspace, &slot, sizeof(slot), Code::kDeviceMismatch));

  // Raw callers must keep the input stable. A changed invalid index still
  // fails completion without an unsafe write or a published view.
  coords[0] = -1;
  CHECK(plan.Execute(out, workspace, &slot));
  CHECK(slot.state == SparseCompletion::kFailed);
  const auto failed = plan.Finalize(view);
  CHECK(failed.state == SparseCompletion::kFailed &&
        failed.error.code == Code::kIndexOutOfBounds && !view);
  CHECK(slot.error.position == 0 && coord_out[1] == 19 && value_out[1] == 19);
  coords[0] = 2;
  CHECK(plan.Execute(out, workspace, &slot) && plan.Finalize(view));
  view.reset();

  SparseOutputPlan csr_plan;
  CHECK(ToSparseCsr::Prepare(handle, config, *input, {}, csr_plan));
  CHECK(csr_plan.requirements().indices.bytes == 6 * sizeof(Index));
  std::array<Index, 10> columns;
  out.layout = TensorLayout::kCsr;
  out.col_indices = {columns.data(), sizeof(columns)};
  out.index_dtype = DataType::kInt32;
  CHECK(csr_plan.Execute(out, workspace, &slot).code ==
        Code::kUnsupportedCapability);
  out.index_dtype = DataType::kInt64;
  CHECK(csr_plan.Execute(out, workspace, &slot) && csr_plan.Finalize(view));
  const Index crow[]{0, 2, 4, 5, 6, 6};
  CHECK(std::equal(std::begin(crow), std::end(crow), coord_out.begin() + 1));
  CHECK(std::equal(std::begin(packed) + 6, std::end(packed), columns.begin()));
  CHECK(plan.Finalize(view).error.code == Code::kInvalidState && !view);
  struct RequiresConfigAwareProvider : Config {
    bool requires_config_aware_provider() const override { return true; }
  };
  CHECK(
      Coalesce::Prepare(handle, RequiresConfigAwareProvider{}, *input, {}, plan)
          .code == Code::kNoProvider);
  Config wrong;
  wrong.set_implementation_index(31);
  CHECK(Coalesce::Prepare(handle, wrong, *input, {}, plan).code ==
        Code::kUnsupportedImplementation);
  CHECK(plan.Execute(out, workspace, &slot).code == Code::kInvalidState);
  return true;
}

bool CheckBoundsAndFixedOutput(const Config& config) {
  std::optional<SparseTensor> input, out;
  const Tensor no_indices{nullptr, {2, 0}, DataType::kInt64, cpu};
  const Tensor no_values{nullptr, {0}, DataType::kFloat32, cpu};
  const auto rows = std::numeric_limits<std::size_t>::max() / sizeof(Index);
  CHECK(SparseTensor::CreateCoo({rows, 0}, no_indices, no_values, input));
  SparseOutputPlan plan;
  CHECK(ToSparseCsr::Prepare(handle, config, *input, {}, plan).code ==
        Code::kIntegerOverflow);
  CHECK(Coalesce::Prepare(handle, config, *input, {}, plan));
  SparseResultMetadata result;
  CHECK(plan.Execute({TensorLayout::kCoo,
                      input->shape(),
                      input->dtype(),
                      DataType::kInt64,
                      cpu,
                      {},
                      {},
                      {}},
                     {{}, cpu}, &result));
  CHECK(plan.Finalize(out) && out->nnz() == 0);
  if (config.implementation_index() == 8 && sizeof(std::size_t) == 8) {
    // This logical product fits the RT view but exceeds ATen's signed numel.
    const auto rows = static_cast<std::size_t>(std::uint64_t{1} << 32);
    CHECK(SparseTensor::CreateCoo({rows, std::size_t{1} << 31}, no_indices,
                                  no_values, input));
    CHECK(Coalesce::Prepare(handle, config, *input, {}, plan).code ==
          Code::kIntegerOverflow);
    CHECK(plan.Finalize(out).error.code == Code::kInvalidState && !out);

    // The component byte counts fit individually. Reject their sum before an
    // index scan, so this metadata boundary needs no large allocation.
    const auto n = std::numeric_limits<std::size_t>::max() /
                       (2 * sizeof(Index) + sizeof(double)) +
                   1;
    Index index{};
    double value{};
    CHECK(
        SparseTensor::CreateCoo({1, 1}, {&index, {2, n}, DataType::kInt64, cpu},
                                {&value, {n}, DataType::kFloat64, cpu}, input));
    CHECK(Coalesce::Prepare(handle, config, *input, {}, plan).code ==
          Code::kIntegerOverflow);
  }

  std::int32_t rows32[]{0, 2};
  std::int32_t cols32[]{0, 2};
  float vals[]{0, 3};
  CHECK(SparseTensor::CreateCsr({1, 3}, {rows32, {2}, DataType::kInt32, cpu},
                                {cols32, {2}, DataType::kInt32, cpu},
                                {vals, {2}, DataType::kFloat32, cpu}, input));
  Index coords[]{-17, -17, -17, -17};
  float copied[]{-17, -17};
  const Tensor ids{coords, {2, 2}, DataType::kInt64, cpu};
  const Tensor values{copied, {2}, DataType::kFloat32, cpu};
  CHECK(SparseTensor::CreateCoo(input->shape(), ids, values, out, false));
  CHECK(ToSparseCoo::CheckedCall(handle, config, *input, *out).code ==
        Code::kInvalidMetadata);
  CHECK(SparseTensor::CreateCoo(input->shape(), ids,
                                {coords + 2, {2}, DataType::kFloat32, cpu}, out,
                                true));
  CHECK(ToSparseCoo::CheckedCall(handle, config, *input, *out).code ==
        Code::kAliasing);
  CHECK(SparseTensor::CreateCoo(
      input->shape(), ids, {cols32, {2}, DataType::kFloat32, cpu}, out, true));
  CHECK(ToSparseCoo::CheckedCall(handle, config, *input, *out).code ==
        Code::kAliasing);
  CHECK(SparseTensor::CreateCoo(input->shape(), ids, values, out, true));
  rows32[1] = 17;
  CHECK(ToSparseCoo::CheckedCall(handle, config, *input, *out).code ==
        Code::kInvalidStructure);
  CHECK(copied[0] == -17 && coords[0] == -17);
  rows32[1] = 2;
  CHECK(ToSparseCoo::CheckedCall(handle, config, *input, *out));
  CHECK(coords[0] == 0 && coords[1] == 0 && coords[2] == 0 && coords[3] == 2);
  CHECK(copied[0] == 0 && copied[1] == 3);
  return true;
}

}  // namespace

int main() {
  const auto slots = Coalesce::active_implementation_indices(cpu.type());
  if (slots.empty()) return 1;
  for (const auto slot : slots) {
    Config config;
    config.set_implementation_index(slot);
    if (!CheckDynamic(config) || !CheckBoundsAndFixedOutput(config)) return 1;
  }
  return 0;
}
