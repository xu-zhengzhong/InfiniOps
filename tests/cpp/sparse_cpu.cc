#include <infini/ops.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>

namespace {

using namespace infini::ops;
using Code = ValidationCode;
using Component = ValidationComponent;

const Device cpu{Device::Type::kCpu};
const Config config;
const Handle handle;

#define CHECK(...)                                           \
  do {                                                       \
    if (!(__VA_ARGS__)) {                                    \
      std::cerr << __LINE__ << ": " << #__VA_ARGS__ << '\n'; \
      return false;                                          \
    }                                                        \
  } while (false)

bool Matches(ValidationResult r, Code code, std::size_t argument,
             Component component, std::optional<std::size_t> position = {}) {
  return r.code == code && r.argument == argument && r.component == component &&
         r.position == position;
}

bool CheckCoo() {
  std::int64_t coords[]{1, 0, 1, 0, 2, 0};
  double values[]{1, 5, 2};
  const Tensor indices{coords, {2, 3}, DataType::kInt64, cpu};
  const Tensor entries{values, {3}, DataType::kFloat64, cpu};
  std::optional<SparseTensor> mat;
  CHECK(SparseTensor::CreateCoo({2, 3}, indices, entries, mat));
  std::array<double, 8> guarded;
  guarded.fill(-7);
  const Tensor out{guarded.data() + 1, {2, 3}, DataType::kFloat64, cpu};
  double right[]{1, -1, 2, -2, 3, -3};
  const Tensor rhs{right, {3, 2}, DataType::kFloat64, cpu};
  std::array<double, 4> product;
  const Tensor mm_out{product.data(), {2, 2}, DataType::kFloat64, cpu};

  CHECK(ToDense::CheckedCall(handle, config, *mat, out));
  CHECK((guarded == std::array<double, 8>{-7, 0, 0, 5, 3, 0, 0, -7}));
  CHECK(SparseMm::CheckedCall(handle, config, *mat, rhs, "sum", mm_out));
  CHECK((product == std::array<double, 4>{15, -15, 3, -3}));
  coords[2] = 0;
  values[1] = -5;
  CHECK(ToDense::CheckedCall(handle, config, *mat, out));
  CHECK((guarded == std::array<double, 8>{-7, 2, 0, -5, 1, 0, 0, -7}));

  guarded.fill(23);
  product.fill(23);
  coords[4] = 3;
  CHECK(ToDense::CanMake(config, *mat, out) && ToDense::Make(*mat, out));
  CHECK(Matches(ToDense::CheckedCall(handle, config, *mat, out),
                Code::kIndexOutOfBounds, 0, Component::kIndices, 4));
  // A zero-width product still checks the input.
  const Tensor rhs_empty{nullptr, {3, 0}, DataType::kFloat64, cpu};
  const Tensor out_empty{nullptr, {2, 0}, DataType::kFloat64, cpu};
  CHECK(Matches(
      SparseMm::CheckedCall(handle, config, *mat, rhs_empty, "sum", out_empty),
      Code::kIndexOutOfBounds, 0, Component::kIndices, 4));
  coords[4] = 2;
  coords[0] = -1;
  CHECK(Matches(ToDense::CheckedCall(handle, config, *mat, out),
                Code::kIndexOutOfBounds, 0, Component::kIndices, 0));
  coords[0] = 1;
  CHECK(SparseTensor::CreateCoo({2, 3}, indices, entries, mat, true));
  CHECK(Matches(SparseMm::CheckedCall(handle, config, *mat, rhs, "sum", mm_out),
                Code::kInvalidCoalescedClaim, 0, Component::kIndices, 1));
  const Tensor index_alias{coords, {2, 3}, DataType::kFloat64, cpu};
  CHECK(Matches(ToDense::CheckedCall(handle, config, *mat, index_alias),
                Code::kAliasing, 0, Component::kIndices));
  CHECK(std::all_of(guarded.begin(), guarded.end(),
                    [](auto v) { return v == 23; }));
  CHECK(std::all_of(product.begin(), product.end(),
                    [](auto v) { return v == 23; }));
  return true;
}

template <typename Index, typename Value>
bool CheckCsr(DataType index_type, DataType value_type) {
  std::array<Index, 4> rows{0, 2, 2, 3}, cols{1, 3, 0};
  std::array<Value, 4> values{2, 0, -3, 9};
  const Tensor crow{rows.data(), {4}, index_type, cpu};
  const Tensor col{cols.data(), {3}, index_type, cpu};
  const Tensor entries{values.data(), {3}, value_type, cpu};
  std::optional<SparseTensor> mat;
  CHECK(SparseTensor::CreateCsr({3, 4}, crow, col, entries, mat));
  std::array<Value, 12> dense;
  const Tensor out{dense.data(), {3, 4}, value_type, cpu};
  CHECK(ToDense::CheckedCall(handle, config, *mat, out));
  CHECK((dense == std::array<Value, 12>{0, 2, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0}));

  // The output begins exactly at the end of the right operand.
  std::array<Value, 7> shared{1, 2, 3, 4};
  const Tensor rhs{shared.data(), {4, 1}, value_type, cpu};
  const Tensor product{shared.data() + 4, {3, 1}, value_type, cpu};
  CHECK(SparseMm::CheckedCall(handle, config, *mat, rhs, "sum", product));
  CHECK((shared == std::array<Value, 7>{1, 2, 3, 4, 4, 0, -3}));
  const Tensor read_alias{values.data(), {4, 1}, value_type, cpu};
  CHECK(
      SparseMm::CheckedCall(handle, config, *mat, read_alias, "sum", product));
  CHECK((shared == std::array<Value, 7>{1, 2, 3, 4, 0, 0, -6}));

  const auto check_alias = [&](void* storage, Component component,
                               std::size_t argument) {
    const Tensor overlapping{storage, {3, 1}, value_type, cpu};
    return Matches(
        SparseMm::CheckedCall(handle, config, *mat, rhs, "sum", overlapping),
        Code::kAliasing, argument, component);
  };
  CHECK(check_alias(rows.data() + 1, Component::kCrowIndices, 0));
  CHECK(check_alias(cols.data() + 1, Component::kColIndices, 0));
  CHECK(check_alias(values.data() + 1, Component::kValues, 0));
  CHECK(check_alias(shared.data() + 3, Component::kValues, 1));

  dense.fill(17);
  const auto rejected = [&](Code code, Component component, std::size_t p) {
    return Matches(ToDense::CheckedCall(handle, config, *mat, out), code, 0,
                   component, p) &&
           Matches(
               SparseMm::CheckedCall(handle, config, *mat, rhs, "sum", product),
               code, 0, component, p);
  };
  rows[0] = 1;
  CHECK(rejected(Code::kInvalidStructure, Component::kCrowIndices, 0));
  rows = {0, 2, 1, 3};
  CHECK(rejected(Code::kInvalidStructure, Component::kCrowIndices, 2));
  rows = {0, std::numeric_limits<Index>::max(), 2, 3};
  CHECK(rejected(Code::kInvalidStructure, Component::kCrowIndices, 1));
  rows = {0, 2, 2, 2};
  CHECK(rejected(Code::kInvalidStructure, Component::kCrowIndices, 3));
  rows = {0, 2, 2, 3};
  for (const Index bad : {0, 1, -1, 4}) {
    cols[1] = bad;
    const auto code =
        bad < 0 || bad >= 4 ? Code::kIndexOutOfBounds : Code::kInvalidStructure;
    CHECK(rejected(code, Component::kColIndices, 1));
  }
  CHECK(
      std::all_of(dense.begin(), dense.end(), [](auto v) { return v == 17; }));
  CHECK((shared == std::array<Value, 7>{1, 2, 3, 4, 0, 0, -6}));
  return true;
}

bool CheckMetadata() {
  const auto type = DataType::kFloat64;
  const Tensor indices{nullptr, {2, 0}, DataType::kInt64, cpu};
  const Tensor values{nullptr, {0}, type, cpu};
  std::optional<SparseTensor> mat;
  CHECK(SparseTensor::CreateCoo({2, 2}, indices, values, mat));
  double data[4]{};
  const Tensor valid{data, {2, 2}, type, cpu};
  struct RequiresConfigAwareProvider : Config {
    bool requires_config_aware_provider() const override { return true; }
  };
  CHECK(ToDense::CheckedCall(handle, RequiresConfigAwareProvider{}, *mat, valid)
            .code == Code::kNoProvider);
  const Tensor wrong_shape{data, {1, 4}, type, cpu};
  CHECK(!ToDense::CanMake(config, *mat, wrong_shape));
  CHECK(Matches(ToDense::CheckedCall(handle, config, *mat, wrong_shape),
                Code::kInvalidMetadata, 1, Component::kShape));
  CHECK(!SparseMm::CanMake(config, *mat, valid, "mean", valid));
  CHECK(!SparseMm::CanMake(config, *mat, *mat, "sum", valid));
  CHECK(
      Matches(SparseMm::CheckedCall(handle, config, *mat, valid, "mean", valid),
              Code::kUnsupportedCapability, 2, Component::kNone));
  CHECK(Matches(SparseMm::CheckedCall(handle, config, *mat, *mat, "sum", valid),
                Code::kUnsupportedCapability, 1, Component::kLayout));
  CHECK(Matches(
      SparseMm::CheckedCall(handle, config, *mat, wrong_shape, "sum", valid),
      Code::kInvalidMetadata, 1, Component::kShape));
  CHECK(Matches(
      SparseMm::CheckedCall(handle, config, *mat, valid, "sum", wrong_shape),
      Code::kInvalidMetadata, 3, Component::kShape));
  const auto check_out = [&](const Tensor& out, Code code) {
    const auto r = ToDense::CheckedCall(handle, config, *mat, out);
    return r.code == code && r.argument == 1;
  };
  CHECK(check_out({nullptr, {2, 2}, type, cpu}, Code::kNullData));
  CHECK(check_out({data, {2, 2}, type, cpu, {-2, 1}},
                  Code::kUnsupportedCapability));
  CHECK(check_out({data, {2, 2}, type, cpu, {1, 2}},
                  Code::kUnsupportedCapability));
  CHECK(check_out({data, {2, 2}, type, Device{Device::Type::kCpu, 1}},
                  Code::kDeviceMismatch));
  CHECK(check_out({data, {2, 2}, DataType::kFloat32, cpu},
                  Code::kInvalidMetadata));
  auto* unaligned = reinterpret_cast<char*>(data) + 1;
  CHECK(check_out({unaligned, {2, 2}, type, cpu}, Code::kInvalidMetadata));
  auto* high =
      reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - 7);
  CHECK(check_out({high, {2, 2}, type, cpu}, Code::kIntegerOverflow));
  const auto rows =
      std::numeric_limits<std::size_t>::max() / sizeof(double) + 1;
  CHECK(SparseTensor::CreateCoo({rows, 1}, indices, values, mat));
  CHECK(
      check_out({data, {rows, 1}, type, cpu, {1, 1}}, Code::kIntegerOverflow));
  return true;
}

}  // namespace

int main() {
  const bool passed =
      CheckCoo() &&
      CheckCsr<std::int32_t, float>(DataType::kInt32, DataType::kFloat32) &&
      CheckCsr<std::int64_t, double>(DataType::kInt64, DataType::kFloat64) &&
      CheckMetadata();
  return passed ? 0 : 1;
}
