#include <cstdint>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>

#include "operator.h"

// Assertions remain active in Release. No real sparse kernel is registered by
// OP1: these providers expose dispatch and repeated-validation behavior only.
#define CHECK(condition)                                              \
  do {                                                                \
    if (!(condition)) {                                               \
      std::cerr << "line " << __LINE__ << ": " << #condition << '\n'; \
      return false;                                                   \
    }                                                                 \
  } while (false)

namespace infini::ops {
namespace test {
inline int constructions = 0, executions = 0, validations = 0;
inline int legacy_executions = 0, tuning_lookups = 0, tuning_records = 0;
inline int throw_mode = 0;
inline std::size_t selected = 0;
inline int64_t observed_index = 0;
inline float observed_value = 0;

template <typename... Args>
ValidationResult Execute(const Config& config, const Args&... args) {
  if (throw_mode == 1) throw std::bad_alloc{};
  if (throw_mode == 2) throw std::runtime_error("test backend failure");
  selected = config.implementation_index();
  ValidationResult result;
  detail::VisitTensorArguments(
      [&](const auto& tensor) {
        if constexpr (std::is_same_v<std::decay_t<decltype(tensor)>,
                                     SparseTensor>) {
          ++validations;
          if (tensor.nnz() == 0) return;
          const Tensor* indices = nullptr;
          const Tensor* values = nullptr;
          if (tensor.layout() == TensorLayout::kCoo) {
            (void)tensor.raw_indices(indices);
          } else {
            (void)tensor.col_indices(indices);
          }
          (void)tensor.raw_values(values);
          observed_index =
              indices->dtype() == DataType::kInt64
                  ? static_cast<const int64_t*>(indices->data())[0]
                  : static_cast<const int32_t*>(indices->data())[0];
          if (observed_index < 0) {
            result = {ValidationCode::kIndexOutOfBounds};
            return;
          }
          observed_value = static_cast<const float*>(values->data())[0];
        }
      },
      args...);
  if (result.ok()) ++executions;
  return result;
}
}  // namespace test

// A controllable replacement avoids linking the Ops DSO or JSON dependency.
// Unexpected sparse tuning can be detected even when it would silently hit a
// prepopulated tuning result and avoid benchmarking.
TuningManager& TuningManager::Instance() {
  static TuningManager manager;
  return manager;
}
void TuningManager::InitializeFromEnvironment() { enabled_ = true; }
std::optional<std::size_t> TuningManager::Lookup(std::string_view, Device::Type,
                                                 const TuningSignature&) const {
  ++test::tuning_lookups;
  return 7;
}
void TuningManager::Record(std::string_view, Device::Type,
                           const TuningSignature&, std::size_t) {
  ++test::tuning_records;
}

struct Probe : Operator<Probe> {
  template <typename... Args>
  explicit Probe(const Args&...) {
    ++test::constructions;
  }
  template <typename... Args>
  void operator()(const Args&...) const {
    ++test::legacy_executions;
  }
};
struct Missing : Operator<Missing> {};
struct LegacyOnly : Operator<LegacyOnly> {
  template <typename... Args>
  explicit LegacyOnly(const Args&...) {}
};

template <>
class Operator<Probe, Device::Type::kCpu, 2> : public Probe {
 public:
  using Probe::Probe;
  // checked_call without an explicit supports() must fail closed.
  template <typename... Args>
  static ValidationResult checked_call(const Handle&, const Config&,
                                       const Args&...) {
    ++test::legacy_executions;
    return {};
  }
};
template <>
class Operator<Probe, Device::Type::kCpu, 7> : public Probe {
 public:
  using Probe::Probe;
  template <typename... Args>
  static bool supports(const Args&... args) {
    bool supported = true;
    detail::VisitTensorArguments(
        [&](const auto& tensor) {
          if constexpr (std::is_same_v<std::decay_t<decltype(tensor)>,
                                       SparseTensor>)
            supported &= tensor.layout() == TensorLayout::kCoo;
        },
        args...);
    return supported;
  }
  template <typename... Args>
  static ValidationResult checked_call(const Handle&, const Config& config,
                                       const Args&... args) {
    return test::Execute(config, args...);
  }
};
template <>
class Operator<Probe, Device::Type::kCpu, 11> : public Probe {
 public:
  using Probe::Probe;
  template <typename... Args>
  static bool supports(const Args&...) {
    return true;
  }
  template <typename... Args>
  static ValidationResult checked_call(const Handle&, const Config& config,
                                       const Args&... args) {
    return test::Execute(config, args...);
  }
};
template <>
class Operator<LegacyOnly, Device::Type::kCpu, 0> : public LegacyOnly {
 public:
  using LegacyOnly::LegacyOnly;
  template <typename... Args>
  static bool supports(const Args&...) {
    return true;
  }
};
}  // namespace infini::ops

namespace {
using namespace infini::ops;
const Device cpu{Device::Type::kCpu};

SparseTensor Coo(void* indices, void* values, bool coalesced = false,
                 Device device = cpu, DataType dtype = DataType::kFloat32,
                 SparseTensor::Shape shape = {2, 3}, std::size_t nnz = 2) {
  Tensor i{indices, {2, nnz}, DataType::kInt64, device};
  Tensor v{values, {nnz}, dtype, device};
  std::optional<SparseTensor> result;
  if (!SparseTensor::CreateCoo(shape, i, v, result, coalesced))
    throw std::runtime_error("invalid test COO metadata");
  return *result;
}

SparseTensor Csr(void* crow, void* col, void* values,
                 DataType index_dtype = DataType::kInt64) {
  Tensor r{crow, {3}, index_dtype, cpu};
  Tensor c{col, {2}, index_dtype, cpu};
  Tensor v{values, {2}, DataType::kFloat32, cpu};
  std::optional<SparseTensor> result;
  if (!SparseTensor::CreateCsr({2, 3}, r, c, v, result))
    throw std::runtime_error("invalid test CSR metadata");
  return *result;
}

template <typename F>
bool Rejects(F&& function) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

bool TraitsAndMetadata() {
  static_assert(detail::TensorArgumentTraits<Tensor>::kDense);
  static_assert(!detail::TensorArgumentTraits<SparseTensor>::kDense);
  static_assert(detail::TensorArgumentTraits<const SparseTensor&>::kSparse);
  static_assert(
      detail::ContainsSparseArgument<int, Tensor,
                                     std::optional<std::vector<SparseTensor>>>);
  static_assert(!detail::ContainsSparseArgument<Tensor, std::vector<Tensor>>);
  static_assert(!detail::IsTensorLike<SparseTensor>::value);
  static_assert(std::is_same_v<decltype(detail::AsCallArg(
                                   std::declval<const SparseTensor&>())),
                               const SparseTensor&>);

  int64_t indices[] = {0, 1, 0, 2}, other_indices[] = {1, 1, 1, 2};
  int64_t crow[] = {0, 1, 2}, col[] = {0, 2};
  int32_t crow32[] = {0, 1, 2}, col32[] = {0, 2};
  float values[] = {1, 2}, other_values[] = {3, 4}, out_data[6] = {};
  const auto a = Coo(indices, values), b = Coo(other_indices, other_values);
  const auto csr = Csr(crow, col, values);
  const auto metadata = detail::SparseMetadataKey::Build(a);
  CHECK(metadata == detail::SparseMetadataKey::Build(b));
  CHECK(metadata.Hash() == detail::SparseMetadataKey::Build(b).Hash());
  CHECK(!(metadata == detail::SparseMetadataKey::Build(csr)));
  CHECK(!(metadata ==
          detail::SparseMetadataKey::Build(Coo(indices, values, true))));
  CHECK(!(metadata ==
          detail::SparseMetadataKey::Build(
              Coo(indices, values, false, cpu, DataType::kFloat32, {3, 3}))));
  CHECK(!(detail::SparseMetadataKey::Build(csr) ==
          detail::SparseMetadataKey::Build(
              Csr(crow32, col32, values, DataType::kInt32))));
  Tensor out{out_data, {2, 3}, DataType::kFloat32, cpu};
  const auto key = detail::CacheKey::Build(a, out, 3);
  auto same = detail::CacheKey::Build(b, out, 3);
  std::equal_to<detail::CacheKey> equal;
  CHECK(equal(key, same));
  CHECK(key.hash == same.hash);
  CHECK(key.tensors.size() == 1 && key.tensors[0].data() == nullptr);
  CHECK(!equal(key, detail::CacheKey::Build(csr, out, 3)));
  CHECK(!equal(key, detail::CacheKey::Build(a, out, 4)));
  CHECK(!equal(key, detail::CacheKey::Build(out, a, 3)));
  CHECK(!equal(detail::CacheKey::Build(std::vector<SparseTensor>{a, b}),
               detail::CacheKey::Build(a, b)));
  CHECK(!equal(detail::CacheKey::Build(std::optional<SparseTensor>{a}),
               detail::CacheKey::Build(std::optional<SparseTensor>{})));

  std::optional<std::vector<SparseTensor>> nested{
      std::in_place, std::initializer_list<SparseTensor>{a, b}};
  std::size_t visited = 0;
  detail::VisitTensorArguments([&](const auto&) { ++visited; },
                               std::optional<Tensor>{},
                               std::vector<Tensor>{out}, nested, 13);
  CHECK(visited == 3);
  CHECK(detail::FirstDeviceType(std::optional<SparseTensor>{}, nested) ==
        Device::Type::kCpu);
  CHECK(detail::FirstDeviceType(std::vector<SparseTensor>{}) ==
        Device::Type::kCount);
  CHECK(detail::ValidateSparseArguments(nested, out).ok());

  // Metadata operations must not read the aligned but inaccessible buffers.
  auto inaccessible =
      Coo(reinterpret_cast<void*>(0x1000), reinterpret_cast<void*>(0x2000));
  CHECK(detail::ValidateSparseArguments(inaccessible).ok());
  CHECK(detail::SparseMetadataKey::Build(inaccessible) == metadata);
  const auto constructions = test::constructions;
  CHECK(Probe::CanMake(Config{}, inaccessible));
  CHECK(test::constructions == constructions);
  CHECK(Probe::Make(Config{}, inaccessible) != nullptr);
  CHECK(Probe::Make(Config{}, out, inaccessible) != nullptr);
  CHECK(Probe::CanMake(Config{}, nested, out));
  CHECK(Probe::Make(out, inaccessible) != nullptr);
  CHECK(Probe::Make(std::vector<Tensor>{}, inaccessible) != nullptr);
  CHECK(Probe::Make(std::optional<Tensor>{out}, inaccessible) != nullptr);
  CHECK(Probe::CanMake(Config{}, std::optional<Tensor>{}, inaccessible));
  auto prepared = Probe::Make(Config{}, inaccessible);
  CHECK(Rejects([&] { (*prepared)(Handle{}, inaccessible); }));
  CHECK(Rejects([&] { std::as_const (*prepared)(inaccessible); }));
  return true;
}

bool CheckedDispatchAndMutation() {
  int64_t indices[] = {0, 1, 0, 2}, other_indices[] = {1, 1, 1, 2};
  int64_t crow[] = {0, 1, 2}, col[] = {0, 2};
  float values[] = {1, 2}, other_values[] = {3, 4}, out_data[6] = {};
  const auto a = Coo(indices, values), b = Coo(other_indices, other_values);
  Tensor out{out_data, {2, 3}, DataType::kFloat32, cpu};
  TuningManager::Instance().InitializeFromEnvironment();
  CHECK(TuningManager::Instance().IsEnabled());
  int validations = test::validations;
  int executions = test::executions;
  CHECK(Probe::CheckedCall(Handle{}, Config{}, a, out).ok());
  CHECK(test::selected == 7 && test::observed_index == 0);
  CHECK(Probe::CheckedCall(Handle{}, Config::Automatic(), out, b).ok());
  CHECK(test::observed_index == 1 && test::observed_value == 3);
  indices[0] = 1;  // Same descriptor and address, changed content.
  CHECK(Probe::CheckedCall(Handle{}, Config{}, a, out).ok());
  CHECK(test::observed_index == 1);
  values[0] = 9;  // Values updates must also reach the current call.
  CHECK(Probe::CheckedCall(Handle{}, Config{}, a, out).ok());
  CHECK(test::observed_value == 9);
  CHECK(test::validations == validations + 4 &&
        test::executions == executions + 4);
  indices[0] = -1;
  CHECK(Probe::CheckedCall(Handle{}, Config{}, a, out).code ==
        ValidationCode::kIndexOutOfBounds);
  CHECK(test::executions == executions + 4);
  // Reuse the same storage for a new descriptor; a prior failure is not cached.
  indices[0] = 0;
  const auto recycled = Coo(indices, values);
  CHECK(Probe::CheckedCall(Handle{}, Config{}, recycled, out).ok());
  CHECK(test::observed_index == 0);
  CHECK(Probe::CheckedCall(
            Handle{}, Config{},
            std::optional<std::vector<SparseTensor>>{
                std::in_place, std::initializer_list<SparseTensor>{a, b}},
            out)
            .ok());
  CHECK(Probe::CheckedCall(Handle{}, Config{}, Csr(crow, col, values)).ok());
  CHECK(test::selected == 11);
  CHECK(test::legacy_executions == 0);

  Config explicit_good;
  explicit_good.set_implementation_index(7);
  CHECK(Probe::CheckedCall(Handle{}, explicit_good, a).ok());
  CHECK(Probe::CheckedCall(Handle{}, explicit_good, Csr(crow, col, values))
            .code == ValidationCode::kUnsupportedImplementation);
  Config explicit_bad;
  explicit_bad.set_implementation_index(31);
  CHECK(Probe::CheckedCall(Handle{}, explicit_bad, a).code ==
        ValidationCode::kUnsupportedImplementation);
  explicit_bad.set_implementation_index(2);
  CHECK(Probe::CheckedCall(Handle{}, explicit_bad, a).code ==
        ValidationCode::kUnsupportedImplementation);
  CHECK(Missing::CheckedCall(Handle{}, Config{}, a).code ==
        ValidationCode::kNoProvider);
  CHECK(LegacyOnly::CheckedCall(Handle{}, Config{}, a).code ==
        ValidationCode::kNoProvider);
  CHECK(Rejects([&] { Probe::Call(Handle{}, Config{}, a, out); }));
  CHECK(Rejects([&] { Probe::Call(Handle{}, Config{}, out, a); }));
  CHECK(Rejects([&] { Probe::Call(out, a); }));
  CHECK(Rejects(
      [&] { TuningSignature::Build(out, std::optional<SparseTensor>{}); }));
  CHECK(Rejects([&] {
    detail::BenchmarkImplementation<Probe>(Handle{}, Device::Type::kCpu, 7, out,
                                           a);
  }));
  CHECK(detail::ResolveImplementationIndex<Probe>(Config{}, Device::Type::kCpu,
                                                  out, a) ==
        Config::kAutomaticImplementation);
  CHECK(detail::ResolveImplementationIndexOnline<Probe>(
            Handle{}, Config{}, Device::Type::kCpu,
            std::vector<SparseTensor>{a},
            out) == Config::kAutomaticImplementation);
  CHECK(test::tuning_lookups == 0 && test::tuning_records == 0);
  test::throw_mode = 1;
  CHECK(Probe::CheckedCall(Handle{}, Config{}, a).code ==
        ValidationCode::kAllocationFailure);
  test::throw_mode = 2;
  CHECK(Probe::CheckedCall(Handle{}, Config{}, a).code ==
        ValidationCode::kExecutionFailure);
  test::throw_mode = 0;
  // Dense Call keeps its old path and explicit implementation behavior.
  Probe::Call(Handle{}, explicit_good, out);
  CHECK(test::legacy_executions == 1);
  return true;
}

bool MetadataFailures() {
  int64_t indices[] = {0, 1, 0, 2};
  float values[] = {1, 2}, output[6] = {};
  double values64[] = {1, 2};
  const auto a = Coo(indices, values);
  const int executions = test::executions;
  Tensor other_device{
      output, {2, 3}, DataType::kFloat32, Device{Device::Type::kCpu, 1}};
  CHECK(Probe::CheckedCall(Handle{}, Config{}, a, other_device).code ==
        ValidationCode::kDeviceMismatch);
  CHECK(Probe::CheckedCall(Handle{}, Config{},
                           std::vector<Tensor>{other_device},
                           std::optional<SparseTensor>{a})
            .code == ValidationCode::kDeviceMismatch);
  CHECK(Probe::CheckedCall(
            Handle{}, Config{},
            Coo(indices, values, false, Device{Device::Type::kNvidia}))
            .code == ValidationCode::kUnsupportedCapability);
  CHECK(Probe::CheckedCall(Handle{}, Config{},
                           Coo(indices, values, false, cpu, DataType::kInt32))
            .code == ValidationCode::kUnsupportedCapability);
  Tensor out{output, {2, 3}, DataType::kFloat32, cpu};
  CHECK(Probe::CheckedCall(
            Handle{}, Config{},
            Coo(indices, values64, false, cpu, DataType::kFloat64), out)
            .code == ValidationCode::kInvalidMetadata);
  Tensor null_out{nullptr, {2, 3}, DataType::kFloat32, cpu};
  CHECK(Probe::CheckedCall(Handle{}, Config{}, a, null_out).code ==
        ValidationCode::kNullData);
  Tensor strided{output, {2, 3}, DataType::kFloat32, cpu, {6, 2}};
  CHECK(Probe::CheckedCall(Handle{}, Config{}, a, strided).code ==
        ValidationCode::kUnsupportedCapability);
  Tensor negative{output, {2, 3}, DataType::kFloat32, cpu, {-3, 1}};
  CHECK(Probe::CheckedCall(Handle{}, Config{}, a, negative).code ==
        ValidationCode::kUnsupportedCapability);
  Tensor unaligned{
      reinterpret_cast<char*>(output) + 1, {2, 3}, DataType::kFloat32, cpu};
  CHECK(Probe::CheckedCall(Handle{}, Config{}, a, unaligned).code ==
        ValidationCode::kInvalidMetadata);
  Tensor endpoint{
      reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - 3),
      {2},
      DataType::kFloat32,
      cpu};
  CHECK(Probe::CheckedCall(Handle{}, Config{}, a, endpoint).code ==
        ValidationCode::kIntegerOverflow);
  CHECK(Probe::CheckedCall(Handle{}, Config{}, std::optional<SparseTensor>{})
            .code == ValidationCode::kInvalidMetadata);
  CHECK(test::executions == executions);
  Tensor empty{nullptr, {0, 3}, DataType::kFloat32, cpu};
  CHECK(Probe::CheckedCall(
            Handle{}, Config{},
            Coo(nullptr, nullptr, false, cpu, DataType::kFloat32, {0, 3}, 0),
            empty)
            .ok());

  infini::rt::ValidationResult rt_error{
      infini::rt::ValidationCode::kIntegerOverflow,
      infini::rt::ValidationComponent::kIndices, 5};
  auto mapped = FromRtValidation(rt_error);
  CHECK(mapped.code == ValidationCode::kIntegerOverflow);
  CHECK(mapped.component == rt_error.component &&
        mapped.position == rt_error.position);
  return true;
}
}  // namespace

int main() {
  if (!TraitsAndMetadata() || !CheckedDispatchAndMutation() ||
      !MetadataFailures())
    return 1;
  std::cout << "OP1 sparse contract passed\n";
}
