#ifndef INFINI_OPS_SPARSE_H_
#define INFINI_OPS_SPARSE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <vector>

#include "data_type.h"
#include "device.h"
#include "tensor.h"

namespace infini::ops {

enum class ValidationCode {
  kSuccess,
  kInvalidMetadata,
  kUnsupportedLayout,
  kUnsupportedCapability,
  kInvalidIndexType,
  kDeviceMismatch,
  kNullData,
  kIntegerOverflow,
  kIndexOutOfBounds,
  kInvalidStructure,
  kInvalidCoalescedClaim,
  kInsufficientCapacity,
  kInvalidWorkspace,
  kAliasing,
  kNoProvider,
  kUnsupportedImplementation,
  kInvalidState,
  kStaleStructure,
  kAutogradUnsupported,
  kAllocationFailure,
  kExecutionFailure,
};

using ValidationComponent = infini::rt::ValidationComponent;

struct [[nodiscard]] ValidationResult {
  ValidationCode code{ValidationCode::kSuccess};

  ValidationComponent component{ValidationComponent::kNone};

  std::optional<std::size_t> position{};

  // Top-level argument index; component and position refine the diagnostic.
  std::optional<std::size_t> argument{};

  constexpr bool ok() const noexcept {
    return code == ValidationCode::kSuccess;
  }

  constexpr explicit operator bool() const noexcept { return ok(); }
};

inline ValidationResult FromRtValidation(
    const infini::rt::ValidationResult& result,
    std::optional<std::size_t> argument = std::nullopt) noexcept {
  using RtCode = infini::rt::ValidationCode;
  ValidationCode code = ValidationCode::kInvalidState;
  switch (result.code) {
    case RtCode::kSuccess:
      return {};
    case RtCode::kInvalidMetadata:
      code = ValidationCode::kInvalidMetadata;
      break;
    case RtCode::kUnsupportedLayout:
      code = ValidationCode::kUnsupportedLayout;
      break;
    case RtCode::kUnsupportedCapability:
      code = ValidationCode::kUnsupportedCapability;
      break;
    case RtCode::kInvalidIndexType:
      code = ValidationCode::kInvalidIndexType;
      break;
    case RtCode::kDeviceMismatch:
      code = ValidationCode::kDeviceMismatch;
      break;
    case RtCode::kNullData:
      code = ValidationCode::kNullData;
      break;
    case RtCode::kIntegerOverflow:
      code = ValidationCode::kIntegerOverflow;
      break;
    case RtCode::kInvalidState:
      code = ValidationCode::kInvalidState;
      break;
    case RtCode::kAllocationFailure:
      code = ValidationCode::kAllocationFailure;
      break;
  }
  return {code, result.component, result.position, argument};
}

namespace detail {

// Sparse views are immutable, but repeat the metadata-only RT checks at the
// checked boundary. This does not validate index contents or coalesced claims.
inline ValidationResult ValidateSparseMetadata(const SparseTensor& tensor) {
  const Tensor* indices = nullptr;
  const Tensor* col_indices = nullptr;
  const Tensor* values = nullptr;
  auto status = tensor.raw_values(values);
  if (!status) return FromRtValidation(status);
  std::optional<SparseTensor> checked;
  if (tensor.layout() == TensorLayout::kCoo) {
    status = tensor.raw_indices(indices);
    if (!status) return FromRtValidation(status);
    bool coalesced = false;
    status = tensor.is_coalesced(coalesced);
    if (!status) return FromRtValidation(status);
    return FromRtValidation(SparseTensor::CreateCoo(
        tensor.shape(), *indices, *values, checked, coalesced));
  }
  if (tensor.layout() == TensorLayout::kCsr) {
    status = tensor.crow_indices(indices);
    if (!status) return FromRtValidation(status);
    status = tensor.col_indices(col_indices);
    if (!status) return FromRtValidation(status);
    return FromRtValidation(SparseTensor::CreateCsr(
        tensor.shape(), *indices, *col_indices, *values, checked));
  }
  return {ValidationCode::kUnsupportedLayout, ValidationComponent::kLayout};
}

// Validate dense operands in sparse calls without overflow-prone numel() or
// element_size() queries. All supported value operands are float32/float64.
inline ValidationResult ValidateSparseMetadata(const Tensor& tensor) {
  using Code = ValidationCode;
  constexpr auto component = ValidationComponent::kValues;
  const auto shape = tensor.shape();
  const auto strides = tensor.strides();
  if (shape.size() != strides.size()) {
    return {Code::kInvalidMetadata, component};
  }
  if (tensor.dtype() != DataType::kFloat32 &&
      tensor.dtype() != DataType::kFloat64) {
    return {Code::kUnsupportedCapability, component};
  }
  bool empty = false;
  for (std::size_t axis = 0; axis < shape.size(); ++axis) {
    if (strides[axis] < 0) {
      return {Code::kUnsupportedCapability, component, axis};
    }
    empty = empty || shape[axis] == 0;
  }
  std::size_t elements = empty ? 0 : 1;
  if (!empty) {
    for (std::size_t axis = shape.size(); axis != 0;) {
      --axis;
      if (shape[axis] > 1) {
        if (elements > static_cast<std::size_t>(
                           std::numeric_limits<Tensor::Stride>::max())) {
          return {Code::kIntegerOverflow, component, axis};
        }
        if (static_cast<std::size_t>(strides[axis]) != elements) {
          return {Code::kUnsupportedCapability, component, axis};
        }
      }
      if (elements > std::numeric_limits<std::size_t>::max() / shape[axis]) {
        return {Code::kIntegerOverflow, component};
      }
      elements *= shape[axis];
    }
  }
  const auto element_size =
      tensor.dtype() == DataType::kFloat32 ? sizeof(float) : sizeof(double);
  const auto alignment =
      tensor.dtype() == DataType::kFloat32 ? alignof(float) : alignof(double);
  if (elements > std::numeric_limits<std::size_t>::max() / element_size) {
    return {Code::kIntegerOverflow, component};
  }
  const auto bytes = elements * element_size;
  if (bytes == 0) return {};
  if (tensor.data() == nullptr) return {Code::kNullData, component};
  const auto address = reinterpret_cast<std::uintptr_t>(tensor.data());
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - address) {
    return {Code::kIntegerOverflow, component};
  }
  if (address % alignment != 0) return {Code::kInvalidMetadata, component};
  return {};
}

// Common P1/P2 metadata policy only. Providers must separately check operator
// shapes, resources, aliasing and index contents on every checked invocation.
template <typename... Args>
ValidationResult ValidateSparseArguments(const Args&... args) {
  ValidationResult result;
  std::optional<Device> device;
  std::optional<DataType> dtype;
  std::size_t argument = 0;
  const auto check_device = [&](const auto& arg) {
    VisitTensorArguments(
        [&](const auto& tensor) {
          if (!result) return;
          const auto current = tensor.device();
          if (current.type() < Device::Type::kCpu ||
              current.type() >= Device::Type::kCount || current.index() < 0) {
            result = {ValidationCode::kInvalidMetadata};
          } else if (device && current != *device) {
            result = {ValidationCode::kDeviceMismatch};
          } else {
            device = current;
          }
          if (!result) result.argument = argument;
        },
        arg);
    ++argument;
  };
  (check_device(args), ...);
  if (!result) return result;
  if (!device) return {ValidationCode::kInvalidMetadata};
  if (device->type() != Device::Type::kCpu) {
    return {ValidationCode::kUnsupportedCapability};
  }
  argument = 0;
  const auto check_metadata = [&](const auto& arg) {
    VisitTensorArguments(
        [&](const auto& tensor) {
          if (!result) return;
          result = ValidateSparseMetadata(tensor);
          if (result && tensor.dtype() != DataType::kFloat32 &&
              tensor.dtype() != DataType::kFloat64) {
            result = {ValidationCode::kUnsupportedCapability,
                      ValidationComponent::kValues};
          }
          if (result && dtype && tensor.dtype() != *dtype) {
            result = {ValidationCode::kInvalidMetadata,
                      ValidationComponent::kValues};
          }
          if (!result) {
            result.argument = argument;
          } else {
            dtype = tensor.dtype();
          }
        },
        arg);
    ++argument;
  };
  (check_metadata(args), ...);
  return result;
}

// No data pointers, validation proofs or content-dependent analysis may be
// retained in these keys. Equal keys do not imply equal index contents.
struct SparseMetadataKey {
  struct Component {
    std::vector<Tensor::Size> shape;

    std::vector<Tensor::Stride> strides;

    DataType dtype;

    Device device;

    bool operator==(const Component& other) const {
      return std::tie(shape, strides, dtype, device) ==
             std::tie(other.shape, other.strides, other.dtype, other.device);
    }
  };

  TensorLayout layout;

  SparseTensor::Shape shape;

  Tensor::Size sparse_dim;

  Tensor::Size dense_dim;

  Tensor::Size batch_dim;

  Tensor::Size nnz;

  DataType dtype;

  DataType index_dtype;

  Device device;

  bool coalesced{false};

  std::vector<Component> components{};

  static SparseMetadataKey Build(const SparseTensor& tensor) {
    SparseMetadataKey key{
        tensor.layout(),    tensor.shape(),       tensor.sparse_dim(),
        tensor.dense_dim(), tensor.batch_dim(),   tensor.nnz(),
        tensor.dtype(),     tensor.index_dtype(), tensor.device()};
    const auto append = [&](const Tensor* component) {
      const auto shape = component->shape();
      const auto strides = component->strides();
      key.components.push_back({{shape.begin(), shape.end()},
                                {strides.begin(), strides.end()},
                                component->dtype(),
                                component->device()});
    };
    // Valid immutable RT views guarantee access to their raw components.
    const Tensor* component = nullptr;
    if (tensor.layout() == TensorLayout::kCoo) {
      (void)tensor.is_coalesced(key.coalesced);
      (void)tensor.raw_indices(component);
      append(component);
    } else {
      (void)tensor.crow_indices(component);
      append(component);
      (void)tensor.col_indices(component);
      append(component);
    }
    (void)tensor.raw_values(component);
    append(component);
    return key;
  }

  bool operator==(const SparseMetadataKey& other) const {
    return std::tie(layout, shape, sparse_dim, dense_dim, batch_dim, nnz, dtype,
                    index_dtype, device, coalesced, components) ==
           std::tie(other.layout, other.shape, other.sparse_dim,
                    other.dense_dim, other.batch_dim, other.nnz, other.dtype,
                    other.index_dtype, other.device, other.coalesced,
                    other.components);
  }

  std::size_t Hash() const {
    std::size_t hash = 0;
    HashCombine(hash, layout);
    for (const auto size : shape) HashCombine(hash, size);
    HashCombine(hash, sparse_dim);
    HashCombine(hash, dense_dim);
    HashCombine(hash, batch_dim);
    HashCombine(hash, nnz);
    HashCombine(hash, dtype);
    HashCombine(hash, index_dtype);
    HashCombine(hash, device);
    HashCombine(hash, coalesced);
    HashCombine(hash, components.size());
    for (const auto& component : components) {
      HashCombine(hash, component.shape);
      HashCombine(hash, component.strides);
      HashCombine(hash, component.dtype);
      HashCombine(hash, component.device);
    }
    return hash;
  }
};

}  // namespace detail

}  // namespace infini::ops

#endif
