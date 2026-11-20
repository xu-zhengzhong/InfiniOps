#ifndef INFINI_OPS_TENSOR_H_
#define INFINI_OPS_TENSOR_H_

#include <infini/rt.h>

#include <optional>
#include <type_traits>
#include <vector>

namespace infini::ops {

using Tensor = infini::rt::TensorView;

using SparseTensor = infini::rt::SparseTensorView;

using TensorLayout = infini::rt::TensorLayout;

namespace detail {

template <typename T>
struct TensorArgumentTraitsImpl {
  static constexpr bool kDense = std::is_same_v<T, Tensor>;

  static constexpr bool kSparse = std::is_same_v<T, SparseTensor>;

  template <typename Visitor>
  static void Visit(Visitor& visitor, const T& value) {
    if constexpr (kDense || kSparse) visitor(value);
  }
};

template <typename T>
struct TensorArgumentTraits : TensorArgumentTraitsImpl<std::decay_t<T>> {};

template <typename T>
struct TensorArgumentTraitsImpl<std::optional<T>> : TensorArgumentTraits<T> {
  template <typename Visitor>
  static void Visit(Visitor& visitor, const std::optional<T>& value) {
    if (value) TensorArgumentTraits<T>::Visit(visitor, *value);
  }
};

template <typename T, typename Allocator>
struct TensorArgumentTraitsImpl<std::vector<T, Allocator>>
    : TensorArgumentTraits<T> {
  template <typename Visitor>
  static void Visit(Visitor& visitor, const std::vector<T, Allocator>& values) {
    for (const auto& value : values) {
      TensorArgumentTraits<T>::Visit(visitor, value);
    }
  }
};

template <typename... Args>
inline constexpr bool ContainsSparseArgument =
    (TensorArgumentTraits<Args>::kSparse || ...);

template <typename Visitor, typename... Args>
void VisitTensorArguments(Visitor&& visitor, const Args&... args) {
  (TensorArgumentTraits<Args>::Visit(visitor, args), ...);
}

template <typename... Args>
std::optional<infini::rt::Device> FirstTensorDevice(const Args&... args) {
  std::optional<infini::rt::Device> result;
  VisitTensorArguments(
      [&](const auto& tensor) {
        if (!result) result = tensor.device();
      },
      args...);
  return result;
}

}  // namespace detail

}  // namespace infini::ops

#endif
