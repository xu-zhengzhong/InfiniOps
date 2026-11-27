#ifndef INFINI_OPS_PYTHON_SPARSE_IMPORT_H_
#define INFINI_OPS_PYTHON_SPARSE_IMPORT_H_

#include "native/cpu/sparse_validation.h"
#include "pybind11_utils.h"

namespace infini::ops {

// pybind11 checks these object types (also inside STL casters) during
// overload resolution, before any tensor or component is imported.
class PyDenseTensor : public py::object {
 public:
  using py::object::object;

  static bool check_(py::handle obj) {
    const auto layout = detail::GetPythonLayout(obj);
    return layout == detail::PythonLayout::kStrided ||
           (layout == detail::PythonLayout::kUnspecified &&
            py::hasattr(obj, detail::GetInternedNames().data_ptr));
  }
};

class PySparseTensor : public py::object {
 public:
  using py::object::object;

  static bool check_(py::handle obj) {
    const auto layout = detail::GetPythonLayout(obj);
    return layout == detail::PythonLayout::kCoo ||
           layout == detail::PythonLayout::kCsr;
  }
};

namespace detail {

struct PythonValidationError {
  PyObject* type;
  const char* message;
};

inline PythonValidationError PythonErrorFor(ValidationCode code) {
  using Code = ValidationCode;
  switch (code) {
    case Code::kInvalidMetadata:
      return {PyExc_ValueError, "InvalidMetadata"};
    case Code::kUnsupportedLayout:
      return {PyExc_NotImplementedError, "UnsupportedLayout"};
    case Code::kUnsupportedCapability:
      return {PyExc_NotImplementedError, "UnsupportedCapability"};
    case Code::kInvalidIndexType:
      return {PyExc_TypeError, "InvalidIndexType"};
    case Code::kDeviceMismatch:
      return {PyExc_ValueError, "DeviceMismatch"};
    case Code::kNullData:
      return {PyExc_ValueError, "NullData"};
    case Code::kIntegerOverflow:
      return {PyExc_OverflowError, "IntegerOverflow"};
    case Code::kIndexOutOfBounds:
      return {PyExc_IndexError, "IndexOutOfBounds"};
    case Code::kInvalidStructure:
      return {PyExc_ValueError, "InvalidStructure"};
    case Code::kInvalidCoalescedClaim:
      return {PyExc_ValueError, "InvalidCoalescedClaim"};
    case Code::kInsufficientCapacity:
      return {PyExc_ValueError, "InsufficientCapacity"};
    case Code::kInvalidWorkspace:
      return {PyExc_ValueError, "InvalidWorkspace"};
    case Code::kAliasing:
      return {PyExc_ValueError, "Aliasing"};
    case Code::kNoProvider:
      return {PyExc_NotImplementedError, "NoProvider"};
    case Code::kUnsupportedImplementation:
      return {PyExc_ValueError, "UnsupportedImplementation"};
    case Code::kInvalidState:
      return {PyExc_RuntimeError, "InvalidState"};
    case Code::kStaleStructure:
      return {PyExc_RuntimeError, "StaleStructure"};
    case Code::kAutogradUnsupported:
      return {PyExc_NotImplementedError,
              "AutogradUnsupported (call detach() explicitly)"};
    case Code::kAllocationFailure:
      return {PyExc_MemoryError, "AllocationFailure"};
    case Code::kExecutionFailure:
      return {PyExc_RuntimeError, "ExecutionFailure"};
    case Code::kSuccess:
      break;
  }
  return {PyExc_RuntimeError, "Unknown validation code"};
}

inline const char* PythonComponentName(ValidationComponent component) {
  using Component = ValidationComponent;
  switch (component) {
    case Component::kLayout:
      return "layout";
    case Component::kShape:
      return "shape";
    case Component::kComponents:
      return "components";
    case Component::kIndices:
      return "indices";
    case Component::kCrowIndices:
      return "crow_indices";
    case Component::kColIndices:
      return "col_indices";
    case Component::kValues:
      return "values";
    case Component::kCoalesced:
      return "coalesced";
    case Component::kNone:
      break;
  }
  return "";
}

[[noreturn]] inline void RaiseSparseError(ValidationResult result,
                                          const char* hint = nullptr) {
  const auto error = PythonErrorFor(result.code);
  std::string message = error.message;
  if (result.argument)
    message += " at argument " + std::to_string(*result.argument);
  if (result.component != ValidationComponent::kNone) {
    message += " ";
    message += PythonComponentName(result.component);
  }
  if (result.position) message += "[" + std::to_string(*result.position) + "]";
  if (hint) message += std::string{" ("} + hint + ")";
  PyErr_SetString(error.type, message.c_str());
  throw py::error_already_set();
}

// ATen can invoke a Python dispatch hook. Its C++ exception has already been
// translated by the provider, but PyTorch can leave the thread's Python error
// set. Consume it before returning a submission or raising the checked error.
inline ValidationResult ConsumeProviderPythonError(ValidationResult result) {
  if (!PyErr_Occurred()) return result;
  py::error_already_set error;
  if (result.code != ValidationCode::kExecutionFailure) return result;
  if (error.matches(PyExc_NotImplementedError)) {
    result.code = ValidationCode::kUnsupportedCapability;
  } else if (error.matches(PyExc_MemoryError) ||
             error.matches(py::getattr(py::module_::import("torch"),
                                       "OutOfMemoryError", py::none()))) {
    result.code = ValidationCode::kAllocationFailure;
  }
  return result;
}

}  // namespace detail

inline void CheckSparseResult(ValidationResult result) {
  if (!result.ok())
    detail::RaiseSparseError(detail::ConsumeProviderPythonError(result));
}

// The binding owns its arguments. Accessors can return temporary Python
// objects, so additionally retain every component through the synchronous call.
class PythonSparseCall {
 public:
  explicit PythonSparseCall(std::optional<std::uintptr_t> stream = {})
      : requested_stream_(stream) {
    handle_.set_stream(reinterpret_cast<void*>(stream.value_or(0)));
  }

  const Handle& handle() const { return handle_; }

  void CheckStream() const {
#ifdef WITH_NVIDIA
    if (stream_device_ && CurrentStream(*stream_device_) != handle_.stream())
      detail::RaiseSparseError({Code::kUnsupportedCapability},
                               "execute on the CUDA stream used for prepare");
#endif
  }

  template <typename T, typename PyArg>
  T Import(const PyArg& obj, std::size_t argument, bool out_only = false) {
    return ConvertNestedTensor<T>(obj, [&](const auto& tensor) {
      return ImportTensor(tensor, argument, out_only);
    });
  }

  static DataType AttributeType(py::handle dtype, std::size_t argument) {
    return ImportDtype(dtype, argument, Component::kNone,
                       Code::kUnsupportedCapability,
                       ConcatType<AllTypes, List<DataType::kBool>>{});
  }

 private:
  using Code = ValidationCode;
  using Component = ValidationComponent;

  static Device ForwardDevice(py::handle obj, std::size_t argument,
                              Component component) {
    if (obj.attr("requires_grad").cast<bool>()) {
      detail::RaiseSparseError(
          {Code::kAutogradUnsupported, component, {}, argument});
    }
    const auto device = obj.attr("device");
    const auto name = device.attr("type").cast<std::string>();
#ifdef WITH_NVIDIA
    if (name == "cuda")
      return Device{Device::Type::kNvidia, device.attr("index").cast<int>()};
#endif
    if (name != "cpu") {
      detail::RaiseSparseError(
          {Code::kUnsupportedCapability, component, {}, argument},
          "unsupported sparse import device");
    }
    const auto index = device.attr("index");
    return Device{Device::Type::kCpu, index.is_none() ? 0 : index.cast<int>()};
  }

  template <DataType... types>
  static DataType ImportDtype(py::handle dtype, std::size_t argument,
                              Component component, Code failure,
                              List<types...>) {
    const auto spelling = py::str(dtype).cast<std::string>();
    const auto name =
        std::string_view(spelling).substr(spelling.find_last_of('.') + 1);
    for (const auto type : {types...}) {
      if (name == kDataTypeToDesc.at(type)) return type;
    }
    detail::RaiseSparseError({failure, component, {}, argument},
                             spelling.c_str());
  }

  static Tensor ImportStrided(py::handle obj, std::size_t argument,
                              Component component) {
    if (!PyDenseTensor::check_(obj)) {
      detail::RaiseSparseError(
          {Code::kUnsupportedLayout, component, {}, argument});
    }
    const auto device = ForwardDevice(obj, argument, component);
    if (py::hasattr(obj, "is_neg") && obj.attr("is_neg")().cast<bool>()) {
      detail::RaiseSparseError(
          {Code::kUnsupportedCapability, component, {}, argument},
          "a borrowed view cannot represent an unresolved negative bit");
    }
    // Check the P1 dtype set before entering the dense RT name lookup, which
    // expects a known dtype. Sparse import never casts indices or values.
    const auto dtype = obj.attr("dtype");
    const auto type =
        component == Component::kNone || component == Component::kValues
            ? ImportDtype(dtype, argument, component,
                          Code::kUnsupportedCapability, FloatTypes{})
            : ImportDtype(dtype, argument, component, Code::kInvalidIndexType,
                          List<DataType::kInt32, DataType::kInt64>{});
    return detail::StridedTensorFromPybind11Handle(obj, type, device);
  }

  Tensor ImportComponent(py::handle tensor, const char* accessor,
                         std::size_t argument, Component component) {
    auto owner = tensor.attr(accessor)();
    if (!py::isinstance(owner, tensor_type_)) {
      detail::RaiseSparseError(
          {Code::kInvalidMetadata, component, {}, argument},
          "expected a torch.Tensor component");
    }
    owners_.push_back(std::move(owner));
    return ImportStrided(owners_.back(), argument, component);
  }

  static Tensor ImportTensor(const PyDenseTensor& tensor, std::size_t argument,
                             bool) {
    return ImportStrided(tensor, argument, Component::kNone);
  }

  SparseTensor ImportTensor(const PySparseTensor& tensor, std::size_t argument,
                            bool out_only) {
    const auto layout = detail::GetPythonLayout(tensor);
    const auto device = ForwardDevice(tensor, argument, Component::kNone);
    const auto shape =
        detail::VectorFromPybind11Handle<Tensor::Shape>(tensor.attr("shape"));
    if (shape.size() != 2 || tensor.attr("sparse_dim")().cast<int>() != 2 ||
        tensor.attr("dense_dim")().cast<int>() != 0) {
      detail::RaiseSparseError(
          {Code::kUnsupportedCapability, Component::kShape, {}, argument},
          "expected a two-dimensional COO or CSR tensor with scalar values");
    }
    const SparseTensor::Shape size{shape[0], shape[1]};
    std::optional<SparseTensor> view;
    if (layout == detail::PythonLayout::kCoo) {
      const bool coalesced = tensor.attr("is_coalesced")().cast<bool>();
      const auto indices =
          ImportComponent(tensor, "_indices", argument, Component::kIndices);
      const auto values =
          ImportComponent(tensor, "_values", argument, Component::kValues);
      CheckSparseResult(FromRtValidation(
          SparseTensor::CreateCoo(size, indices, values, view, coalesced),
          argument));
    } else if (layout == detail::PythonLayout::kCsr) {
      const auto crow = ImportComponent(tensor, "crow_indices", argument,
                                        Component::kCrowIndices);
      const auto col = ImportComponent(tensor, "col_indices", argument,
                                       Component::kColIndices);
      const auto values =
          ImportComponent(tensor, "values", argument, Component::kValues);
      CheckSparseResult(FromRtValidation(
          SparseTensor::CreateCsr(size, crow, col, values, view), argument));
    } else {
      detail::RaiseSparseError(
          {Code::kUnsupportedLayout, Component::kLayout, {}, argument});
    }
    if (device != view->device()) {
      detail::RaiseSparseError(
          {Code::kDeviceMismatch, Component::kComponents, {}, argument});
    }
#ifdef WITH_NVIDIA
    if (device.type() == Device::Type::kNvidia) {
      if (stream_device_ && *stream_device_ != device.index())
        detail::RaiseSparseError(
            {Code::kDeviceMismatch, Component::kComponents, {}, argument});
      stream_device_ = device.index();
      auto current = CurrentStream(device.index());
      if (requested_stream_ &&
          reinterpret_cast<void*>(*requested_stream_) != current)
        detail::RaiseSparseError(
            {Code::kUnsupportedCapability, Component::kNone, {}, argument},
            "select the current PyTorch CUDA stream");
      handle_.set_stream(current);
      // The checked NVIDIA provider validates on this stream after alias and
      // capacity checks. A host validator cannot inspect these components.
      return *view;
    }
#endif
    if (!out_only) {
      auto status = native::cpu::ValidateSparseIndices(*view);
      status.argument = argument;
      CheckSparseResult(status);
    }
    return *view;
  }

#ifdef WITH_NVIDIA
  static void* CurrentStream(int device) {
    const auto stream = py::module_::import("torch").attr("cuda").attr(
        "current_stream")(device);
    return reinterpret_cast<void*>(
        stream.attr("cuda_stream").cast<std::uintptr_t>());
  }
  std::optional<int> stream_device_;
#endif
  std::optional<std::uintptr_t> requested_stream_;
  Handle handle_;
  py::object tensor_type_{py::module_::import("torch").attr("Tensor")};

  std::vector<py::object> owners_;
};

}  // namespace infini::ops

#endif
