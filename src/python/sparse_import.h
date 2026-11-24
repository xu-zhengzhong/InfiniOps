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

}  // namespace detail

inline void CheckSparseResult(ValidationResult result) {
  if (!result.ok()) detail::RaiseSparseError(result);
}

// The binding owns its arguments. Accessors can return temporary Python
// objects, so additionally retain every component through the synchronous call.
class PythonSparseCall {
 public:
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

  static Device ForwardCpuDevice(py::handle obj, std::size_t argument,
                                 Component component) {
    if (obj.attr("requires_grad").cast<bool>()) {
      detail::RaiseSparseError(
          {Code::kAutogradUnsupported, component, {}, argument});
    }
    const auto device = obj.attr("device");
    if (device.attr("type").cast<std::string>() != "cpu") {
      detail::RaiseSparseError(
          {Code::kUnsupportedCapability, component, {}, argument},
          "sparse import requires CPU tensors");
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
    const auto device = ForwardCpuDevice(obj, argument, component);
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
    const auto device = ForwardCpuDevice(tensor, argument, Component::kNone);
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
    if (!out_only) {
      auto status = native::cpu::ValidateSparseIndices(*view);
      status.argument = argument;
      CheckSparseResult(status);
    }
    return *view;
  }

  py::object tensor_type_{py::module_::import("torch").attr("Tensor")};

  std::vector<py::object> owners_;
};

}  // namespace infini::ops

#endif
