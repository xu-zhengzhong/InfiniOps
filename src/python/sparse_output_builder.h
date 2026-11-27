#ifndef INFINI_OPS_PYTHON_SPARSE_OUTPUT_BUILDER_H_
#define INFINI_OPS_PYTHON_SPARSE_OUTPUT_BUILDER_H_

#include "python/sparse_import.h"
#include "sparse_output.h"

namespace infini::ops {

class PythonSparseOutput {
 public:
  using Prepare = ValidationResult (*)(const Handle&, const Config&,
                                       const SparseTensor&,
                                       const SparsePrepareOptions&,
                                       SparseOutputPlan&);

  PythonSparseOutput(Prepare prepare, const PySparseTensor& input,
                     std::optional<std::size_t> implementation,
                     bool count_first, bool capture)
      : input_(input) {
    const auto view = owners_.Import<SparseTensor>(input, 0, /*out_only=*/true);
    Config config;
    if (implementation) config.set_implementation_index(*implementation);
    const SparsePrepareOptions options{count_first
                                           ? SparseCapacityMode::kCountFirst
                                           : SparseCapacityMode::kUpperBound,
                                       capture};
    CheckSparseResult(prepare(owners_.handle(), config, view, options, plan_));
    shape_ = view.shape();
    dtype_ = view.dtype();
    device_ = view.device();
    if (!input_.attr("is_inference")().cast<bool>())
      version_ = input_.attr("_version").cast<std::uint64_t>();
  }

  py::dict Requirements() const {
    const auto& req = plan_.requirements();
    const auto size_align = [](SparseBufferRequirement r) {
      return py::make_tuple(r.bytes, r.alignment);
    };
    py::dict result;
    result["nnz_upper_bound"] = req.nnz_upper_bound;
    result["indices"] = size_align(req.indices);
    result["col_indices"] = size_align(req.col_indices);
    result["values"] = size_align(req.values);
    result["workspace"] = size_align(req.workspace);
    result["result"] = size_align(req.result);
    result["result_copy_bytes_upper_bound"] = req.result_copy_bytes_upper_bound;
    return result;
  }

  void Execute(py::object indices, py::object col_indices, py::object values,
               py::object workspace) {
    buffers_.reset();
    owners_.CheckStream();
    CheckInput();
    const auto& req = plan_.requirements();
    const auto i64 = torch_.attr("int64");
    const bool csr = plan_.output_layout() == TensorLayout::kCsr;
    const auto value_type =
        torch_.attr(dtype_ == DataType::kFloat32 ? "float32" : "float64");
    if (!csr && !col_indices.is_none())
      detail::RaiseSparseError(
          {ValidationCode::kInvalidMetadata, ValidationComponent::kColIndices});
    Buffers buffers;
    buffers.indices = Buffer(std::move(indices), i64,
                             req.indices.bytes / sizeof(std::int64_t),
                             csr ? ValidationComponent::kCrowIndices
                                 : ValidationComponent::kIndices);
    if (csr)
      buffers.columns = Buffer(std::move(col_indices), i64,
                               req.col_indices.bytes / sizeof(std::int64_t),
                               ValidationComponent::kColIndices);
    buffers.values = Buffer(std::move(values), value_type, req.nnz_upper_bound,
                            ValidationComponent::kValues);
    auto scratch = Buffer(std::move(workspace), torch_.attr("uint8"),
                          req.workspace.bytes, ValidationComponent::kNone);
    const SparseOutputBuffer out{
        plan_.output_layout(),
        shape_,
        dtype_,
        DataType::kInt64,
        device_,
        Span(buffers.indices),
        csr ? Span(buffers.columns) : SparseBufferSpan{},
        Span(buffers.values)};
    const SparseWorkspace ws{Span(scratch), device_};
    // Python allocation and tensor hooks can run between Prepare and Execute.
    CheckInput();
    CheckSparseResult(plan_.Execute(out, ws, &result_));
    if (result_.state.load() == SparseCompletion::kFailed)
      result_.error = detail::ConsumeProviderPythonError(result_.error);
    buffers_.emplace(std::move(buffers));
  }

  py::object Finalize() const {
    if (!buffers_) detail::RaiseSparseError({ValidationCode::kInvalidState});
    std::optional<SparseTensor> view;
    const auto completion = plan_.Finalize(view);
    CheckSparseResult(completion.error);
    if (completion.state == SparseCompletion::kPending)
      detail::RaiseSparseError({ValidationCode::kInvalidState},
                               "synchronous sparse result is pending");
    const auto n = view->nnz();
    const auto prefix = [](const py::object& storage, std::size_t count) {
      return storage.attr("narrow")(0, 0, count);
    };
    auto values = prefix(buffers_->values, n);
    auto size = py::make_tuple(shape_[0], shape_[1]);
    if (view->layout() == TensorLayout::kCsr) {
      return torch_.attr("sparse_csr_tensor")(
          prefix(buffers_->indices, shape_[0] + 1),
          prefix(buffers_->columns, n), values, size,
          py::arg("dtype") = values.attr("dtype"),
          py::arg("device") = values.attr("device"),
          py::arg("check_invariants") = false);
    }
    auto indices = prefix(buffers_->indices, 2 * n).attr("view")(2, n);
    return torch_.attr("sparse_coo_tensor")(
        indices, values, size, py::arg("is_coalesced") = true,
        py::arg("dtype") = values.attr("dtype"),
        py::arg("device") = values.attr("device"),
        py::arg("check_invariants") = false);
  }

 private:
  struct Buffers {
    py::object indices, columns, values;
  };

  void CheckInput() const {
    if (input_.attr("requires_grad").cast<bool>())
      detail::RaiseSparseError({ValidationCode::kAutogradUnsupported});
    if (version_ && input_.attr("_version").cast<std::uint64_t>() != *version_)
      detail::RaiseSparseError({ValidationCode::kStaleStructure});
  }

  py::object Buffer(py::object obj, py::object dtype, std::size_t length,
                    ValidationComponent component) const {
    using Code = ValidationCode;
    using Component = ValidationComponent;
    const bool scratch = component == Component::kNone;
    if (obj.is_none()) {
      try {
        return torch_.attr("empty")(length, py::arg("dtype") = dtype,
                                    py::arg("device") = input_.attr("device"));
      } catch (py::error_already_set& error) {
        if (error.matches(torch_.attr("OutOfMemoryError")))
          detail::RaiseSparseError({Code::kAllocationFailure, component});
        throw;
      }
    }
    if (!py::isinstance(obj, torch_.attr("Tensor")) ||
        detail::GetPythonLayout(obj) != detail::PythonLayout::kStrided)
      detail::RaiseSparseError(
          {scratch ? Code::kInvalidWorkspace : Code::kUnsupportedLayout,
           component});
    if (obj.attr("requires_grad").cast<bool>())
      detail::RaiseSparseError({Code::kAutogradUnsupported, component});
    if (!obj.attr("device").equal(input_.attr("device")))
      detail::RaiseSparseError(
          {scratch ? Code::kInvalidWorkspace : Code::kDeviceMismatch,
           component});
    const py::object actual_dtype = obj.attr("dtype");
    if (!actual_dtype.is(dtype)) {
      auto code = scratch ? Code::kInvalidWorkspace : Code::kInvalidIndexType;
      if (component == Component::kValues)
        code = Code::kInvalidMetadata;
      else if (!scratch && plan_.output_layout() == TensorLayout::kCsr &&
               actual_dtype.is(torch_.attr("int32")))
        code = Code::kUnsupportedCapability;
      detail::RaiseSparseError({code, component});
    }
    if (!obj.attr("is_contiguous")().cast<bool>() ||
        obj.attr("is_neg")().cast<bool>())
      detail::RaiseSparseError(
          {scratch ? Code::kInvalidWorkspace : Code::kUnsupportedCapability,
           component});
    return obj.attr("view")(-1);
  }

  static SparseBufferSpan Span(const py::object& tensor) {
    return {reinterpret_cast<void*>(
                tensor.attr("data_ptr")().cast<std::uintptr_t>()),
            tensor.attr("numel")().cast<std::size_t>() *
                tensor.attr("element_size")().cast<std::size_t>()};
  }

  py::object torch_{py::module_::import("torch")};
  py::object input_;
  PythonSparseCall owners_;
  SparseTensor::Shape shape_;
  DataType dtype_;
  Device device_;
  std::optional<std::uint64_t> version_;
  SparseOutputPlan plan_;
  SparseResultMetadata result_;
  std::optional<Buffers> buffers_;
};

template <typename Op>
void BindSparseOutput(py::module_& m, const char* name) {
  if (!py::hasattr(m, "SparseOutputBuilder")) {
    py::class_<PythonSparseOutput>(m, "SparseOutputBuilder")
        .def_property_readonly("requirements",
                               &PythonSparseOutput::Requirements)
        .def("execute", &PythonSparseOutput::Execute, py::kw_only(),
             py::arg("indices") = py::none(),
             py::arg("col_indices") = py::none(),
             py::arg("values") = py::none(), py::arg("workspace") = py::none())
        .def("finalize", &PythonSparseOutput::Finalize);
  }
  const auto prepare = [](const py::object& input,
                          std::optional<std::size_t> implementation,
                          bool count_first, bool capture) {
    if (!PySparseTensor::check_(input))
      detail::RaiseSparseError({ValidationCode::kUnsupportedLayout,
                                ValidationComponent::kLayout,
                                {},
                                0});
    return std::make_unique<PythonSparseOutput>(
        &Op::Prepare, py::reinterpret_borrow<PySparseTensor>(input),
        implementation, count_first, capture);
  };
  m.def((std::string("prepare_") + name).c_str(), prepare, py::arg("input"),
        py::kw_only(), py::arg("implementation_index") = py::none(),
        py::arg("count_first") = false, py::arg("capture") = false);
  m.def(
      name,
      [prepare](const py::object& input,
                std::optional<std::size_t> implementation) {
        auto builder = prepare(input, implementation, false, false);
        builder->Execute(py::none(), py::none(), py::none(), py::none());
        return builder->Finalize();
      },
      py::arg("input"), py::kw_only(),
      py::arg("implementation_index") = py::none());
}

}  // namespace infini::ops

#endif  // INFINI_OPS_PYTHON_SPARSE_OUTPUT_BUILDER_H_
