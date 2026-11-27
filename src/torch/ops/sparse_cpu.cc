#ifdef WITH_CPU

#include "native/cpu/sparse_output_provider.h"
#include "torch/ops/coalesce/coalesce.h"
#include "torch/ops/sparse_mm/sparse_mm.h"
#include "torch/ops/sparse_sampled_addmm/sparse_sampled_addmm.h"
#include "torch/ops/to_dense/to_dense.h"
#include "torch/ops/to_sparse_coo/to_sparse_coo.h"
#include "torch/ops/to_sparse_csr/to_sparse_csr.h"
#include "torch/sparse_tensor.h"

namespace infini::ops {
namespace {

ValidationResult ExecuteAtenOutput(void* stream, const SparseTensor& input,
                                   const SparseOutputBuffer& out,
                                   SparseBufferSpan, std::size_t& actual) {
  return aten::WithSparseInput(stream, input, [&](const at::Tensor& source) {
    auto canonical = source.coalesce();
    auto result = out.layout == TensorLayout::kCsr ? canonical.to_sparse_csr()
                                                   : canonical;
    const auto nnz = static_cast<std::size_t>(result._nnz());
    // Execute checked all capacities against `input.nnz()` before submission.
    if (nnz > input.nnz())
      return ValidationResult{ValidationCode::kInsufficientCapacity};
    std::optional<SparseTensor> target;
    auto status = detail::MakeSparseOutputView(out, nnz, target);
    if (status) status = aten::CopySparseResult(result, *target);
    if (status) actual = nnz;
    return status;
  });
}

ValidationResult PrepareAtenOutput(const Handle& handle,
                                   const SparseTensor& input,
                                   TensorLayout layout,
                                   SparseOutputPlan* plan) {
  auto status = aten::CheckAtenSize(input);
  if (!status) return status;
  return native::cpu::SparseOutputProvider::PrepareWithKernel(
      input, layout, plan, ExecuteAtenOutput, handle.stream(),
      /*workspace=*/{}, /*copies_result=*/true);
}

}  // namespace

ValidationResult Operator<Coalesce, Device::Type::kCpu, 8>::checked_call(
    const Handle& handle, const Config&, const SparseTensor& input,
    SparseOutputPlan* plan) {
  return PrepareAtenOutput(handle, input, TensorLayout::kCoo, plan);
}

ValidationResult Operator<ToSparseCsr, Device::Type::kCpu, 8>::checked_call(
    const Handle& handle, const Config&, const SparseTensor& input,
    SparseOutputPlan* plan) {
  return PrepareAtenOutput(handle, input, TensorLayout::kCsr, plan);
}

ValidationResult Operator<ToDense, Device::Type::kCpu, 8>::checked_call(
    const Handle& handle, const Config&, const SparseTensor& input,
    const Tensor& out) {
  auto status = detail::CheckSparseWrites(input, {out});
  if (!status) return status;
  return aten::WithSparseInput(
      handle.stream(), input, [&](const at::Tensor& source) {
        return aten::CopyDenseResult(source.to_dense(), out);
      });
}

ValidationResult Operator<SparseMm, Device::Type::kCpu, 8>::checked_call(
    const Handle& handle, const Config&, const SparseTensor& input,
    const Tensor& mat2, const std::string& reduce, const Tensor& out) {
  auto status = detail::CheckSparseWrites(input, {out}, {mat2});
  if (status) status = aten::CheckAtenSize(mat2, 1);
  if (!status) return status;
  return aten::WithSparseInput(
      handle.stream(), input, [&](const at::Tensor& source) {
        auto rhs = aten::BorrowCpuTensor(mat2);
        // The explicit sum kernel accepts CSR in either index width. COO uses
        // ATen's two-argument sparse mm, whose reduction is always sum.
        auto result = source.is_sparse_csr()
                          ? at::_sparse_mm(source, rhs, reduce)
                          : at::_sparse_mm(source, rhs);
        return aten::CopyDenseResult(result, out);
      });
}

ValidationResult Operator<ToSparseCoo, Device::Type::kCpu, 8>::checked_call(
    const Handle& handle, const Config&, const SparseTensor& input,
    const SparseTensor& out) {
  const detail::SparseComponents dst(out);
  auto status = detail::CheckSparseWrites(input, {*dst.indices, *dst.values});
  if (!status) return status;
  return aten::WithSparseInput(
      handle.stream(), input, [&](const at::Tensor& source) {
        return aten::CopySparseResult(source.to_sparse(at::kSparse), out);
      });
}

ValidationResult
Operator<SparseSampledAddmm, Device::Type::kCpu, 8>::checked_call(
    const Handle& handle, const Config&, const SparseTensor& input,
    const Tensor& mat1, const Tensor& mat2, double beta, double alpha,
    const SparseTensor& out) {
  const detail::SparseComponents dst(out);
  auto status = detail::CheckSparseWrites(
      input, {*dst.indices, *dst.columns, *dst.values}, {mat1, mat2}, 5);
  if (status) status = aten::CheckAtenSize(mat1, 1);
  if (!status) return status;
  // The checked API casts both scalars to the value dtype. ATen rejects a
  // finite double outside the float range, but accepts the converted infinity.
  if (input.dtype() == DataType::kFloat32) {
    beta = static_cast<float>(beta);
    alpha = static_cast<float>(alpha);
  }
  return aten::WithSparseInput(
      handle.stream(), input, [&](const at::Tensor& source) {
        // ATen's functional result can widen CSR indices. Computing on an owned
        // clone preserves the input index dtype even if the kernel resizes it.
        auto result = source.clone();
        at::sparse_sampled_addmm_out(result, result,
                                     aten::BorrowCpuTensor(mat1),
                                     aten::BorrowCpuTensor(mat2), beta, alpha);
        return aten::CopySparseResult(result, out);
      });
}

}  // namespace infini::ops

#endif  // WITH_CPU
