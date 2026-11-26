"""Inspect the ATen interoperability boundary in a Torch-enabled CPU install."""

import infini.ops as ops
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tests.test_sparse_cpu import CPU_SLOTS, FORMATS, _components, _csr_out, _matrix


@pytest.fixture(scope="module", autouse=True)
def aten_cpu_build(request):
    devices = request.config.getoption("--devices")

    if 8 not in CPU_SLOTS or (devices and "cpu" not in devices):
        pytest.skip("requires a Torch-enabled CPU build")


@pytest.mark.parametrize("layout,index_type", FORMATS)
def test_aten_borrows_raw_components(layout, index_type):
    mat = _matrix(layout, index_type, torch.float32)
    pointers = [part.data_ptr() for part in _components(mat)]
    observed = []

    class Inspect(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten._to_dense.default:
                source = args[0]
                assert source.layout == layout and source.shape == mat.shape
                assert source.device == mat.device and source.dtype == mat.dtype
                assert source._nnz() == mat._nnz()

                if layout == torch.sparse_coo:
                    assert not source.is_coalesced()

                observed.append([part.data_ptr() for part in _components(source)])

            return func(*args, **(kwargs or {}))

    out = torch.empty(mat.shape)
    with Inspect(), torch.autocast("cpu"):
        ops.to_dense(mat, out, implementation_index=8)

    assert observed == [pointers]
    torch.testing.assert_close(out, mat.to_dense())


@pytest.mark.parametrize("index_type", [torch.int32, torch.int64])
def test_aten_reallocated_csr_is_copied_to_caller(index_type):
    mat = _matrix(torch.sparse_csr, index_type, torch.float64)
    lhs = torch.ones(4, 2, dtype=mat.dtype)
    rhs = torch.ones(2, 5, dtype=mat.dtype)
    out = _csr_out(mat.shape, mat._nnz(), index_type, mat.dtype)
    pointers = [part.data_ptr() for part in _components(out)]
    expected = torch.sparse.sampled_addmm(mat, lhs, rhs, out=mat.clone())
    reallocated = []

    class ResizeResult(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))

            if func == torch.ops.aten.sparse_sampled_addmm.out:
                saved = result.clone()
                before = [part.data_ptr() for part in _components(result)]
                assert set(before).isdisjoint(pointers)
                result.resize_as_sparse_(
                    torch.ones(16, 16, dtype=mat.dtype).to_sparse_csr()
                )
                result.resize_as_sparse_(saved)
                result.copy_(saved)
                reallocated.append(
                    before != [part.data_ptr() for part in _components(result)]
                )

            return result

    with ResizeResult():
        ops.sparse_sampled_addmm(mat, lhs, rhs, 1, 1, out, implementation_index=8)

    assert reallocated == [True]
    assert [part.data_ptr() for part in _components(out)] == pointers

    for actual, reference in zip(_components(out), _components(expected), strict=True):
        torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize(
    "exception,public_error,code",
    [
        (RuntimeError, RuntimeError, "ExecutionFailure"),
        (NotImplementedError, NotImplementedError, "UnsupportedCapability"),
        (
            getattr(torch, "OutOfMemoryError", MemoryError),
            MemoryError,
            "AllocationFailure",
        ),
    ],
)
def test_aten_failure_and_retry(exception, public_error, code):
    mat = _matrix(torch.sparse_coo, torch.int64, torch.float32)
    out = torch.full(mat.shape, -17.0)
    builder = ops.prepare_coalesce(mat, implementation_index=8)
    indices = torch.full((2, mat._nnz()), -17, dtype=torch.int64)
    values = torch.full((mat._nnz(),), -17.0)

    class FailKernel(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func in (
                torch.ops.aten._coalesce.default,
                torch.ops.aten._to_dense.default,
            ):
                raise exception("Injected ATen kernel failure.")

            return func(*args, **(kwargs or {}))

    with FailKernel():
        with pytest.raises(public_error, match=code):
            ops.to_dense(mat, out, implementation_index=8)

        builder.execute(indices=indices, values=values)

    with pytest.raises(public_error, match=code):
        builder.finalize()

    assert (out == -17).all() and (indices == -17).all() and (values == -17).all()
    builder.execute(indices=indices, values=values)
    torch.testing.assert_close(builder.finalize().to_dense(), mat.to_dense())
