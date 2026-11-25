"""Exercise the installed native CPU sparse operators through both public APIs."""

from pathlib import Path

import infini.ops as ops
import pytest
import torch

from tests.test_cpp_api import _compile_cpp, _run


@pytest.fixture(scope="module", autouse=True)
def native_cpu_build(request):
    devices = request.config.getoption("--devices")

    if devices and "cpu" not in devices:
        pytest.skip("native CPU sparse tests require CPU selection")

    if not any(
        0 in cls.active_implementation_indices("cpu")
        for cls in vars(ops).values()
        if hasattr(cls, "active_implementation_indices")
    ):
        pytest.skip("no native CPU implementation is enabled in this build")


FORMATS = [
    pytest.param(torch.sparse_coo, torch.int64, id="coo64"),
    pytest.param(torch.sparse_csr, torch.int32, id="csr32"),
    pytest.param(torch.sparse_csr, torch.int64, id="csr64"),
]


def _matrix(layout, index_type, dtype):
    # Include a duplicate COO coordinate, an explicit zero and trailing empty axes.
    delta = 2.0**-35
    if layout == torch.sparse_coo:
        return torch.sparse_coo_tensor(
            [[2, 0, 0, 0], [0, 3, 1, 1]],
            torch.tensor([-3, 0, 3 + delta, -1], dtype=dtype),
            (4, 5),
            is_coalesced=False,
            check_invariants=True,
        )

    return torch.sparse_csr_tensor(
        torch.tensor([0, 2, 2, 3, 3], dtype=index_type),
        torch.tensor([1, 3, 0], dtype=index_type),
        torch.tensor([2 + delta, 0, -3], dtype=dtype),
        (4, 5),
        check_invariants=True,
    )


def _components(tensor):
    if tensor.layout == torch.sparse_coo:
        return tensor._indices(), tensor._values()

    return tensor.crow_indices(), tensor.col_indices(), tensor.values()


@pytest.mark.parametrize("layout,index_type", FORMATS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("width", [0, 1, 7])
def test_fixed_outputs(layout, index_type, dtype, width):
    mat = _matrix(layout, index_type, dtype)
    originals = [part.clone() for part in _components(mat)]
    dense = torch.full(mat.shape, float("nan"), dtype=dtype)
    dense_ptr = dense.data_ptr()
    rhs = torch.arange(5 * width, dtype=dtype).reshape(5, width)
    guarded = torch.full((4 * width + 2,), -99, dtype=dtype)
    product = guarded[1:-1].view(4, width)
    product_ptr = product.data_ptr()

    ops.to_dense(mat, dense)
    ops.sparse_mm(mat, rhs, "sum", out=product, implementation_index=0)
    torch.testing.assert_close(dense, mat.to_dense(), rtol=0, atol=0)
    torch.testing.assert_close(product, torch.sparse.mm(mat, rhs), rtol=0, atol=0)
    assert guarded[0] == guarded[-1] == -99
    assert (dense.data_ptr(), product.data_ptr()) == (dense_ptr, product_ptr)

    for current, original in zip(_components(mat), originals, strict=True):
        assert torch.equal(current, original)

    if layout == torch.sparse_coo:
        assert not mat.is_coalesced()


@pytest.mark.parametrize("layout,index_type", FORMATS)
@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (3, 4), (0, 0)])
def test_zero_stored_entries(layout, index_type, shape):
    rows, inner = shape
    values = torch.empty(0, dtype=torch.float64)

    if layout == torch.sparse_coo:
        mat = torch.sparse_coo_tensor(
            torch.empty(2, 0, dtype=index_type), values, shape, check_invariants=True
        )
    else:
        mat = torch.sparse_csr_tensor(
            torch.zeros(rows + 1, dtype=index_type),
            torch.empty(0, dtype=index_type),
            values,
            shape,
            check_invariants=True,
        )

    dense = torch.full(shape, -1, dtype=values.dtype)
    product = torch.full((rows, 2), -1, dtype=values.dtype)
    ops.to_dense(mat, dense, implementation_index=0)
    ops.sparse_mm(mat, torch.ones(inner, 2, dtype=values.dtype), "sum", product)
    assert torch.equal(dense, torch.zeros_like(dense))
    assert torch.equal(product, torch.zeros_like(product))


@pytest.mark.parametrize("layout,index_type", FORMATS)
def test_structure_and_value_changes(layout, index_type):
    mat = _matrix(layout, index_type, torch.float32)
    dense = torch.empty(4, 5)
    rhs = torch.arange(5.0).reshape(5, 1)
    product = torch.empty(4, 1)

    for col, value in [(3, -3), (2, 7), (4, 0)]:
        if layout == torch.sparse_coo:
            mat._indices()[1, 0] = col
            mat._values()[0] = value
        else:
            mat.col_indices()[-1] = col
            mat.values()[-1] = value

        ops.to_dense(mat, dense)
        ops.sparse_mm(mat, rhs, "sum", product)
        assert torch.equal(dense, mat.to_dense())
        assert torch.equal(product, torch.sparse.mm(mat, rhs))


@pytest.mark.parametrize("layout,index_type", FORMATS)
def test_equal_metadata_different_indices(layout, index_type):
    first = _matrix(layout, index_type, torch.float64)

    if layout == torch.sparse_coo:
        canonical = first.coalesce()
        # PyTorch can retain the original row stride after merging duplicates.
        first = torch.sparse_coo_tensor(
            canonical.indices().contiguous(),
            canonical.values(),
            canonical.shape,
            is_coalesced=True,
            check_invariants=True,
        )

    second = first.clone()

    if layout == torch.sparse_coo:
        second._indices()[1, -1] = 4
    else:
        second.col_indices()[-1] = 4

    dense = torch.empty(4, 5, dtype=torch.float64)
    rhs = torch.arange(5, dtype=torch.float64).view(5, 1)
    product = torch.empty(4, 1, dtype=torch.float64)

    for mat in (first, second, first):
        ops.to_dense(mat, dense)
        ops.sparse_mm(mat, rhs, "sum", product)
        assert torch.equal(dense, mat.to_dense())
        assert torch.equal(product, torch.sparse.mm(mat, rhs))


@pytest.mark.parametrize("layout,index_type", FORMATS)
def test_explicit_zero_is_multiplied(layout, index_type):
    rhs = torch.tensor([[float("nan")], [float("inf")]])
    out = torch.empty(2, 1)

    if layout == torch.sparse_coo:
        mat = torch.sparse_coo_tensor([[0], [1]], [0.0], (2, 2), check_invariants=True)
    else:
        mat = torch.sparse_csr_tensor(
            torch.tensor([0, 1, 1], dtype=index_type),
            torch.tensor([1], dtype=index_type),
            [0.0],
            (2, 2),
            check_invariants=True,
        )

    ops.sparse_mm(mat, rhs, "sum", out)
    assert torch.isnan(out[0, 0])
    assert out[1, 0] == 0


def test_capability_shape_and_storage_errors():
    mat = _matrix(torch.sparse_coo, torch.int64, torch.float32)
    rhs = torch.ones(5, 2)
    out = torch.full((4, 2), 17.0)

    with pytest.raises(NotImplementedError, match="UnsupportedLayout"):
        ops.to_dense(torch.zeros_like(out).to_sparse_csc(), out)
    with pytest.raises(NotImplementedError, match=r"UnsupportedCapability.*argument 1"):
        ops.sparse_mm(mat, rhs.to_sparse(), "sum", out)
    with pytest.raises(NotImplementedError, match=r"UnsupportedCapability.*argument 2"):
        ops.sparse_mm(mat, rhs, "mean", out)
    with pytest.raises(ValueError, match=r"InvalidMetadata.*argument 1"):
        ops.sparse_mm(mat, rhs.double(), "sum", out)
    with pytest.raises(ValueError, match=r"InvalidMetadata.*argument 1"):
        ops.sparse_mm(mat, rhs[:3], "sum", out)
    with pytest.raises(ValueError, match=r"InvalidMetadata.*argument 3"):
        ops.sparse_mm(mat, rhs, "sum", out.view(2, 4))
    with pytest.raises(NotImplementedError, match="UnsupportedCapability"):
        ops.sparse_mm(mat, rhs.t(), "sum", out)
    with pytest.raises(ValueError, match="UnsupportedImplementation"):
        ops.sparse_mm(mat, rhs, "sum", out, implementation_index=8)
    with pytest.raises(NotImplementedError, match="detach"):
        ops.sparse_mm(mat, rhs.detach().requires_grad_(), "sum", out)
    with pytest.raises(ValueError, match="Aliasing"):
        ops.sparse_mm(mat, rhs, "sum", rhs[1:])

    assert torch.all(out == 17)
    assert torch.all(rhs == 1)


def test_coalesced_with_noncontiguous_indices():
    backing = torch.tensor([[0, 0, 1, -1], [0, 2, 1, -1]])
    mat = torch.sparse_coo_tensor(
        backing[:, :3],
        [1.0, 2.0, 3.0],
        (2, 3),
        is_coalesced=True,
        check_invariants=True,
    )
    out = torch.full((2, 3), 17.0)
    assert mat.indices().stride() == (4, 1)

    with pytest.raises(NotImplementedError, match=r"UnsupportedCapability.*indices"):
        ops.to_dense(mat, out)

    assert torch.all(out == 17)


def test_installed_cpp_consumer(tmp_path):
    source = Path(__file__).parent / "cpp/sparse_cpu.cc"
    binary = _compile_cpp(tmp_path, "sparse_cpu", source.read_text())
    _run([str(binary)])
