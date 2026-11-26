"""Exercise the installed native CPU sparse operators through both public APIs."""

import gc
import weakref
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


@pytest.mark.parametrize("name", ["sparse_cpu", "sparse_output"])
def test_installed_cpp_consumer(tmp_path, name):
    source = Path(__file__).parent / f"cpp/{name}.cc"
    binary = _compile_cpp(tmp_path, name, source.read_text())
    _run([str(binary)])


def _coo_out(shape, nnz, dtype):
    # Output-only descriptors may contain indices that have never been written.
    return torch.sparse_coo_tensor(
        torch.full((2, nnz), -17, dtype=torch.int64),
        torch.full((nnz,), -17, dtype=dtype),
        shape,
        is_coalesced=True,
        check_invariants=False,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dynamic_conversion_roundtrip(dtype):
    mat = _matrix(torch.sparse_coo, torch.int64, dtype)
    original = [part.clone() for part in _components(mat)]
    expected = mat.coalesce()
    coo = ops.coalesce(mat)
    csr = ops.to_sparse_csr(mat, implementation_index=0)
    rebuilt = _coo_out(mat.shape, csr._nnz(), dtype)
    ptrs = [part.data_ptr() for part in _components(rebuilt)]
    ops.to_sparse_coo(csr, out=rebuilt)
    assert [part.data_ptr() for part in _components(rebuilt)] == ptrs
    assert torch.equal(csr.crow_indices(), torch.tensor([0, 2, 2, 3, 3]))
    assert csr.crow_indices().dtype == csr.col_indices().dtype == torch.int64

    for result in (coo, rebuilt):
        assert result.is_coalesced() and result.indices().is_contiguous()
        assert result.shape == mat.shape and result._nnz() == 3
        assert torch.equal(result.indices(), expected.indices())
        torch.testing.assert_close(result.values(), expected.values(), rtol=0, atol=0)
        assert torch.equal(result.to_dense(), mat.to_dense())

    for part, saved in zip(_components(mat), original, strict=True):
        assert torch.equal(part, saved)

    assert not mat.is_coalesced()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("index_type", [torch.int32, torch.int64])
def test_csr_to_coo_index_width(dtype, index_type):
    mat = _matrix(torch.sparse_csr, index_type, dtype)
    out = _coo_out(mat.shape, mat._nnz(), dtype)
    ops.to_sparse_coo(mat, out)
    ref = mat.to_sparse_coo()
    assert out.indices().dtype == torch.int64
    assert torch.equal(out.indices(), ref.indices())
    assert torch.equal(out.values(), ref.values())


def _csr_out(shape, nnz, index_type, dtype):
    return torch.sparse_csr_tensor(
        torch.full((shape[0] + 1,), -17, dtype=index_type),
        torch.full((nnz,), -17, dtype=index_type),
        torch.full((nnz,), -17, dtype=dtype),
        shape,
        check_invariants=False,
    )


@pytest.mark.parametrize("index_type", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sampled_addmm_stored_zero(index_type, dtype):
    mat = torch.sparse_csr_tensor(
        torch.tensor([0, 2], dtype=index_type),
        torch.tensor([0, 1], dtype=index_type),
        torch.tensor([0, 2], dtype=dtype),
        (1, 2),
        check_invariants=True,
    )
    lhs = torch.tensor([[2, 3]], dtype=dtype)
    rhs = torch.tensor([[4, 5], [6, 7]], dtype=dtype)
    out = _csr_out(mat.shape, mat._nnz(), index_type, dtype)
    pointers = [part.data_ptr() for part in _components(out)]
    originals = [part.clone() for part in _components(mat)]
    # PyTorch preserves int32 indices when given an explicit int32 output.
    ref = torch.sparse.sampled_addmm(mat, lhs, rhs, out=mat.clone())
    assert ops.sparse_sampled_addmm(mat, lhs, rhs, beta=1, alpha=1, out=out) is None
    assert out.layout == torch.sparse_csr and out.shape == mat.shape and out._nnz() == 2
    assert [part.data_ptr() for part in _components(out)] == pointers
    assert out.values().tolist() == [26, 33]

    for actual, expected in zip(_components(out), _components(ref), strict=True):
        torch.testing.assert_close(actual, expected)

    for actual, original in zip(_components(mat), originals, strict=True):
        torch.testing.assert_close(actual, original)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "value,left,right,beta,alpha,expected",
    [
        (float("nan"), [1], [1], 0, 1, float("nan")),
        (float("inf"), [1], [1], 0, 1, float("nan")),
        (2, [float("inf")], [1], 1, 0, float("nan")),
        (2, [1], [float("nan")], 1, 0, float("nan")),
        (2, [3], [1], -1, -2, -8),
        (2, [], [], 3, float("nan"), 6),
        (float("inf"), [], [], 0, 1, float("nan")),
    ],
)
def test_sampled_addmm_special_values(dtype, value, left, right, beta, alpha, expected):
    mat = torch.sparse_csr_tensor([0, 1], [0], [value], (1, 1), dtype=dtype)
    lhs = torch.tensor(left, dtype=dtype).reshape(1, -1)
    rhs = torch.tensor(right, dtype=dtype).reshape(-1, 1)
    out = _csr_out(mat.shape, 1, torch.int64, dtype)
    ops.sparse_sampled_addmm(mat, lhs, rhs, beta, alpha, out, implementation_index=0)
    ref = torch.sparse.sampled_addmm(mat, lhs, rhs, beta=beta, alpha=alpha)
    torch.testing.assert_close(out.values(), ref.values(), equal_nan=True)
    torch.testing.assert_close(
        out.values(), torch.tensor([expected], dtype=dtype), equal_nan=True
    )
    assert torch.equal(out.crow_indices(), mat.crow_indices())
    assert torch.equal(out.col_indices(), mat.col_indices())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sampled_addmm_casts_scalars_to_value_dtype(dtype):
    mat = torch.ones(1, 1, dtype=dtype).to_sparse_csr()
    operand = torch.ones(1, 1, dtype=dtype)
    out = _csr_out(mat.shape, 1, torch.int64, dtype)
    delta = 2.0**-25
    for beta, alpha in [(1 + delta, -1), (-1, 1 + delta)]:
        ops.sparse_sampled_addmm(mat, operand, operand, beta, alpha, out)
        assert out.values().item() == (0 if dtype == torch.float32 else delta)


@pytest.mark.parametrize("shape", [(0, 4), (3, 0), (3, 4), (0, 0)])
def test_sampled_addmm_empty_pattern(shape):
    mat = torch.zeros(shape, dtype=torch.float64).to_sparse_csr()
    out = _csr_out(shape, 0, mat.crow_indices().dtype, mat.dtype)
    lhs = torch.ones(shape[0], 2, dtype=mat.dtype)
    rhs = torch.ones(2, shape[1], dtype=mat.dtype)
    ops.sparse_sampled_addmm(mat, lhs, rhs, 1, 1, out)
    assert out.shape == shape and out._nnz() == 0
    assert torch.equal(out.crow_indices(), mat.crow_indices())
    assert out.col_indices().numel() == out.values().numel() == 0


def test_sampled_addmm_reuses_out_and_revalidates_pattern():
    mat = _matrix(torch.sparse_csr, torch.int32, torch.float64)
    lhs = torch.randn(4, 9, dtype=mat.dtype)
    rhs = torch.randn(9, 5, dtype=mat.dtype)
    out = _csr_out(mat.shape, mat._nnz(), torch.int32, mat.dtype)

    for column in (0, 4):
        mat.col_indices()[-1] = column
        ref = torch.sparse.sampled_addmm(
            mat, lhs, rhs, beta=-0.5, alpha=2, out=mat.clone()
        )
        ops.sparse_sampled_addmm(mat, lhs, rhs, -0.5, 2, out)
        for actual, expected in zip(_components(out), _components(ref), strict=True):
            torch.testing.assert_close(actual, expected)

    saved = [part.clone() for part in _components(out)]
    mat.col_indices()[-1] = 5
    with pytest.raises(IndexError, match=r"IndexOutOfBounds.*argument 0.*col_indices"):
        ops.sparse_sampled_addmm(mat, lhs, rhs, -0.5, 2, out)

    for actual, expected in zip(_components(out), saved, strict=True):
        assert torch.equal(actual, expected)


def test_sampled_addmm_rejects_unsupported_arguments():
    mat = _matrix(torch.sparse_csr, torch.int32, torch.float32)
    lhs = torch.ones(4, 2)
    rhs = torch.ones(2, 5)
    out = _csr_out(mat.shape, mat._nnz(), torch.int32, mat.dtype)
    call = ops.sparse_sampled_addmm
    with pytest.raises(NotImplementedError, match=r"UnsupportedCapability.*argument 0"):
        call(mat.to_sparse_coo(), lhs, rhs, 1, 1, out)
    with pytest.raises(NotImplementedError, match=r"UnsupportedCapability.*argument 5"):
        call(mat, lhs, rhs, 1, 1, _coo_out(mat.shape, mat._nnz(), mat.dtype))
    with pytest.raises(TypeError, match=r"InvalidIndexType.*argument 5"):
        call(
            mat, lhs, rhs, 1, 1, _csr_out(mat.shape, mat._nnz(), torch.int64, mat.dtype)
        )
    with pytest.raises(ValueError, match=r"InvalidMetadata.*argument 5"):
        call(mat, lhs, rhs, 1, 1, _csr_out((5, 4), mat._nnz(), torch.int32, mat.dtype))
    with pytest.raises(ValueError, match=r"InvalidMetadata.*argument 5"):
        call(mat, lhs, rhs, 1, 1, _csr_out(mat.shape, 1, torch.int32, mat.dtype))
    with pytest.raises(ValueError, match=r"InvalidMetadata.*argument 1"):
        call(mat, lhs[:3], rhs, 1, 1, out)
    with pytest.raises(ValueError, match=r"InvalidMetadata.*argument 2"):
        call(mat, lhs, rhs[:1], 1, 1, out)
    with pytest.raises(ValueError, match=r"InvalidMetadata.*argument 2"):
        call(mat, lhs, rhs.double(), 1, 1, out)
    with pytest.raises(NotImplementedError, match=r"UnsupportedCapability.*argument 1"):
        call(mat, lhs.view(-1), rhs, 1, 1, out)
    with pytest.raises(NotImplementedError, match=r"UnsupportedCapability.*argument 2"):
        call(mat, lhs, torch.ones(5, 2).t(), 1, 1, out)
    with pytest.raises(NotImplementedError, match="detach"):
        call(mat, lhs, rhs.detach().requires_grad_(), 1, 1, out)
    with pytest.raises(ValueError, match="UnsupportedImplementation"):
        call(mat, lhs, rhs, 1, 1, out, implementation_index=8)
    with pytest.raises(ValueError, match="Aliasing"):
        call(mat, lhs, rhs, 1, 1, mat)
    with pytest.raises(ValueError, match=r"deprecated.*CSR input and out"):
        call(mat.to_dense(), lhs, rhs, 1, 1, torch.zeros(mat.shape))

    assert all(torch.all(part == -17) for part in _components(out))


def test_sampled_addmm_dense_migration_precedes_tensor_import():
    dense = torch.empty((2, 2), device="meta")
    args = (dense, dense, dense, 1, 1, dense)
    for entry in (ops.sparse_sampled_addmm, ops.SparseSampledAddmm):
        with pytest.raises(ValueError, match=r"deprecated.*CSR input and out"):
            entry(*args)


@pytest.mark.parametrize("shape", [(0, 4), (3, 0), (3, 4), (0, 0)])
def test_empty_sparse_conversion(shape):
    empty = torch.sparse_coo_tensor(torch.empty(2, 0, dtype=torch.int64), [], shape)
    coo = ops.coalesce(empty)
    csr = ops.to_sparse_csr(empty)
    out = _coo_out(shape, 0, empty.dtype)
    ops.to_sparse_coo(csr, out)
    assert coo.is_coalesced() and out.is_coalesced()
    assert coo.shape == csr.shape == out.shape == shape
    assert coo._nnz() == csr._nnz() == out._nnz() == 0
    assert torch.equal(csr.crow_indices(), torch.zeros(shape[0] + 1, dtype=torch.int64))


@pytest.mark.parametrize("prepare", ["prepare_coalesce", "prepare_to_sparse_csr"])
def test_builder_packing_and_ownership(prepare):
    mat = torch.sparse_coo_tensor(
        [[2, 1, 0, 3, 1, 0, 2, 2, 1, 0], [1, 1, 3, 2, 0, 1, 1, 1, 0, 3]],
        [1.0, 2, 3, 4, 5, 6, -1, 0, 2, 1],
        (5, 6),
    )
    ref = mat.coalesce()
    builder = getattr(ops, prepare)(mat)
    req = builder.requirements
    assert req["nnz_upper_bound"] == 10
    indices = torch.full((req["indices"][0] // 8 + 2,), -99, dtype=torch.int64)
    values = torch.full((12,), -99.0)
    cols = torch.full((12,), -99, dtype=torch.int64)
    csr = prepare == "prepare_to_sparse_csr"
    builder.execute(
        indices=indices[1:-1],
        values=values[1:-1],
        col_indices=cols[1:-1] if csr else None,
        workspace=torch.empty(req["workspace"][0], dtype=torch.uint8),
    )
    result = builder.finalize()
    assert builder.finalize().values().data_ptr() == result.values().data_ptr()
    assert result._nnz() == 6 and result.values().data_ptr() == values[1:].data_ptr()
    assert values[0] == -99 and torch.all(values[7:] == -99)
    assert indices[0] == indices[-1] == -99

    if csr:
        assert result.crow_indices().data_ptr() == indices[1:].data_ptr()
        assert result.col_indices().data_ptr() == cols[1:].data_ptr()
        assert cols[0] == -99 and torch.all(cols[7:] == -99)
    else:
        assert result.indices().data_ptr() == indices[1:].data_ptr()
        assert result.indices().stride() == (6, 1)
        assert torch.equal(indices[1:13], ref.indices().reshape(-1))
        assert torch.all(indices[13:] == -99)

    del builder, indices, cols, values, mat
    gc.collect()
    torch.empty(1024).fill_(101)
    assert torch.equal(result.values(), ref.values())
    assert torch.equal(result.to_dense(), ref.to_dense())


def test_cancellation_and_already_coalesced_output():
    mat = torch.sparse_coo_tensor([[1, 1], [2, 2]], [1.0, -1], (3, 4))
    coo = ops.coalesce(mat)
    csr = ops.to_sparse_csr(mat)
    assert coo._nnz() == csr._nnz() == 1
    assert torch.equal(coo.indices(), torch.tensor([[1], [2]]))
    assert coo.values().item() == csr.values().item() == 0
    builder = ops.prepare_coalesce(coo)
    assert builder.requirements["workspace"][0] == 0
    builder.execute()
    copied = builder.finalize()
    assert copied.indices().data_ptr() != coo.indices().data_ptr()
    assert copied.values().data_ptr() != coo.values().data_ptr()


def test_documented_coo_example_with_default_device():
    mat = torch.sparse_coo_tensor([[0, 0, 2, 2], [1, 1, 0, 3]], [2.0, 3, 0, -1], (3, 4))

    with torch.device("meta"):
        coo = ops.coalesce(mat)
        csr = ops.to_sparse_csr(mat)

    assert coo.device == csr.device == mat.device
    assert coo._nnz() == csr._nnz() == 3
    assert torch.equal(coo.indices(), torch.tensor([[0, 2, 2], [1, 0, 3]]))
    assert torch.equal(coo.values(), torch.tensor([5.0, 0, -1]))
    assert torch.equal(csr.to_dense(), coo.to_dense())
    assert torch.count_nonzero(coo.to_dense()).item() == 2


def test_builder_rejected_resources_and_stale_input():
    mat = _matrix(torch.sparse_coo, torch.int64, torch.float64)
    builder = ops.prepare_coalesce(mat)
    guarded = torch.full((4,), 13.0, dtype=mat.dtype)

    with pytest.raises(RuntimeError, match="InvalidState"):
        builder.finalize()
    with pytest.raises(ValueError, match="InsufficientCapacity"):
        builder.execute(indices=torch.empty(7, dtype=torch.int64), values=guarded)
    with pytest.raises(ValueError, match="InvalidWorkspace"):
        builder.execute(values=guarded, workspace=torch.empty(1, dtype=torch.uint8))
    with pytest.raises(ValueError, match="Aliasing"):
        builder.execute(indices=mat._indices(), values=guarded)
    with pytest.raises(NotImplementedError, match="UnsupportedCapability"):
        ops.prepare_to_sparse_csr(mat).execute(
            indices=torch.empty(5, dtype=torch.int32)
        )
    with pytest.raises(ValueError, match="UnsupportedImplementation"):
        ops.coalesce(mat, implementation_index=8)
    with pytest.raises(NotImplementedError, match="UnsupportedLayout"):
        ops.to_sparse_csr(torch.zeros(2, 2))
    with pytest.raises(NotImplementedError, match="UnsupportedCapability"):
        ops.prepare_coalesce(mat, count_first=True)
    with pytest.raises(NotImplementedError, match="UnsupportedCapability"):
        ops.prepare_coalesce(mat, capture=True)
    with pytest.raises(NotImplementedError, match="detach"):
        ops.coalesce(mat.detach().requires_grad_())
    with pytest.raises(RuntimeError, match="InvalidState"):
        builder.finalize()

    mat._indices()[1, 0] = 4
    with pytest.raises(RuntimeError, match="StaleStructure"):
        builder.execute(values=guarded)

    assert torch.all(guarded == 13)


def test_builder_releases_partial_allocation(monkeypatch):
    mat = _matrix(torch.sparse_coo, torch.int64, torch.float32)
    builder = ops.prepare_to_sparse_csr(mat)
    allocated = []
    empty = torch.empty

    def fail_third(*args, **kwargs):
        if len(allocated) == 2:
            raise torch.OutOfMemoryError("Injected allocation failure.")

        part = empty(*args, **kwargs)
        allocated.append(weakref.ref(part))
        return part

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", fail_third)
        with pytest.raises(MemoryError, match="AllocationFailure"):
            builder.execute()

    gc.collect()
    assert len(allocated) == 2 and all(ref() is None for ref in allocated)
    with pytest.raises(RuntimeError, match="InvalidState"):
        builder.finalize()

    builder.execute()
    assert torch.equal(builder.finalize().to_dense(), mat.to_dense())
