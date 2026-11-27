"""GPU1 installation and numerical tests for the native NVIDIA sparse provider."""

from pathlib import Path

import pytest
import torch

import infini.ops as ops
from tests.test_cpp_api import _install_prefixes, _run

OPERATORS = (
    "ToDense",
    "SparseMm",
    "ToSparseCoo",
    "SparseSampledAddmm",
    "Coalesce",
    "ToSparseCsr",
)


@pytest.fixture(scope="module", autouse=True)
def nvidia_install(request):
    selected = request.config.getoption("--devices")
    if selected and not {"nvidia", "cuda"}.intersection(selected):
        pytest.skip("NVIDIA was not selected")
    if not selected and not any(
        hasattr(ops, name)
        and getattr(ops, name).active_implementation_indices("nvidia")
        for name in OPERATORS
    ):
        pytest.skip("no NVIDIA sparse provider in this installation")


@pytest.fixture
def cuda_stream():
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime has no accessible NVIDIA device")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        yield stream


def test_installed_registration():
    for name in OPERATORS:
        assert getattr(ops, name).active_implementation_indices("nvidia") == [0]


@pytest.fixture(scope="module")
def cpp_consumer(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("sparse_nvidia_cpp")
    source = Path(__file__).parent / "cpp/sparse_nvidia.cc"
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.18)\nproject(sparse_consumer LANGUAGES CXX)\n"
        "find_package(InfiniOps CONFIG REQUIRED)\n"
        f'add_executable(consumer "{source.resolve()}")\n'
        "target_compile_features(consumer PRIVATE cxx_std_17)\n"
        "target_link_libraries(consumer PRIVATE InfiniOps::infiniops)\n"
    )
    build = tmp_path / "build"
    _run(
        [
            "cmake",
            "-S",
            str(tmp_path),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_PREFIX_PATH=" + ";".join(map(str, _install_prefixes())),
        ]
    )
    _run(["cmake", "--build", str(build)])
    return str(build / "consumer")


def test_cpp_metadata_boundaries(cpp_consumer):
    _run([cpp_consumer])


def test_cpp_nondefault_stream_and_capture(cuda_stream, cpp_consumer):
    _run([cpp_consumer, "cuda"])


def coo(coords, values, shape, dtype=torch.float32, *, coalesced=False):
    return torch.sparse_coo_tensor(
        torch.tensor(coords, dtype=torch.int64, device="cuda"),
        torch.tensor(values, dtype=dtype, device="cuda"),
        size=shape,
        is_coalesced=coalesced,
        check_invariants=False,
    )


def csr(rows, cols, values, shape, index=torch.int64, dtype=torch.float32):
    return torch.sparse_csr_tensor(
        torch.tensor(rows, dtype=index, device="cuda"),
        torch.tensor(cols, dtype=index, device="cuda"),
        torch.tensor(values, dtype=dtype, device="cuda"),
        size=shape,
        check_invariants=False,
    )


def check(actual, expected):
    torch.testing.assert_close(
        actual.cpu(),
        torch.as_tensor(expected, dtype=actual.dtype),
        rtol=0,
        atol=0,
        equal_nan=True,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "index", [None, torch.int32, torch.int64], ids=["coo", "csr32", "csr64"]
)
@pytest.mark.parametrize("width", [0, 1, 7])
def test_dense_and_mm(cuda_stream, dtype, index, width):
    if index is None:
        x = coo([[2, 0, 0, 0], [1, 0, 2, 0]], [-1, 5, 0, -2], (4, 3), dtype)
    else:
        x = csr([0, 2, 2, 3, 3], [0, 2, 1], [3, 0, -1], (4, 3), index, dtype)
    expected = torch.tensor([[3, 0, 0], [0, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=dtype)
    out = torch.full(x.shape, -71, device="cuda", dtype=dtype)
    rhs = torch.arange(3 * width, device="cuda", dtype=dtype).view(3, width)
    product = torch.full((4, width), -71, device="cuda", dtype=dtype)
    ops.to_dense(x, out)
    ops.sparse_mm(
        x, rhs, "sum", product, stream=cuda_stream.cuda_stream, implementation_index=0
    )
    check(out, expected)
    check(product, expected @ rhs.cpu())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("already_sorted", [False, True])
def test_dynamic_outputs(cuda_stream, dtype, already_sorted):
    x = coo([[2, 0, 2, 0], [0, 1, 0, 0]], [3, 2, -3, 5], (4, 3), dtype)
    if already_sorted:
        x = coo([[0, 0, 2], [0, 1, 0]], [5, 2, 0], (4, 3), dtype, coalesced=True)
    builder = ops.prepare_coalesce(x)
    capacity = x._nnz()
    indices = torch.full((2 * capacity + 1,), -77, device="cuda", dtype=torch.int64)
    values = torch.full((capacity + 1,), -77, device="cuda", dtype=dtype)
    required, _ = builder.requirements["workspace"]
    assert required > 0
    ws = torch.empty(required, device="cuda", dtype=torch.uint8)
    with pytest.raises(ValueError, match="InvalidWorkspace"):
        builder.execute(indices=indices[:-1], values=values[:-1], workspace=ws[:-1])
    check(indices, [-77] * indices.numel())
    builder.execute(indices=indices[:-1], values=values[:-1], workspace=ws)
    result = builder.finalize()
    assert result.is_coalesced() and result.indices().data_ptr() == indices.data_ptr()
    assert result.values().data_ptr() == values.data_ptr()
    check(indices, [0, 0, 2, 0, 1, 0] + [-77] * (indices.numel() - 6))
    check(values, [5, 2, 0] + [-77] * (values.numel() - 3))
    compressed = ops.to_sparse_csr(x)
    check(compressed.crow_indices(), [0, 2, 2, 3, 3])
    check(compressed.col_indices(), [0, 1, 0])
    check(compressed.values(), [5, 2, 0])


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 0), (3, 4)])
def test_empty(cuda_stream, shape):
    x = coo([[], []], [], shape)
    assert ops.coalesce(x)._nnz() == 0
    y = ops.to_sparse_csr(x)
    check(y.crow_indices(), [0] * (shape[0] + 1))
    for sparse in (x, y):
        dense = torch.full(shape, -71.0, device="cuda")
        ops.to_dense(sparse, dense)
        check(dense, torch.zeros(shape))
        rhs = torch.empty((shape[1], 2), device="cuda")
        out = torch.full((shape[0], 2), -71.0, device="cuda")
        ops.sparse_mm(sparse, rhs, "sum", out)
        check(out, torch.zeros(shape[0], 2))


@pytest.mark.parametrize("index", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sample_and_csr_expansion(cuda_stream, index, dtype):
    x = csr([0, 2, 2], [0, 1], [0, 2], (2, 2), index, dtype)
    out = csr([-1, -1, -1], [-1, -1], [-1, -1], (2, 2), index, dtype)
    lhs = torch.tensor([[2, 3], [0, 0]], dtype=dtype, device="cuda")
    rhs = torch.tensor([[4, 5], [6, 7]], dtype=dtype, device="cuda")
    ops.sparse_sampled_addmm(x, lhs, rhs, 1, 1, out)
    check(out.values(), [26, 33])
    check(out.crow_indices(), [0, 2, 2])
    check(out.col_indices(), [0, 1])
    rebuilt = coo([[-1, -1], [-1, -1]], [-1, -1], (2, 2), dtype, coalesced=True)
    ops.to_sparse_coo(out, rebuilt)
    check(rebuilt.indices(), [[0, 0], [0, 1]])
    check(rebuilt.values(), [26, 33])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "value,lhs,alpha,beta,expected",
    [
        (float("nan"), [1], 1, 0, float("nan")),
        (float("inf"), [1], 1, 0, float("nan")),
        (2, [float("inf")], 0, 1, float("nan")),
        (2, [3], -2, -1, -8),
        (2, [], float("nan"), 3, 6),
    ],
)
def test_sampled_special_values(cuda_stream, dtype, value, lhs, alpha, beta, expected):
    x = csr([0, 1], [0], [value], (1, 1), dtype=dtype)
    out = csr([-7, -7], [-7], [-7], (1, 1), dtype=dtype)
    left = torch.tensor(lhs, device="cuda", dtype=dtype).view(1, len(lhs))
    right = torch.ones((len(lhs), 1), device="cuda", dtype=dtype)
    ops.sparse_sampled_addmm(x, left, right, beta, alpha, out)
    check(out.values(), [expected])


@pytest.mark.parametrize(
    "make,exc,code",
    [
        (
            lambda: csr([0, 3, 1], [0], [1], (2, 2)),
            ValueError,
            "InvalidStructure.*crow_indices",
        ),
        (
            lambda: csr([0, 2, 2], [1, 1], [1, 2], (2, 2)),
            ValueError,
            "InvalidStructure.*col_indices",
        ),
        (
            lambda: csr([0, 1, 1], [-1], [1], (2, 2)),
            IndexError,
            "IndexOutOfBounds.*col_indices",
        ),
        (lambda: coo([[0], [2]], [1], (2, 2)), IndexError, "IndexOutOfBounds.*indices"),
        (
            lambda: coo([[1, 0], [0, 0]], [1, 1], (2, 2), coalesced=True),
            ValueError,
            "InvalidCoalescedClaim.*indices",
        ),
    ],
)
def test_reject_before_output_write(cuda_stream, make, exc, code):
    out = torch.full((2, 2), -71.0, device="cuda")
    with pytest.raises(exc, match=code):
        ops.to_dense(make(), out)
    check(out, torch.full((2, 2), -71.0))


def test_recheck_and_stream(cuda_stream):
    x = coo([[0, 0], [1, 1]], [2, 3], (1, 2))
    builder = ops.prepare_coalesce(x)
    with (
        torch.cuda.stream(torch.cuda.Stream()),
        pytest.raises(NotImplementedError, match="stream"),
    ):
        builder.execute()
    builder.execute()
    check(builder.finalize().values(), [5])
    out = torch.empty((1, 2), device="cuda")
    ops.to_dense(x, out)
    x._indices()[0, 0] = -1
    with pytest.raises(IndexError, match="IndexOutOfBounds"):
        ops.to_dense(x, out)
    check(out, [[0, 5]])


def test_sort_multiple_blocks_and_wide_coordinates(cuda_stream):
    width = 2**32 + 9
    n = 4097
    pos = torch.arange(n)
    coords = torch.stack([(pos * 17) % 11, 2**32 + (pos * 3) % 7])
    values = (pos % 5 - 2).float()
    reference = torch.sparse_coo_tensor(coords, values, (13, width)).coalesce()
    x = torch.sparse_coo_tensor(coords.cuda(), values.cuda(), (13, width))
    actual = ops.coalesce(x)
    check(actual.indices(), reference.indices())
    check(actual.values(), reference.values())
    compressed = ops.to_sparse_csr(x)
    check(compressed.col_indices(), reference.indices()[1])
    check(compressed.values(), reference.values())


def test_failed_execute_does_not_publish(cuda_stream):
    # Inference tensors have no version counter. Execute must inspect the indices again.
    with torch.inference_mode():
        x = coo([[0, 0], [0, 1]], [1, 2], (1, 2))
        builder = ops.prepare_to_sparse_csr(x)
        x._indices()[1, 0] = -1
        builder.execute()
        with pytest.raises(IndexError, match="IndexOutOfBounds"):
            builder.finalize()
        x._indices()[1, 0] = 0
        builder.execute()
        check(builder.finalize().values(), [1, 2])


def test_stored_float_subnormal(cuda_stream):
    tiny = torch.finfo(torch.float32).tiny / 2
    x = coo([[0, 0], [0, 0]], [tiny, 0], (1, 1))
    out = torch.empty((1, 1), device="cuda", dtype=torch.float32)
    ops.to_dense(x, out)
    check(out, [[tiny]])
    ops.sparse_mm(x, torch.ones_like(out), "sum", out)
    check(out, [[tiny]])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sampled_overflow_before_add(cuda_stream, dtype):
    x = csr([0, 1], [0], [float("-inf")], (1, 1), dtype=dtype)
    out = csr([-1, -1], [-1], [-1], (1, 1), dtype=dtype)
    lhs = torch.full((1, 1), 2, dtype=dtype, device="cuda")
    ops.sparse_sampled_addmm(
        x, lhs, torch.ones_like(lhs), 1, torch.finfo(dtype).max, out
    )
    check(out.values(), [float("nan")])
