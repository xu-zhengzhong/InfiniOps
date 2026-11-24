"""Exercise generated Python sparse overloads with a synchronous CPU test provider."""

import gc
import importlib.util
import shutil
import sysconfig
import weakref
from pathlib import Path

import pytest
import torch

from tests.test_cpp_api import _build_sparse_codegen_probe, _install_prefixes, _run


@pytest.fixture(scope="module")
def probe_module(tmp_path_factory):
    work = tmp_path_factory.mktemp("sparse_python")
    root = Path(__file__).resolve().parents[1]
    prefixes = _install_prefixes()
    compiler, includes, libraries = _build_sparse_codegen_probe(work, prefixes)
    import pybind11

    # The extension consumes the public Operator declarations. Only libprobe
    # compiles provider selection and the concrete operator instantiations.
    generated = work / "generated"
    shutil.copy(prefixes[0] / "include/operator.h", generated / "include")
    binary = work / ("ops" + sysconfig.get_config_var("EXT_SUFFIX"))
    _run(
        [
            *compiler,
            "-DNDEBUG",
            "-O2",
            "-fPIC",
            "-shared",
            "-fvisibility=hidden",
            "-DWITH_CPU=1",
            f"-I{sysconfig.get_path('include')}",
            f"-I{pybind11.get_include()}",
            f"-I{generated / 'include'}",
            f"-I{work / 'src'}",
            f"-I{root / 'src'}",
            f"-I{work}",
            *includes,
            str(generated / "bindings/ops.cc"),
            str(generated / "bindings/sparse_probe.cc"),
            *libraries,
            "-o",
            str(binary),
        ]
    )
    spec = importlib.util.spec_from_file_location("sparse_probe.ops", binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe(probe_module):
    return probe_module.sparse_probe


def coo(dtype=torch.float32, *, coalesced=False):
    indices = [[1, 1, 0], [0, 0, 2]]
    if coalesced:
        indices = [[0, 0, 1], [1, 2, 0]]
    return torch.sparse_coo_tensor(
        indices,
        torch.tensor([2, -1, 0], dtype=dtype),
        (3, 5),
        is_coalesced=coalesced,
        check_invariants=True,
    )


def csr(index_dtype=torch.int64, dtype=torch.float32):
    return torch.sparse_csr_tensor(
        torch.tensor([0, 2, 2, 3], dtype=index_dtype),
        torch.tensor([0, 2, 1], dtype=index_dtype),
        torch.tensor([2, -1, 0], dtype=dtype),
        size=(3, 5),
        check_invariants=True,
    )


@pytest.mark.parametrize(
    "make,state",
    [
        (coo, 0),
        (lambda: coo(torch.float64, coalesced=True), 1),
        (lambda: csr(torch.int32), -1),
        (lambda: csr(torch.int64, torch.float64), -1),
        (
            lambda: torch.sparse_coo_tensor(
                torch.empty(2, 0, dtype=torch.int64),
                torch.empty(0),
                (0, 5),
                is_coalesced=False,
            ),
            0,
        ),
        (lambda: torch.zeros(3, 0).to_sparse_csr(), -1),
    ],
)
def test_structure_and_component_owners(probe, make, state):
    tensor = make()
    components = (
        ["_indices", "_values"]
        if tensor.layout == torch.sparse_coo
        else ["crow_indices", "col_indices", "values"]
    )
    addresses = [getattr(tensor, name)().data_ptr() for name in components]
    out = torch.full((4,), -9, dtype=tensor.dtype)
    expected = [*tensor.shape, tensor._nnz(), state]
    refs = []
    released = {}

    # Accessors produce fresh Python wrappers around the original storage.
    # A weakref callback observes whether each wrapper outlives the provider write.
    class WatchedTensor:
        def __getattr__(self, name):
            if name in ("data_ptr", "stride", "coalesce", "indices", "to_dense"):
                raise AssertionError(f"unexpected sparse accessor {name}")
            if name not in components:
                return getattr(tensor, name)

            def component():
                gc.collect()
                value = getattr(tensor, name)()
                refs.append(
                    weakref.ref(value, lambda _: released.update({name: out.tolist()}))
                )
                return value

            return component

    probe(WatchedTensor(), addresses, out)
    assert out.tolist() == expected
    assert released == dict.fromkeys(components, expected)
    assert all(ref() is None for ref in refs)


def test_overloads_and_containers(probe):
    dense = torch.ones(1)
    sparse = coo()
    probe(input=dense, scale=0)
    probe(input=sparse, scale=2, implementation_index=0)
    with pytest.raises(ValueError, match="InvalidMetadata"):
        probe(sparse, 0)
    with pytest.raises(ValueError, match="UnsupportedImplementation"):
        probe(sparse, 2, implementation_index=27)
    probe(dense, [None, sparse])
    probe(dense, (None, sparse))
    probe(dense, [dense])
    probe(dense, (item for item in (dense,)))
    probe(dense, [])
    with pytest.raises(TypeError):
        probe(dense, (item for item in (None, sparse)))
    with pytest.raises(TypeError):
        probe(dense, [dense, sparse])
    with pytest.raises(ValueError, match="InvalidStructure"):
        probe(dense, [sparse])
    probe([sparse], dense)
    with pytest.raises(ValueError, match="InvalidMetadata"):
        probe(None, dense)
    with pytest.raises(ValueError, match="InvalidStructure"):
        probe([], dense)


def test_dense_instance_rejects_sparse_before_dense_access(probe_module):
    dense = torch.ones(1)
    op = probe_module.SparseProbe(dense, 0)
    with pytest.raises(NotImplementedError, match="UnsupportedLayout"):
        op(coo(), 0)


def test_revalidates_borrowed_indices(probe):
    tensor = coo()
    probe(tensor, 2)
    tensor._indices()[1, 2] = 5
    with pytest.raises(IndexError, match=r"IndexOutOfBounds.*argument 0.*indices\[5\]"):
        probe(tensor, 2)
    tensor._indices()[1, 2] = 2
    probe(tensor, 2)


def test_false_coalesced_claim(probe):
    tensor = coo()
    bad = torch.sparse_coo_tensor(
        tensor._indices(),
        tensor._values(),
        tensor.shape,
        is_coalesced=True,
        check_invariants=False,
    )
    with pytest.raises(ValueError, match="InvalidCoalescedClaim"):
        probe(bad, 2)


@pytest.mark.parametrize(
    "crow,col,code",
    [
        ([1, 2, 2], [0, 1], "InvalidStructure"),
        ([0, 3, 2], [0, 1], "InvalidStructure"),
        ([0, 1, 1], [0, 1], "InvalidStructure"),
        ([0, 2, 2], [1, 1], "InvalidStructure"),
        ([0, 2, 2], [1, 0], "InvalidStructure"),
        ([0, 2, 2], [-1, 1], "IndexOutOfBounds"),
    ],
)
def test_checked_csr_inputs_and_out_only(probe, crow, col, code):
    bad = torch.sparse_csr_tensor(
        crow, col, torch.ones(2), size=(2, 3), check_invariants=False
    )
    with pytest.raises(
        IndexError if code == "IndexOutOfBounds" else ValueError, match=code
    ):
        probe(bad, 2)
    # The out-only overload has no checked provider. Its index storage is writable
    # capacity and its initial contents must never be validated as input.
    with pytest.raises(NotImplementedError, match="NoProvider"):
        probe(csr(), bad)


def test_forward_only_checks_every_operand(probe):
    sparse = coo().requires_grad_()
    dense = torch.ones(1)
    with pytest.raises(NotImplementedError, match=r"AutogradUnsupported.*detach"):
        probe(dense, [None, sparse])
    with pytest.raises(NotImplementedError, match=r"AutogradUnsupported.*detach"):
        probe(dense.requires_grad_(), [None, sparse.detach()])
    probe(dense.detach(), [None, sparse.detach()])


def test_rt_metadata_errors(probe):
    def make(crow, col, values, size):
        return torch.sparse_csr_tensor(
            crow, col, values, size=size, check_invariants=False
        )

    bad_width = make(
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0]),
        torch.ones(1),
        (1, 2),
    )
    with pytest.raises(TypeError, match="InvalidIndexType.*col_indices"):
        probe(bad_width, 2)
    large = make(
        torch.zeros(2, dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.empty(0),
        (1, 2**31),
    )
    with pytest.raises(OverflowError, match="IntegerOverflow.*shape"):
        probe(large, 2)
    strided = make([0, 2], [0, 1], torch.arange(4.0)[::2], (1, 2))
    with pytest.raises(NotImplementedError, match="UnsupportedCapability.*values"):
        probe(strided, 2)
    with pytest.raises(NotImplementedError, match="UnsupportedCapability"):
        probe(torch._neg_view(torch.ones(2)), [None, coo()])


@pytest.mark.parametrize(
    "make,code",
    [
        (lambda: torch.eye(2).to_sparse_csc(), "UnsupportedLayout"),
        (lambda: torch.eye(2).to_sparse_bsr((1, 1)), "UnsupportedLayout"),
        (lambda: torch.eye(2).to_sparse_bsc((1, 1)), "UnsupportedLayout"),
        (lambda: torch.ones(2, 3).to_sparse(1), "UnsupportedCapability"),
        (lambda: torch.ones(2, 3, 3).to_sparse_csr(), "UnsupportedCapability"),
        (lambda: coo(torch.complex64), "UnsupportedCapability"),
        (
            lambda: torch.sparse_coo_tensor(size=(2, 2), device="meta"),
            "UnsupportedCapability",
        ),
    ],
)
def test_unsupported_inputs(probe, make, code):
    with pytest.raises(NotImplementedError, match=code):
        probe(make(), 2)
