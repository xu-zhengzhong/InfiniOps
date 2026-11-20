"""OP1 source-header regression; only an installed CPU InfiniRT is needed.

Run with INFINI_RT_ROOT pointing to the RT2/RT3 installation. The executable
uses instrumented providers and a tuning-manager test double, not sparse kernels.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("build_mode", ["debug", "release"])
def test_sparse_operator_contract(tmp_path, build_mode):
    prefix = os.environ.get("INFINI_RT_ROOT")
    if not prefix:
        pytest.skip("INFINI_RT_ROOT must name an installed CPU InfiniRT")
    prefix = Path(prefix)
    library_dir = next(
        (
            directory
            for directory in (prefix / "lib", prefix / "lib64", prefix)
            if (directory / "libinfinirt.so").exists()
        ),
        None,
    )
    assert library_dir is not None, f"libinfinirt.so missing under {prefix}"
    root = Path(__file__).resolve().parents[1]
    binary = tmp_path / "sparse_operator"
    command = [
        os.environ.get("CXX", "c++"),
        "-std=c++17",
        "-Werror",
        *(["-O2", "-DNDEBUG"] if build_mode == "release" else ["-O0", "-g"]),
        f"-I{root / 'src'}",
        f"-I{prefix / 'include'}",
        str(root / "tests/cpp/sparse_operator.cc"),
        f"-L{library_dir}",
        "-linfinirt",
        f"-Wl,-rpath,{library_dir}",
        "-o",
        str(binary),
    ]
    _run(command)
    result = _run([str(binary)])
    assert result.stdout.strip() == "OP1 sparse contract passed"
    environment = os.environ.copy()
    environment["INFINI_OPS_TRACE_CALLS"] = "1"
    environment["INFINI_OPS_TRACE_CALLS_DETAILED"] = "1"
    traced = _run([str(binary)], env=environment)
    basic_prefix = "[INFINI_OPS_TRACE_CALLS] "
    detail_prefix = "[INFINI_OPS_TRACE_CALLS_DETAILED] "
    basic = [
        json.loads(line.removeprefix(basic_prefix))
        for line in traced.stderr.splitlines()
        if line.startswith(basic_prefix)
    ]
    detailed = [
        json.loads(line.removeprefix(detail_prefix))
        for line in traced.stderr.splitlines()
        if line.startswith(detail_prefix)
    ]
    assert basic and all(event["device_type"] == "cpu" for event in basic)
    sparse = [tensor for event in detailed for tensor in event.get("sparse_tensors", [])]
    assert {tensor["layout"] for tensor in sparse} == {"coo", "csr"}
    assert all(tensor["device"] == "cpu:0" for tensor in sparse)
    assert all(tensor["index_dtype"] == "int64" for tensor in sparse)
    assert any(tensor["nnz"] == 0 for tensor in sparse)
    assert any(tensor.get("coalesced") is False for tensor in sparse)


def _run(command, **kwargs):
    result = subprocess.run(command, text=True, capture_output=True, **kwargs)
    assert result.returncode == 0, (
        f"command failed ({result.returncode}): {' '.join(map(str, command))}\n"
        f"{result.stdout}\n{result.stderr}"
    )
    return result
