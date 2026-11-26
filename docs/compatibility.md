# Compatibility

This page defines the current documentation and compatibility boundary for
InfiniOps users and contributors.

## Public API Entries

The formal public entries are:

```cpp
#include <infini/ops.h>
```

```python
import infini.ops
```

InfiniOps exposes C++ as its formal native API and generated Python bindings as
its formal Python API. A separate C API is not currently exposed or planned.

The supported C++ compatibility boundary includes:

- the installed `<infini/ops.h>` entry point
- the core types documented under `docs/api/`
- the documented operator classes and `Call(...)` forms

The C++ boundary is intentionally narrower than the complete installed header
tree. A header being installed or included transitively does not by itself make
every symbol in that header part of the public API.

## InfiniRT Dependency

InfiniOps depends on InfiniRT headers and libraries from the configured
`INFINI_RT_ROOT`. Consumers should treat the installed InfiniOps wheel,
InfiniOps headers, and bundled or linked InfiniRT library as a matching set
from the same build.

## Generated Files

The build may generate files under `generated/`, including:

- public operator call instantiations
- generated base classes
- generated PyTorch C++ backend wrappers
- generated Python bindings
- generated metadata for tests

Generated files are build artifacts. Do not edit them by hand. Change the
source generator, allowlist, or source operator definitions instead.

## Implementation Details

Backend implementations under `src/native/**` and `src/torch/**`, generated
implementation sources, generated binding sources, and backend-specific runtime
adapters are implementation-facing. They are useful for contributors, tests,
and repository examples, but downstream users should not include them directly.

Support headers pulled in transitively by `<infini/ops.h>` compile the public C++
API. Undocumented helpers and templates in those headers remain implementation
details unless an API page explicitly includes them in the supported boundary.

## Backend-Specific Behavior

Operator availability and supported dtypes, layouts, strides, and implementation
indexes can differ by backend. Tests should document backend-specific skips or
tolerances explicitly instead of hiding them in broad generated results.

## Public API Compatibility

Follow these rules when changing public C++ and Python surfaces:

- Keep operator signatures stable unless the PR clearly documents the migration
  path.
- Preserve generated Python binding names when changing implementation details.
- Update documentation and tests in the same PR when user-visible behavior
  changes.
- Treat shared dispatch, wrapper generation, and backend selection changes as
  high-risk and validate them across affected platforms.

## Sparse D0 Boundary

The native CPU build exports `ToDense::CheckedCall` and `SparseMm::CheckedCall`
through `<infini/ops.h>`. Their Python entry points are `infini.ops.to_dense`
and `infini.ops.sparse_mm`. They use the installed RT sparse view and support
two-dimensional COO/CSR with contiguous float32 or float64 values. SparseMm
takes a dense right operand and an explicit `"sum"` reduction. OP5 adds
caller-owned dynamic COO/CSR output, COO coalescing, and exact-size CSR-to-COO
conversion. OP6 adds `SparseSampledAddmm::CheckedCall` and
`infini.ops.sparse_sampled_addmm(input, mat1, mat2, beta, alpha, out)`.
Both write the input's CSR pattern and index dtype into caller-owned storage.
The former dense overload reports how to migrate to CSR. With `WITH_CPU=ON`
and `WITH_TORCH=ON`, all six sparse operators also support CPU implementation 8.
ATen borrows input components for the synchronous call and copies results into
the validated caller storage. [AT1 verification](sparse/16-at1-implementation.md)
See the [D0 contract](sparse/06-d0-contract.md),
[P1 verification](sparse/13-op4-implementation.md), [OP5 interfaces](sparse/14-op5-implementation.md),
[OP6 verification](sparse/15-op6-implementation.md) and
[operator API alignment](operator-api-alignment.md#sparse-api-contract).

`Tensor` remains the dense `TensorView` alias. Sparse support must not alter its
object layout, hash, equality, or existing dense call signatures. Generators,
optional/list arguments, dispatch, caches, and Python conversion must preserve
the explicit dense/sparse type distinction. Support is per operation, layout,
dtype, index width, and complete device; a layout enum or installed type does
not establish compute support. Unsupported requests must not silently densify,
copy to another device, narrow indices, or select a different explicitly
requested implementation.

The sparse checked boundary returns status for external failures in Debug and
Release; assertions are only for internal invariants on this path. This scoped
decision does not change existing dense error behavior. ATen exceptions are
contained and translated at the provider boundary. Python raises the mapped
exception instead of publishing a failed or pending result. Initial support is
forward-only, without implicit detach or online tuning; capture is disabled
until a provider passes separate validation.

All output buffers remain caller-owned. Capacity does not become `nnz` until
successful completion and Finalize. Aliasing between any write range and
inputs or other write ranges is initially prohibited. Component owners,
workspace, and result slots live through all consuming streams; Python bridges
and ATen wrappers must honor the same rule. Structure mutation invalidates
validation and dependent analysis, even at the same address.

Cross-repository integration must pin a tested RT SHA or installation package
and record it with the Ops revision. Preserve the matching headers/library/
wheel rule above; unchanged dense representation does not promise arbitrary
cross-version binary compatibility. RT and Ops installed consumers, Release
error cases, generated signatures, dense regressions, and the dedicated sparse
matrix must pass before marking each capability supported. The CPU native
path must build without a Torch provider; optional ATen remains in slot 8.
