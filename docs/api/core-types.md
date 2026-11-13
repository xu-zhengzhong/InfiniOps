# Core Types

This page summarizes the core types used by InfiniOps operators. InfiniOps
reuses the public InfiniRT runtime types where possible.

## Tensor

`infini::ops::Tensor` is an alias for `infini::rt::TensorView`.

```cpp
#include "tensor.h"

std::vector<float> values(16);
infini::ops::Tensor tensor{
    values.data(),
    std::vector<std::size_t>{4, 4},
    infini::ops::Device{infini::ops::Device::Type::kCpu, 0}};
```

`Tensor` is a non-owning view. It stores the data pointer, shape, data type,
device, and strides, but it does not own the memory it references.

## SparseTensorView: D0 Contract

**Design frozen; Ops API not implemented or exported yet.** RT1 implements
the independent `infini::rt::SparseTensorView`, and
[RT2](../sparse/08-rt2-implementation.md) exports it through the installed
`<infini/rt.h>`. RT3's full metadata test suite and the dependent Ops integration
remain pending. `Tensor` remains the dense alias.
The complete [D0 contract](../sparse/06-d0-contract.md) defines validation,
checked execution, errors, dynamic results, and ownership. The existing dense
`TensorView` layout, hash, and equality must remain unchanged.

The sparse view owns logical shape and component metadata and borrows every
component's data. It has no aggregate `data()` or `strides()`. Initial
representation is two-dimensional, non-batch, non-hybrid COO/CSR with contiguous
components and an explicit shape `[M, K]`:

| Layout | Components | Index dtype |
| --- | --- | --- |
| COO | `indices=[2,N]`, `values=[N]` | int64 |
| CSR | `crow_indices=[M+1]`, `col_indices=[N]`, `values=[N]` | Matching int32 or int64 |

`nnz=N` counts stored entries, including explicit zeros and COO duplicates;
logical `numel`, component bytes, and output capacity are different quantities.
All components share the full device `(type, index)`. RT factories validate
metadata and overflow without reading contents or submitting device work.
Empty components may have null data; an empty CSR still needs `M+1` row
pointers. Each logical extent and N must fit the signed index type, and all
element, byte, and address-range arithmetic must be checked.
Nonempty typed components must be naturally aligned before any content read;
output and result-slot alignment also obeys the requirements reported by
Prepare. Misaligned components or result slots return `InvalidMetadata`;
workspace alignment errors return `InvalidWorkspace`.

COO `indices()/values()` require the declared coalesced state; raw access
supports importing uncoalesced data without changing it. The declaration
defaults to false, including for empty COO. A declaration is not
proof of content validity. Ops checked entry validates bounds and any claimed
ordering, and validates CSR pointers and strictly increasing columns per row.
RT construction success alone does not make a view executable. Unsupported
layouts, dtypes, devices, or reductions fail explicitly. Initial computation
requires CPU float32/float64 with matching value dtypes and contiguous dense
operands/outputs; negative strides are rejected. Python requires explicit
detach for inputs needing gradients.

`ValidationResult` carries a code, component, and optional error position;
external sparse errors remain checked in Release and are translated to Python
exceptions at the boundary. Sparse checked entry is separate from today's
void `Call`. Trusted input may skip repeated content scans only while a
validated structure remains unchanged; metadata and resource checks still run.

Fixed-shape and fixed-pattern outputs borrow writable caller storage. A CSR
out-only descriptor may have uninitialized index contents: only output
metadata/capacity is checked before execution writes the input pattern.
Dynamic `SparseOutputBuffer` capacity and `SparseResultMetadata` are separate
from a valid sparse input. Prepare reports capacity/workspace, Execute writes
data and completion metadata, and Finalize publishes a view only after
successful completion with the actual N. COO indices must be packed as
contiguous `[2,N]`; shrinking a `[2,capacity]` shape is insufficient.

All output, workspace, and result-slot write ranges are mutually disjoint and
do not overlap inputs. Every component owner and execution resource survives
until all consumers complete; copying a view does not retain those owners.
Python builders must preserve the owner chain in returned tensors. Structural
mutation invalidates validation and structure plans. Metadata caches cannot
stand in for structure analysis, and sparse online tuning is initially off.

## Device

`infini::ops::Device` is an alias for `infini::rt::Device`.

Known device types include:

- `Device::Type::kCpu`
- `Device::Type::kNvidia`
- `Device::Type::kIluvatar`
- `Device::Type::kMetax`
- `Device::Type::kMoore`
- `Device::Type::kHygon`
- `Device::Type::kCambricon`
- `Device::Type::kAscend`

Some source code also carries experimental or future device enum values. Treat
documented build options in [Backends](../backends.md) as the supported user
surface.

## DataType

`infini::ops::DataType` is imported from InfiniRT. Common values include:

- `DataType::kInt8`, `DataType::kInt16`, `DataType::kInt32`, `DataType::kInt64`
- `DataType::kUInt8`, `DataType::kUInt16`, `DataType::kUInt32`, `DataType::kUInt64`
- `DataType::kFloat16`, `DataType::kBFloat16`, `DataType::kFloat32`, `DataType::kFloat64`
- `DataType::kBool` (distinct from `kUInt8`, with one byte per element)

InfiniOps also exposes type-list helpers such as `FloatTypes`,
`ReducedFloatTypes`, `IntTypes`, `UIntTypes`, and `AllTypes` for template
dispatch.

## Handle

`Handle` carries optional runtime resources for a call:

- backend stream
- workspace pointer
- workspace size

Operators that need temporary memory read the workspace from the handle.
Callers that do not need custom stream or workspace handling can use the
shorter `Op::Call(...)` form without an explicit handle.

## Config

`Config` carries operator configuration that is independent of tensor geometry.
The most common field is `implementation_index`, which selects one of the
active implementations for an operator and device.

```cpp
infini::ops::Config config;
config.set_implementation_index(1);
```

Use the default config unless a specific backend implementation needs to be
selected deliberately.
