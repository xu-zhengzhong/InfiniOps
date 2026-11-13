# RT1：稀疏描述符源码实现

状态：**RT1 源码实现已完成**。本任务落实 [D0 的 RT 类型与构造结果](06-d0-contract.md#rt-类型与构造结果)，未完成 RT2 安装导出、RT3 正式测试集或任何 Ops 稀疏计算任务。新增类型目前使用 InfiniRT 源码头文件，不是 `<infini/rt.h>` 的安装接口。

## 文件与边界

实现位于 `/workspace/InfiniRT`：

| 文件 | 职责 |
| --- | --- |
| `src/tensor_layout.h` | 独立 `TensorLayout` 枚举，区分 COO/CSR/CSC/BSR/BSC |
| `src/sparse_tensor_view.h` | 非拥有型 `SparseTensorView`、`ValidationResult`、工厂和 checked 组件访问 |
| `src/sparse_tensor_view.cc` | 元数据安全校验及构造；不读取组件内容、不调用设备运行时 |

`src/CMakeLists.txt` 已有 `*.cc` 源文件收集规则，因此重新配置可编译新增实现。RT1 不修改 dense `TensorView` 的源码、对象字段、hash 或 equality，也不引入 Torch 或稀疏供应商库依赖。

首期只接受显式二维 shape 的 COO/CSR：COO 索引为 int64，CSR 两个索引组件同为 int32 或 int64。所有现有合法 value dtype 均可作为存储描述；这不声明任何 Ops provider 的计算能力。CSC/BSR/BSC、batch/hybrid 仍拒绝，不能自动转布局或稠密回退。

## 具体 C++ 接口

源码消费者 include `"sparse_tensor_view.h"` 并链接同一次构建的 InfiniRT 库。主要入口为：

```cpp
SparseTensorView::CreateCoo(shape, indices, values, out, coalesced);
SparseTensorView::CreateCsr(shape, crow_indices, col_indices, values, out);
SparseTensorView::Create(layout, shape, components, out, coalesced);
```

`shape` 接受 `ShapeView`、`SparseTensorView::Shape`（`std::array<Size, 2>`）或 `TensorView::Size` 的 initializer list。`components` 是组件指针的 initializer list：COO 顺序为 `{&indices, &values}`，CSR 顺序为 `{&crow_indices, &col_indices, &values}`。`out` 为 `std::optional<SparseTensorView>&`；成功才发布完整 view，失败清空 `out`。COO 的 `coalesced` 默认 false，包括空 COO；CSR 不接受 true 的 COO 标记。

返回的 `ValidationResult` 含 `ValidationCode code`、`ValidationComponent component`、可选 `position`，支持 `ok()` 和显式 bool 转换。RT 实现 D0 的 metadata 错误子集，枚举沿用仓库 `k` 前缀；不借用错误字符串，也不以 assert 代替输入错误处理。`position` 在可定位时表示元数据轴位置，成功时为空。

view 查询 `layout()`、`shape()`、`ndim()`、`sparse_dim()`、`dense_dim()`、`batch_dim()`、`nnz()`、`numel()`、`dtype()`、`index_dtype()` 和 `device()`。`shape()` 返回拥有数据的 `std::array<Size, 2>`。view 可复制/移动构造；与 dense view 一样不支持赋值。组件元数据复制后独立保存，组件 data 始终由调用者保活。

组件访问写入 `const TensorView*&` 并返回 `ValidationResult`，失败时指针为 null：

- COO：`indices()`、`values()` 要求声明 coalesced；`raw_indices()`、`raw_values()` 可用于未 coalesced 输入。
- CSR：`crow_indices()`、`col_indices()`、`values()`；`raw_values()` 也可访问 values。
- 错误布局或状态返回 `kInvalidState`；`is_coalesced(bool&)` 仅对 COO 成功。

组件指针借用 sparse 描述符，不能超过其生命周期；临时 sparse 对象上的组件访问在编译期禁用。整体 sparse 对象不提供 dense 的 `data()`、`strides()`、`IsContiguous()`、切片或 `T()`。

## 校验和所有权

工厂独立检查 rank/shape 关系、组件数、index dtype、完整设备一致性、连续性、非空指针、自然对齐、index width、逻辑元素数、组件元素数/bytes 及地址端点。不会直接使用 dense 的未经检查的 `numel()` 或连续性算术。

空逻辑维度要求 N=0；CSR crow 仍须 M+1 项。空组件允许 null 和非负 stride；非空组件的 singleton 轴不限制连续性，负 stride 始终拒绝。逻辑 numel 不乘 value size 来估算 sparse 占用。合法设备无需已启用后端即可被描述。

所有支持的元数据都保存在对象内，工厂和描述符复制不分配主机或设备内存；`kAllocationFailure` 为未来需要元数据分配的表示保留。没有内容扫描、kernel 提交或 stream 同步。假 coalesced 声明、错误索引内容和 CSR crow 内容由后续 Ops checked entry 验证；RT 成功不证明这些条件。

## 验证记录与后续任务

2026-11-13 在 Linux x86-64、GCC 13.3.0、CPU 后端完成本地验证：

| 检查 | 结果 |
| --- | --- |
| CPU Release 构建及现有 CTest（启用 performance tests） | 15/15 通过 |
| CPU Debug 构建及现有 CTest | 11/11 通过 |
| 独立 RT1 Release 合约探针 | 280 项检查通过 |
| 同一探针的 ASan/UBSan 构建 | 280 项检查通过，无 sanitizer 报告 |
| 新源码以 `-fno-exceptions` 编译 | 通过 |
| clang-format 21.1.8、两库 diff 空白、文档代码围栏与相对链接检查 | 通过 |
| dense footprint、allocation、hash/equality 回归 | 通过；当前平台 `sizeof(TensorView)=160`，原 dense 源码未修改 |

构建与现有测试命令：

```sh
cmake -S /workspace/InfiniRT -B /tmp/infinirt-rt1-release \
  -DCMAKE_BUILD_TYPE=Release -DWITH_CPU=ON \
  -DINFINI_RT_BUILD_TESTING=ON -DINFINI_RT_BUILD_PERFORMANCE_TESTING=ON
cmake --build /tmp/infinirt-rt1-release -j 4
ctest --test-dir /tmp/infinirt-rt1-release --output-on-failure
cmake -S /workspace/InfiniRT -B /tmp/infinirt-rt1-debug \
  -DCMAKE_BUILD_TYPE=Debug -DWITH_CPU=ON -DINFINI_RT_BUILD_TESTING=ON
cmake --build /tmp/infinirt-rt1-debug -j 4
ctest --test-dir /tmp/infinirt-rt1-debug --output-on-failure
```

独立探针是本次工作区临时验证文件 `/tmp/rt1-contract-test.cc`，不是已纳入仓库的正式测试。它以 `PROT_NONE` 组件地址验证工厂不读取内容，检查所有 13 种 value dtype、COO/CSR 索引类型、空/单例 stride、元数据/bytes/地址溢出、设备/对齐、复制与临时组件生命周期、零分配、错误组件访问和工厂自替换。编译时直接使用源码：

```sh
c++ -std=c++17 -O2 -DNDEBUG -I/workspace/InfiniRT/src \
  /tmp/rt1-contract-test.cc /workspace/InfiniRT/src/sparse_tensor_view.cc \
  /workspace/InfiniRT/src/tensor_view.cc -o /tmp/rt1-contract-test-release
/tmp/rt1-contract-test-release
c++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I/workspace/InfiniRT/src /tmp/rt1-contract-test.cc \
  /workspace/InfiniRT/src/sparse_tensor_view.cc \
  /workspace/InfiniRT/src/tensor_view.cc -o /tmp/rt1-contract-final-sanitized
/tmp/rt1-contract-final-sanitized
```

现有安装消费者测试验证的是 dense 公共接口；不证明新 sparse 类型已经导出。RT3 仍须将上述元数据用例纳入正式测试集，并在 RT2 完成后验证只 include `<infini/rt.h>` 的 sparse 安装消费者。本次没有执行任何稀疏计算或设备 kernel。

后续依赖保持 [执行计划](04-execution-plan.md#任务拆分与合并顺序) 的 RT2 → RT3 / OP1–OP2 顺序。P1/P2 未完成，Python 桥接、内容校验、动态输出、稀疏算子及设备 provider 均未由 RT1 交付。
