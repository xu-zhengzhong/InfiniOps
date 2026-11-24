# OP3 Python 稀疏输入桥接

生成的 `infini.ops` 模块函数按每个张量的 `layout` 选择 C++ 重载。稀疏签名调用 `generated_dispatch::CheckedCall…`，沿用 OP1 的自动或显式 provider 选择。当前交付覆盖同步 CPU 输入桥接。ToDense 和 SparseMm 的计算 provider 属于 OP4。

## 输入与所有权

| 输入 | 导入行为 |
| --- | --- |
| 二维 COO | 读取逻辑 shape、`is_coalesced()`、`_indices()` 和 `_values()`。借用原始 int64 坐标与 float32/64 values，保留乱序、重复项、显式零和空 COO 的标记。 |
| 二维 CSR | 借用 `crow_indices()`、`col_indices()` 和 `values()`。两种索引组件同为 int32 或同为 int64。 |
| 混合参数 | 同名函数使用区分 dense 与 sparse 的 pybind11 类型检查。optional 和 list 按声明逐层匹配。空列表或 `None` 缺少布局信息，同等匹配时保留既有 dense 重载。稀疏列表使用可重复遍历的序列，如 list 或 tuple。 |
| 固定 sparse 输出 | 按约定命名为 `out` 的参数导入可写组件元数据。索引内容可由执行阶段初始化。 |

`src/python/sparse_import.h` 持有所有索引和值组件的 Python 强引用，覆盖整个同步调用和异常退出。CPU 路径借用 PyTorch 存储，其释放由 PyTorch owner 管理。设备异步执行所需的 stream 登记和延迟释放归 GPU2。

## Checked import

导入先检查设备、维度、dtype 和梯度要求，再通过 RT 工厂检查组件几何、连续性、对齐及地址范围。每次调用都会扫描输入索引。COO 检查两个坐标的边界，声明 coalesced 时还检查字典序严格递增。CSR 先验证完整 crow 的首项、末项和行区间，再检查每行 col 的范围与严格递增顺序。CPU 内容校验位于 `src/native/cpu/sparse_validation.h`，可供 OP4 的 C++ checked provider 复用。

绑定按 D0 翻译导入及 provider 的失败状态，保留可用的参数序号、组件和位置。越界索引触发 `IndexError`，假 coalesced 声明触发 `ValueError`。稀疏调用中任一张量需要梯度时，`NotImplementedError` 提示调用者显式 `detach()`。未支持布局和设备在读取其存储前返回错误。

## 验证

工作基线为 InfiniOps `abe5575`，配套 InfiniRT 为 `6ffd4e671fce8c591b3b188e30b6dcbb8cd4892e`。原生 CPU Release 构建、安装和测试使用 Python 3.12、pybind11 3.1.0 及 PyTorch `2.11.0+cu128` 的 CPU 张量，关闭 Torch provider。

| 检查 | 结果 |
| --- | --- |
| Python 生成扩展 | 25 项通过。测试 provider 比对实际借用地址、shape、nnz 和 coalesced 标记。弱引用回调和强制 GC 验证每个组件在 provider 写回之后释放。覆盖嵌套参数、非法结构、原地改写后重验、错误映射、空维度与 out-only 索引。 |
| 生成与安装 | Python 专项、三个生成器模块和安装 C++ 消费者合计 106 项通过。569 个含构造函数和调用声明的手写 dense 算子，其生成的 Python 绑定与 OP2 基线逐字一致。消费者在 `-O0`、`-O2` 下通过。独立源码副本仅选择 `sparse_probe`，通过常规 CMake 构建和安装后，从安装的 `infini.ops` 成功调用 dense、COO、CSR 和嵌套参数重载。 |
| 内存边界 | 新增 CPU 索引校验通过 ASan/UBSan 边界探针，覆盖 int32/64 极值、非法行指针、重复坐标和空组件。 |
| 完整 CPU 套件 | 2444 passed、21 failed、7377 skipped。21 个失败用例的 ID 与 OP2 完整回归相同，在 OP2 安装上逐例复现。涉及 Embedding 15 项、CUTLASS 绑定 2 项、CUDA 编译探针 2 项、Ascend workspace 编译 1 项、configless C++ 消费者 1 项。 |

复现专项时，将 `PYTHONPATH` 指向包含本次安装的 `infini` 包的目录，并提供 Ops 和 RT 的安装前缀。测试使用生成器在临时目录构建测试 provider 和 Python 扩展。

```sh
INFINI_OPS_INSTALL_PREFIX="$ops_prefix" INFINI_RT_ROOT="$rt_prefix" \
  python -m pytest -q tests/test_sparse_python.py tests/test_generate_wrappers.py \
  tests/test_generate_torch_ops.py tests/test_generate_public_operator_header.py \
  tests/test_cpp_api.py::test_installed_sparse_codegen_consumer
```
