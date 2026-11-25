# OP5 原生 CPU 稀疏输出、合并与转换

原生 CPU implementation 0 支持二维 COO 合并、COO→CSR 和 CSR→COO。三者都保持显式 shape 和 float32/float64 值类型，保留输入中的显式零以及重复项相加得到的零。输出索引统一为 int64。CSR→COO 接受匹配的 int32 或 int64 输入索引。

| 操作 | C++ 安装入口，包含 `<infini/ops.h>` | Python 安装入口 |
| --- | --- | --- |
| COO 合并 | `Coalesce::Prepare(handle, config, input, options, plan)` | `ops.coalesce(input)` 或 `ops.prepare_coalesce(input)` |
| COO→CSR | `ToSparseCsr::Prepare(handle, config, input, options, plan)` | `ops.to_sparse_csr(input)` 或 `ops.prepare_to_sparse_csr(input)` |
| CSR→COO | `ToSparseCoo::CheckedCall(handle, config, input, out)` | `ops.to_sparse_coo(input, out)` |

## 容量、执行与发布

1. 将默认选项 `{}` 和调用者持有的 `SparseOutputPlan` 传给 Prepare。成功后，`plan.requirements()` 给出 `nnz_upper_bound` 及索引、值、workspace、结果槽各自的字节数和对齐要求。上界为输入 `nnz`。count-first 和 capture 请求返回 `UnsupportedCapability`。
2. 填写 `SparseOutputBuffer` 的目标布局、shape、值类型、索引类型、设备及各组件的指针和容量。`indices` 对应 COO 坐标或 CSR 行指针。`col_indices` 仅用于 CSR。传入 `SparseWorkspace` 及一个已构造的主机 `SparseResultMetadata`，调用 `plan.Execute(out, workspace, &slot)`。所有写入区域按整段声明容量检查对齐、地址端点和重叠。预检失败时保留输出内容，当前调用无法发布结果。
3. Execute 的返回值报告提交状态。CPU 在返回前已将已提交的槽从 Pending 更新为 Success 或 Failed。调用 `auto completion = plan.Finalize(view)`，其中 `view` 为 `std::optional<SparseTensor>`。Success 才有有效视图和 `actual_nnz`。Pending 返回空视图，Failed 返回空视图及 `completion.error`。同一执行可多次 Finalize。复用缓冲或结果槽前，调用者须结束旧结果的使用。

未合并且含多项的 COO 使用 workspace 中的 `N` 个 `size_t` 排序位置，按行、列和原始位置排序，使同坐标的值按输入顺序求和。计算唯一坐标数后，COO 的列坐标紧接实际行坐标，容量为 10、实际为 6 时，列坐标从第 6 个偏移开始。CSR 写入列和值的有效前缀，并生成全部 `M+1` 个行指针。已验证的 coalesced 输入采用零 workspace，仍写入独立输出。

CSR→COO 的 N 已知，调用者提供相同逻辑 shape、相同 N 并声明 `coalesced=true` 的 COO 输出描述。其坐标可以尚未初始化。CheckedCall 先验证写入范围和 CSR 结构，然后展开行号、写入 int64 列号并复制 values。Python 返回 `None`，结果位于原 out 的组件存储中。

## Python 调用与存储

`ops.coalesce` 和 `ops.to_sparse_csr` 调用后直接返回稀疏张量。需要复用已有组件时，可显式使用 builder。以下例子使用 4 个条目的容量，最终得到 3 个条目。

```python
import torch
import infini.ops as ops

x = torch.sparse_coo_tensor(
    [[0, 0, 2, 2], [1, 1, 0, 3]], [2.0, 3, 0, -1], (3, 4),
    dtype=torch.float32, device="cpu",
)
builder = ops.prepare_coalesce(x)
capacity = builder.requirements["nnz_upper_bound"]
indices = torch.empty((2, capacity), dtype=torch.int64, device=x.device)
values = torch.empty(capacity, dtype=x.dtype, device=x.device)
builder.execute(indices=indices, values=values)
result = builder.finalize()
assert result.indices().stride() == (3, 1)
assert result.values().tolist() == [5.0, 0.0, -1.0]
```

`builder.requirements` 中每个缓冲项都给出 `(bytes, alignment)`。Execute 使用调用者提供的连续组件，也可为省略的组件和 uint8 workspace 分配存储。Finalize 用这些存储的有效前缀构造 PyTorch 稀疏张量。删除 builder、输入与原组件变量后，返回张量仍持有输出存储。分配中途失败会释放本次已分配的组件，可重新 Execute。

从 Prepare 到 Execute 完成，调用者须保持输入稳定。Python 检测可用的输入张量版本计数器，版本变化报告 `StaleStructure`。独立存储别名的外部写入、裸指针以及没有版本计数器的 inference tensor 由调用者保证稳定。CPU Execute 在写输出前会重新校验当前索引。

## 验收

InfiniOps 基线为 `345d5474381440960af7b16c264dd52c31f6318d`，配套 InfiniRT 为 `6ffd4e671fce8c591b3b188e30b6dcbb8cd4892e`。Release 安装启用 CPU、Python 绑定和公开 C++ 实例化，关闭 Torch provider。使用 GCC 13.3、Python 3.12 和 pybind11 3.1.0。

| 用例 | 运行证据 |
| --- | --- |
| S01、S02、S09、S10、S15 | 对照 PyTorch 2.9.0+cpu 和解析坐标，覆盖两种值类型、CSR 两种索引宽度、重复项抵消、显式零、空矩阵及 COO↔CSR 往返。分别检查 shape、索引、值、nnz 和 dense 结果。另用随机种子 105 对照 64 组矩阵，每组 0 至 1025 个输入条目，合并及往返结果的坐标和值均一致。 |
| S12、O01、O02 | 安装端 C++ 和 Python 验证容量 10、实际 6 的打包和前后哨兵。少一项、错误零容量、workspace 错误、未对齐、地址溢出和跨组件重叠均有拒绝用例。 |
| O04、O05、R04 | 覆盖版本变化、错误内容导致 Failed、Pending 槽复用、重复 Finalize 和结果槽代次。Python 检查组件地址共享、删除 owner 后的结果以及注入部分分配失败后的回收与重试。 |
| 安装与内存边界 | 固定 PyTorch 基线上的稀疏 CPU 专项共 59 项通过。仅生成 Python 绑定的三算子精简安装通过全部 16 项适用的 Python OP5 用例。独立 Debug 安装的 C++ 消费者通过 ASan/UBSan 检查。 |
| 回归 | 稀疏专项、三个生成器和安装消费者共 167 项通过。PyTorch 2.11.0+cu128 的完整 CPU 套件为 2503 passed、7377 skipped、21 failed。21 个失败 ID 与 OP4 完全一致，已用 OP4 安装复跑。涉及 CPU Embedding 15 项、CUTLASS 绑定 2 项、CUDA 编译探针 2 项、Ascend workspace 编译探针 1 项、configless C++ 消费者 1 项。 |

将 `PYTHONPATH` 指向本次安装包，并设置 `INFINI_OPS_INSTALL_PREFIX` 和 `INFINI_RT_ROOT`，可运行下列命令。C++ 消费者只使用安装头和库。专项和边界断言见 [Python 测试](../../tests/test_sparse_cpu.py) 和 [C++ 测试](../../tests/cpp/sparse_output.cc)。

```sh
python -m pytest tests/test_sparse_cpu.py --devices cpu -q
python -m pytest tests --devices cpu -q -n 10 --dist worksteal
```

本次运行的参考对照、安装集成、sanitizer 和完整回归日志分别位于 `/tmp/infiniops-op5-reference.log`、`/tmp/infiniops-op5-integration.log`、`/tmp/infiniops-op5-san-test.log` 和 `/tmp/infiniops-op5-full.log`。逐项基线复跑记录为 `/tmp/infiniops-op5-baseline-recheck.xml`。
