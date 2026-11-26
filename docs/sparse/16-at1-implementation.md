# AT1 可选 CPU ATen 稀疏实现

启用 `WITH_CPU=ON` 和 `WITH_TORCH=ON` 后，六个稀疏算子的默认 CPU 安装同时提供原生 slot 0 和 ATen slot 8。Python 调用传入 `implementation_index=8`，C++ 调用在传给 `CheckedCall` 或 `Prepare` 的 `Config` 上设置 `set_implementation_index(8)`。关闭 Torch provider 的 CPU 安装提供 slot 0，显式请求 8 返回 `UnsupportedImplementation`。

| 算子 | 输入与 ATen 运算 | 调用方收到的输出 |
| --- | --- | --- |
| `ToDense` | COO 或 CSR 的 `to_dense()` | 写入预分配的稠密矩阵 |
| `SparseMm` | COO 使用双参数 sparse mm，CSR 使用显式 sum 的 sparse mm；右侧为稠密矩阵 | 写入预分配的 `[M,P]` 矩阵 |
| `Coalesce` | COO 的 `coalesce()`，重复坐标求和，保留显式零 | 按实际 N 紧凑写入 `[2,N]` 坐标和值，通过 Finalize 发布 |
| `ToSparseCsr` | COO 合并后转 CSR | 写入全部行指针及列、值的有效前缀，通过 Finalize 发布 |
| `ToSparseCoo` | CSR 转 coalesced COO | 写入精确大小的 int64 坐标和值 |
| `SparseSampledAddmm` | 在输入 CSR 的私有副本上调用 ATen sampled_addmm | 写入原输出的行指针、列和值，保留输入的索引宽度与全部采样位置 |

验证范围为同步 CPU 设备 0、二维标量条目、连续组件及稠密操作数、float32/float64。COO 索引为 int64，CSR 的两种索引组件同为 int32 或同为 int64。稀疏逻辑元素数超出 ATen 的有符号 64 位范围时返回 `IntegerOverflow`。NVIDIA 执行与跨 stream 生命周期由 GPU1、GPU2 交付。

```python
import torch
import infini.ops as ops

x = torch.sparse_coo_tensor(
    [[1, 0, 1], [0, 2, 0]], [1.0, 3.0, -1.0], (3, 4),
    device="cpu", dtype=torch.float32, is_coalesced=False,
)
builder = ops.prepare_coalesce(x, implementation_index=8)
indices = torch.empty((2, 3), dtype=torch.int64, device="cpu")
values = torch.empty(3, dtype=x.dtype, device="cpu")
builder.execute(indices=indices, values=values)
y = builder.finalize()
assert y.indices().tolist() == [[0, 1], [2, 0]]
assert y.values().tolist() == [3.0, 0.0]
assert y.indices().data_ptr() == indices.data_ptr()
assert y.values().data_ptr() == values.data_ptr()
```

## 组件、输出与失败状态

每次执行先检查写入范围、重叠和输入索引。provider 借用各组件的指针、shape、stride 和 dtype，构造具有原始逻辑尺寸、布局和设备的 ATen 张量。COO 使用 raw indices、raw values 和经校验的 coalesced 标记。CPU stream guard 覆盖包装、计算和复制，动态 plan 保存 Prepare 时的 stream。通用 Torch 生成器跳过这些算子及其 ATen 内部拼写，清理旧生成文件。

固定输出在手写 provider 内从 ATen 结果复制。sampled_addmm 的私有 CSR 副本使 ATen 可以调整存储，调用方的输出索引可以尚未初始化。完成后检查结果的布局、shape、条目数、值和索引类型，再复制到原组件。float32 的 alpha 和 beta 先按值类型转换，包括有限 double 超出 float32 范围得到无穷大的情形。空内积和 NaN/Inf 按 [D0 数值契约](06-d0-contract.md#数值与特殊值冻结) 计算。

动态输出复用 Prepare、Execute、Finalize。Execute 先检查上界容量，再对 ATen 的实际 N 检查上界，最后复制到按 N 描述的目标视图。ATen 合并后可能保留输入长度作为坐标行步长，逐组件的 `copy_` 会将第二行写到调用方的第 N 个偏移。Python 结果继续持有调用方存储。

框架缺少内核、分配失败和其他执行异常分别映射为 `UnsupportedCapability`、`AllocationFailure` 和 `ExecutionFailure`。固定调用立即报告状态。动态调用提交成功后，执行失败记录在结果槽中，Finalize 返回该失败，调用方可以重新 Execute。Python 分派钩子留下的异常也在绑定边界转换。

## 复制量与实测

动态资源查询的 `result_copy_bytes_upper_bound` 给出最终写回的最大字节数。设输入容量为 C，实际输出为 N，值宽为 V，行数为 M。紧凑 COO 实际复制 `N*(16+V)` 字节，CSR 实际复制 `8*(M+1)+N*(8+V)` 字节。将 N 换成 C 即得上界。slot 8 的调用方 workspace 为零，ATen 使用自己的分配器管理临时结果和排序缓冲。原生 slot 0 直接写入组件，此复制量为零。C++ 消费者应使用同一构建的安装头和库重新编译。

以下为 PyTorch 2.9.0+cpu、单线程的本机样例。种子 161 的均匀随机 COO 为 `[256,512]`、float32、4096 个存储项，合并后有 4041 个坐标。每项预热 10 次，取 11 组、每组 40 次的每调用中位耗时。独立复制一列在预分配张量上测量同样的组件复制。Execute 一列复用 plan 和缓冲，端到端一列调用 Python 便捷入口，包含分配、校验、计算、复制与发布。

| 算子 | 复制上界 / 实际字节 | 独立复制 | Execute | 端到端 |
| --- | --- | --- | --- | --- |
| Coalesce | 81920 / 80820 | 5.0 µs | 373.6 µs | 427.6 µs |
| ToSparseCsr | 51208 / 50548 | 3.8 µs | 391.5 µs | 468.1 µs |

## 验收

InfiniOps 基线为 `1ba9d1357abd06167682ccd4c74c3f70a350b339`，配套 InfiniRT 为 `6ffd4e671fce8c591b3b188e30b6dcbb8cd4892e`。Release 使用 GCC 13.3、Python 3.12、pybind11 3.1.0，开启公开 C++ 实例化与 Python 绑定。

- **注册和安装**。使用 PyTorch 2.9.0+cpu，显式将六个稀疏算子与 `abs` 一起交给 Torch 生成器，六个 CPU 稀疏算子均报告 `[0, 8]`。完整 Torch 默认列表生成 495 个通用算子的 647 个重载，并与手写实现共同编译安装。
- **正确性**。安装接口、稀疏和生成器专项 276 项通过。另以种子 161 对照 PyTorch 的 128 组输入，覆盖两种值类型、两种 CSR 索引宽度、未合并重复项、显式零、空维度和零输出列。比较 shape、布局、N、索引和值。
- **R07 与错误路径**。分派探针观察到传入 ATen 的 COO/CSR 组件地址与原始输入一致。在 sampled_addmm 内强制改变私有 CSR 的存储地址，调用方三个组件的地址和写回内容均正确。异常注入覆盖同步失败、动态失败和同一 builder 重试。[ATen 边界测试](../../tests/test_sparse_aten.py)
- **内存边界**。独立 Debug 安装通过 `ops.json` 为六个算子只选择 slot 8。两个仅包含 `<infini/ops.h>` 的消费者通过 ASan（开启泄漏检查）和 UBSan，覆盖容量、重叠、哨兵、无效索引和结果槽。[固定输出消费者](../../tests/cpp/sparse_cpu.cc)、[动态输出消费者](../../tests/cpp/sparse_output.cc)

完整 CPU 套件的结果如下。启用 Torch 的对照在同一 PyTorch 2.9.0+cpu 工具链上重新构建并运行了改动前提交。关闭 Torch 的对照采用 OP6 的 PyTorch 2.11.0+cu128 CPU 回归记录。两组均逐项核对 JUnit 测试 ID，21 个失败项全部与对应基线一致，涉及 Embedding 15 项、CUTLASS 绑定 2 项、编译探针 3 项和一个 C++ 消费者。

| CPU 构建 | 通过 | 跳过 | 失败 |
| --- | ---: | ---: | ---: |
| 完整 Torch provider，改动前 | 6273 | 5406 | 21 |
| 完整 Torch provider，AT1 | 6354 | 5406 | 21 |
| 关闭 Torch provider，AT1 | 2534 | 7385 | 21 |

将 `PYTHONPATH` 指向安装包的父目录，并设置 `INFINI_OPS_INSTALL_PREFIX` 和 `INFINI_RT_ROOT`，即可运行安装端专项与全量 CPU 测试。

```sh
python -m pytest tests/test_sparse_aten.py tests/test_sparse_cpu.py \
  tests/test_sparse_python.py tests/test_sparse_operator.py \
  tests/test_generate_wrappers.py tests/test_generate_public_operator_header.py \
  tests/test_generate_torch_ops.py tests/test_cpp_api.py::test_installed_sparse_codegen_consumer \
  --devices cpu -q
python -m pytest tests --devices cpu -q -n 10 --dist worksteal
```
