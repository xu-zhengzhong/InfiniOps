# OP6 原生 CPU 的 CSR sampled_addmm

原生 CPU implementation 0 按输入 CSR 的存储坐标计算矩阵乘加。输出有相同的逻辑 shape、`nnz`、行指针和列索引。显式零也占一个采样位置。输入和输出的索引均为 int32 或均为 int64，值和两个稠密矩阵均为 float32 或均为 float64。所有组件和稠密操作数须连续，计算同步完成。

| 安装入口 | 调用 |
| --- | --- |
| C++，包含 `<infini/ops.h>` | `SparseSampledAddmm::CheckedCall(handle, config, input, mat1, mat2, beta, alpha, out)` 返回 `ValidationResult` |
| Python，导入 `infini.ops as ops` | `ops.sparse_sampled_addmm(input, mat1, mat2, beta, alpha, out)` 写入 `out`，返回 `None` |

## 固定输出与计算

`input` 和 `out` 的 shape 都是 `[M, N]`，`mat1` 为 `[M, K]`，`mat2` 为 `[K, N]`。调用者提供可写的 `out.crow_indices[M+1]`、`out.col_indices[nnz]` 和 `out.values[nnz]`。这三个存储区间必须彼此独立，并与所有输入的有效区间分离。调用前可以让输出索引处于未初始化状态。

每次调用在首次写入前检查元数据、写入区间和输入 CSR 的行指针、列边界及行内严格递增顺序。随后复制行指针和列索引，对每个存储坐标 `(i, j)` 计算 `alpha * dot(mat1[i, :], mat2[:, j]) + beta * input.values[p]`。系数转换和点积累加均使用值类型。执行耗时为 `O(M + nnz + nnz*K)`，无需额外矩阵或调用者工作区。空 CSR 也会写满全部 `M+1` 个行指针。

以下 S17 样例的第一项显式存有零，其结果是 26。

```python
import torch
import infini.ops as ops

with torch.device("cpu"):
    mask = torch.sparse_csr_tensor(
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0.0, 2.0]),
        (1, 2),
    )
    mat1 = torch.tensor([[2.0, 3.0]])
    mat2 = torch.tensor([[4.0, 5.0], [6.0, 7.0]])
    out = mask.clone()
    ops.sparse_sampled_addmm(mask, mat1, mat2, beta=1, alpha=1, out=out)

assert out.crow_indices().tolist() == [0, 2]
assert out.col_indices().tolist() == [0, 1]
assert out.values().tolist() == [26.0, 33.0]
```

S18 覆盖两条数值路径。当 `K > 0` 时，零系数仍参与浮点乘法，因此 `beta=0` 乘输入 NaN/Inf、`alpha=0` 乘 NaN/Inf 点积都会得到 NaN。当 `K=0` 时，每项只写入 `beta * input.values[p]`，例如 `alpha=NaN`、`beta=3`、输入值为 2 时结果为 6。

## 迁移与错误

旧 dense 构造和调用重载保留弃用标记，删除需要单独的破坏性变更。C++ 旧调用抛出 `std::invalid_argument`，Python 旧调用抛出 `ValueError`，提示改用 CSR 输入和输出。Python 模块函数按布局选择重载，旧入口在读取张量存储前给出提示。

CSR 入口使用 checked 状态。输出 shape、存储项数或值类型不匹配返回 `InvalidMetadata`，输出索引宽度与输入不同返回 `InvalidIndexType`，缓冲区重叠返回 `Aliasing`。非法行指针、行内重复或乱序列返回 `InvalidStructure`，越界列返回 `IndexOutOfBounds`。预检失败时三个输出组件均保留调用前内容。显式选择未注册的 implementation 8 返回 `UnsupportedImplementation`。

## 验收

InfiniOps 基线为 `c8b4ef9494a809a77c87e410f6680257f2dc050f`，配套 InfiniRT 为 `6ffd4e671fce8c591b3b188e30b6dcbb8cd4892e`。Release 安装启用 CPU、公开 C++ 实例化和 Python 绑定，关闭 Torch provider。工具链为 GCC 13.3、Python 3.12 和 pybind11 3.1.0。

| 范围 | 实测结果 |
| --- | --- |
| S17、S18、空形状与结构复用 | 对照 PyTorch 2.9.0+cpu 及解析值，覆盖两种值类型、两种索引宽度、负系数、标量舍入、NaN/Inf 和空内积。检查了输出存储地址、输入组件、空行和索引原地修改后的结果。PyTorch 对 int32 输入自动分配的结果使用 int64 索引，显式传入 int32 `out` 时保留 int32。本次使用对应的显式输出比较结构和数值。 |
| 随机对照 | 固定种子 106，在两种值类型和两种索引宽度下各运行 32 组，共 128 组。采样密度覆盖 0、5%、30% 和 100%，内积长度覆盖 0、1、7、29 和 65。每组与 PyTorch 显式输出逐项比较 shape、`nnz`、索引和值，全部通过。 |
| 安装与内存边界 | 固定版本下，稀疏 CPU、Python 导入、checked 分派、三个生成器和安装消费者共 195 项通过，零跳过。独立 Debug 安装的 C++ 消费者通过 ASan/UBSan，覆盖输出哨兵、跨组件重叠、非法 CSR 及失败前后内容。 |
| 启用 Torch 的精简 CPU 安装 | 在 Torch 生成列表中显式请求 `abs` 和 `sparse_sampled_addmm`。普通 `abs` 通过 slot 8 执行，`SparseSampledAddmm` 的 CPU 实现列表为 `[0]`。27 项 OP6 Python 专项全部通过。 |
| 完整 CPU 回归 | PyTorch 2.11.0+cu128 环境运行 CPU 套件，2531 项通过、7377 项跳过、21 项失败。JUnit 失败 ID 与 OP5 完整回归逐项一致，涉及 Embedding 15 项、CUTLASS 绑定 2 项、编译探针 3 项和无 config 的 C++ 消费者 1 项。 |

将 `PYTHONPATH` 指向本次安装包，并设置 `INFINI_OPS_INSTALL_PREFIX` 和 `INFINI_RT_ROOT` 后，可运行以下命令。C++ 测试只包含安装入口 `<infini/ops.h>`。用例见 [Python 稀疏 CPU 测试](../../tests/test_sparse_cpu.py) 和 [C++ 安装消费者](../../tests/cpp/sparse_cpu.cc)。

```sh
python -m pytest tests/test_sparse_cpu.py --devices cpu -q
python -m pytest tests --devices cpu -q -n 10 --dist worksteal
```

本机验证日志为 `/tmp/infiniops-op6-reference.log`、`/tmp/infiniops-op6-san-test.log`、`/tmp/infiniops-op6-torch-test.log` 和 `/tmp/infiniops-op6-full.log`。
