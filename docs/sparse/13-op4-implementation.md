# OP4 原生 CPU ToDense 和 SparseMm

P1 的两个算子通过安装后的 C++ 和 Python 接口接受二维 COO/CSR，使用 CPU implementation index 0。COO 索引为 int64，CSR 两个索引组件同为 int32 或同为 int64。所有值、稠密右矩阵和输出使用同一种 float32 或 float64 类型，所有组件和稠密操作数均须连续。原生库可用 `WITH_CPU=ON`、`WITH_TORCH=OFF` 构建。

## 调用

| 操作 | 安装端 C++（包含 `<infini/ops.h>`） | Python（`import infini.ops as ops`） | 输出 shape |
| --- | --- | --- | --- |
| 转稠密 | `ToDense::CheckedCall(handle, config, input, out)` | `ops.to_dense(input, out)` | 与 input 相同 |
| 稀疏矩阵乘法 | `SparseMm::CheckedCall(handle, config, mat1, mat2, "sum", out)` | `ops.sparse_mm(mat1, mat2, "sum", out)` | `[M,K] × [K,P] → [M,P]` |

C++ 调用者检查返回的 `ValidationResult`。Python 成功时返回 `None`，失败时抛出对应异常。输出由调用者预分配，成功返回时结果已写入原地址。稀疏右矩阵和非 `sum` 归约返回 `UnsupportedCapability`。

## 计算与校验

- `Make`、`CanMake` 和 `CheckedCall` 共用元数据检查。ToDense 检查逻辑输出尺寸，SparseMm 还检查右矩阵的秩、内积维和归约属性。这些检查在 provider 选择前返回形状或能力诊断。
- 执行前按有效字节区间检查输出与每个输入组件、稠密右矩阵的重叠。两个只读输入可以共享存储，首尾相接及零长度区间也合法。随后复用 `src/native/cpu/sparse_validation.h` 验证当前 indices，即使输出有零列也会检查。校验失败时保留原输出。
- 先清零输出。COO 按存储顺序遍历坐标，CSR 按行指针遍历。`ToDense` 将值累加到对应坐标，`SparseMm` 将该值乘以右矩阵对应行并累加到输出行。两者使用输入值的精度，保留输入的重复坐标和显式零。算法工作区需求为 0 字节。
- Python 借用原始组件。PyTorch 合并重复坐标后可能保留较大的索引行步长。例如逻辑形状为 `[2,3]`、步长为 `(4,1)` 的 coalesced COO 会返回 `UnsupportedCapability`。coalesced 声明和组件连续性分别校验。

## 验收环境

InfiniOps 以 `eba3200d60411e19e6e6dd0ff4e11f548e849662` 加本次改动构建，配套 InfiniRT 为 `6ffd4e671fce8c591b3b188e30b6dcbb8cd4892e`。主构建使用 x86-64 Intel Xeon Platinum 8380、GCC 13.3、Release、Python 3.12、pybind11 3.1.0，启用公开实例化和 Python 绑定。原生库链接 InfiniRT。固定数值参考使用 PyTorch `2.9.0+cpu`（`0fabc3ba44823f257e70ce397d989c8de5e362c1`）。完整 CPU 回归沿用 OP3 的 PyTorch `2.11.0+cu128`（`70d99e998b4955e0049d13a98d77ae1b14db1f45`）。

| 验收项 | 实际覆盖 |
| --- | --- |
| S05、S09、S10、S14、S16 | 六种布局/值/索引类型组合与 PyTorch 比对。包含未合并 COO 重复项、CSR 空行、尾部空行列、显式零、空内积、0/1/7 个输出列、已有非零输出和输出边界哨兵。 |
| S03、S06–S08 | 安装 C++ 消费者绕过 Python 导入，直接检查假 coalesced、错误 crow 首尾/区间、行内乱序/重复、负值和越界位置。两个算子均在写输出前报告失败。 |
| S11–S13、S19、R08 | 形状、类型、设备编号、非连续/负步长、空指针、对齐、地址端点与字节数溢出、跨组件及部分地址重叠。C++ 检查 `NoProvider`，Python 检查未支持布局、归约、稀疏右矩阵、显式 implementation 和梯度错误。 |
| C02–C04 | 交替调用几何相同而坐标不同的矩阵，原地改写索引和值后重新计算。输出与当前结构和值一致。 |
| 内存边界 | 独立 Debug 构建对两个原生 provider 和安装消费者启用 ASan/UBSan。C++ 验收程序成功退出，包含非法 CSR 区间、重叠、空指针及无需大分配的溢出样例。 |

| 执行范围 | 结果 |
| --- | --- |
| PyTorch `2.9.0+cpu` 对照和安装端 C++ 消费者 | OP4 专项 42 项全部通过，包含空维度和明确拒绝的输入。 |
| 稀疏专项、三个生成器及安装接口回归 | 150 项全部通过。 |
| 完整 CPU 套件 | 2486 项通过，7377 项跳过，21 项失败。21 个失败用例在变更前的 OP3 源码和安装包上逐项复现，涉及 CPU Embedding 15 项、CUTLASS 绑定 2 项、CUDA 编译探针 2 项、Ascend workspace 编译探针 1 项、无显式配置的 C++ 消费者 1 项。 |

专项使用 `tests/test_sparse_cpu.py`，其中的 C++ 消费者仅包含安装的 `<infini/ops.h>`。复现时将 `PYTHONPATH` 指向含本次安装包的目录，设置两个安装前缀，在对应 PyTorch 环境执行以下命令。

```sh
INFINI_OPS_INSTALL_PREFIX="$ops_prefix" INFINI_RT_ROOT="$rt_prefix" \
  python -m pytest tests/test_sparse_cpu.py --devices cpu -q
```

完整回归使用相同安装前缀，执行 `python -m pytest tests --devices cpu -q -n 10 --dist worksteal`。OP3 的对应结果见[上阶段验证记录](12-op3-implementation.md#验证)。
