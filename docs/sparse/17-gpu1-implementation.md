# GPU1 NVIDIA 原生稀疏算子

`WITH_NVIDIA=ON` 构建为六个算子注册原生 implementation 0。C++ 使用与 CPU 相同的 `CheckedCall` 或 `Prepare / Execute / Finalize`，Python 使用对应的 `infini.ops` 入口。已验证 CUDA 编译、安装、注册和访问设备内存前的错误处理。本次环境中 CUDA 无法枚举 GPU，表中的设备数值行为、内存检查和耗时仍待真机验收。

## 实现矩阵

输入限二维、无 batch/hybrid、连续组件、同设备的 float32 或 float64。COO 坐标为 int64，CSR 行指针和列号同为 int32 或 int64。稠密操作数和输出也须连续且值类型相同。

| 操作 | 输入与输出 | 设备算法 |
| --- | --- | --- |
| 内容校验 | COO、CSR | COO 检查两轴边界和 coalesced 声明。CSR 检查首尾指针、每个区间及行内列序，每个线程只在确认区间位于 `[0,N]` 后读取列号。 |
| ToDense | COO、CSR → 预分配 dense | 清零后按存储项写入。COO 重复坐标通过原子加累加。 |
| SparseMm | COO、CSR × dense → 预分配 dense | `reduce="sum"`。CSR 按行区间计算输出元素，COO 按存储项累加对输出行的贡献。显式零参与乘法。 |
| Coalesce、ToSparseCsr | COO → coalesced COO 或 CSR | 复验已合并声明后直接写回。其余多条目输入按坐标稳定排序、选出每组起点，再按原存储顺序求和。零和保留为存储项。 |
| ToSparseCoo | CSR → 定长 coalesced COO | 展开各行的区间并写 int64 坐标，同时复制 values。int32 列号在这一步显式扩宽。 |
| SparseSampledAddmm | CSR → 同结构、同索引宽度的 CSR | 复制 pattern，对每个存储坐标计算点积，按值类型执行 `alpha * dot + beta * input`。内积维度为零时只缩放 input。输出可含未初始化的索引。 |

稀疏输入索引始终保留原宽度。对于需要排序的 COO，CUB 2.7 的 `DeviceSelect::If` 接收 `int` 条目数。Prepare 在读索引前检查 `N <= INT_MAX`，超过时返回 `UnsupportedCapability`。该限制只针对需要排序的条目数。排序键使用无符号 64 位的 `row * width + col`，RT 已检查逻辑元素数乘法，设备校验再证明坐标有界。[CUB DeviceSelect](https://github.com/NVIDIA/cccl/blob/v2.7.0/cub/cub/device/device_select.cuh)

COO 的 ToDense 和 SpMM 使用原子归约，重复项对同一输出的浮点加法顺序可能变化。float32 的原子累加使用 CAS 和普通舍入加法以保留次正规数。sampled_addmm 的乘法和加法分别舍入，以覆盖乘积溢出后与异号无穷相加的情形。动态转换使用 CUB 稳定 radix sort，按相同坐标的输入顺序累加。ATen v2.9.0 的 CUDA sampled_addmm 调用 cuSPARSE SDDMM 并查询其临时空间。GPU1 用原生 kernel 实现 D0 的 alpha、beta 及空内积分支，排序和选择使用 CUDA Toolkit 自带的 CUB 与计数迭代器，运行时链接沿用 CUDA Runtime。[CUB DeviceRadixSort](https://github.com/NVIDIA/cccl/blob/v2.7.0/cub/cub/device/device_radix_sort.cuh)、[ATen CUDA 实现](https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp)

## Workspace、同步与错误

- C++ 在输入设备和 `Handle::stream()` 上执行并恢复原设备。Python 使用该设备的 PyTorch 当前 stream。固定输出如指定 `stream=`，它须与当前 stream 一致。动态 builder 在 Prepare 和 Execute 时使用同一 stream。若组件来自其他 stream，调用方须先建立生产者到当前 stream 的依赖。
- 每次固定调用、Prepare 和 Execute 都检查输入结构。固定调用和 Prepare 临时分配一个设备诊断槽，读回错误码、组件和位置。Execute 复用调用方 workspace。空 COO 免扫，空 CSR 仍检查全部行指针。校验失败时不启动计算 kernel。
- `plan.requirements()` 和 Python 的 `builder.requirements` 给出各个容量和对齐。已合并或单项 COO 的 Execute 只需要诊断槽，空 COO 需要零字节。需排序时，workspace 包含两组 64 位键、排序后的 values、组起点和计数、对齐填充，以及 sort/select 两次查询中的较大临时空间。两次调用共用这段空间，基址要求 256 字节对齐。
- Execute 在得到实际条目数后按 `[2,actual_nnz]` 紧凑写 COO，CSR 写全长 crow 和有效的 col/values 前缀。Finalize 只发布成功结果。容量、别名及 workspace 错误在写入前返回，已提交的执行错误记录在结果槽中。
- 同步入口返回前等待消费 stream 完成，Python 在此期间保活各组件。活动 capture 返回 `UnsupportedCapability`。异步提交、跨 stream owner 和固定结构 capture 属于 GPU2。[D0 完成与所有权契约](06-d0-contract.md)

## 验证记录

InfiniOps 基线为 `463110555a21d56bf2774a6549693a44e82e8a65`，InfiniRT 为 `6ffd4e671fce8c591b3b188e30b6dcbb8cd4892e`。Release 使用 GCC 13.3、NVCC 12.8.93 和 CUB 2.7.0。

| 范围 | 结果 |
| --- | --- |
| CUDA 编译与安装 | CUDA 12.8 的 CPU+NVIDIA 构建安装了六个算子的 C++ 和 Python 入口。关闭 CPU 和 Torch provider 后，NVIDIA 默认算子集也通过 Release 编译和安装，两种配置目标均为 sm_80。另用 NVCC 为 sm_52 编译了校验、固定输出和动态转换三个 CUDA 翻译单元。sm_80 PTX 中，sampled_addmm 的四种索引和值类型组合均生成分别舍入的乘、加指令。 |
| 独立安装消费者 | NVIDIA 专项中 2 项通过。Python 检查六个算子均注册 slot 0，C++ 通过 `find_package(InfiniOps)` 运行超大条目数、容量算术溢出、alias 和显式 slot 错误路径。 |
| 设备验收 | PyTorch 2.11.0+cu128 枚举到 0 张卡，NVML 初始化失败。52 项设备用例跳过，含 S17/S18、两个索引宽度、紧凑打包、跨 block 排序、大于 INT32_MAX 的列号、错误后的重试、float32 次正规数、乘法溢出后的 NaN、非默认 stream 和 capture 拒绝。Compute Sanitizer 和性能数据待实机。 |
| CPU 与生成器回归 | 三类生成器 83 项通过。PyTorch 2.9.0+cpu 的 CPU ATen 稀疏、原生稀疏、分派和安装消费者专项 193 项通过。原生 CPU 全套有 2534 项通过、7430 项跳过、21 项失败。与 AT1 原生 CPU 基线逐项比对 JUnit，失败 ID 完全相同。 |

真机复验使用与设备匹配的 CUDA 架构构建 InfiniRT 和 InfiniOps，启用 `WITH_NVIDIA`、`GENERATE_OPERATOR_CALL_INSTANTIATIONS` 和 `GENERATE_PYTHON_BINDINGS`。将 `PYTHONPATH` 指向安装包父目录，设置 `INFINI_OPS_INSTALL_PREFIX` 和 `INFINI_RT_ROOT` 后执行以下命令。C++ 消费者由 pytest 编译并启动。[Python 验收用例](../../tests/test_sparse_nvidia.py)、[C++ 安装消费者](../../tests/cpp/sparse_nvidia.cc)

```sh
python -m pytest tests/test_sparse_nvidia.py --devices nvidia -q -rs
compute-sanitizer --tool memcheck --target-processes all --error-exitcode 1 \
  python -m pytest tests/test_sparse_nvidia.py --devices nvidia -q
```
