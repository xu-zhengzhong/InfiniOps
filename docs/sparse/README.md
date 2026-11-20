# InfiniOps / InfiniRT 稀疏张量支持：分析与执行方案

状态：**D0 / P0 契约已冻结，RT1–RT3 与 OP1–OP2 已完成**。InfiniRT 已通过 `<infini/rt.h>` 提供独立 COO/CSR 描述符和 metadata-only 工厂，正式元数据测试及安装消费者已纳入 CTest；InfiniOps 已接入参数 traits、metadata-only 能力检查、无数据指针的缓存键及 checked 分派入口，稀疏 tuning 保持禁用。生成器已保留 sparse 重载及容器类型，C++ 安装入口已通过测试 provider 验证，见 [11 OP2 实现记录](11-op2-implementation.md)。冻结决议见 [06 D0 接口契约](06-d0-contract.md)，RT API、导出和测试见 [07 RT1](07-rt1-implementation.md)、[08 RT2](08-rt2-implementation.md)、[09 RT3](09-rt3-implementation.md)，Ops 基础设施边界见 [10 OP1 实现记录](10-op1-implementation.md)；类型和分派入口可用不表示稀疏计算可用。

## 结论与范围

D0 确定由 **InfiniRT 提供独立的非拥有型 `SparseTensorView`，由 InfiniOps 提供稀疏计算、格式转换、布局感知分派及 PyTorch 桥接**。保留现有 `TensorView` 和 `infini::ops::Tensor` 的稠密语义，先打通 COO / CSR 的 CPU 路径，再扩展 NVIDIA、CSC、BSR、BSC、batch 和 hybrid。

对齐对象是 PyTorch 的布局语义、组成张量、不变量、算子输入输出及后端能力约束；不复制其拥有型 `TensorImpl`、自动求导引擎或整个 dispatcher。首期明确只支持前向计算。完整布局规划和最小可交付范围分开列出，不能把首期称为“完整兼容 PyTorch sparse”。

文档集中放在 InfiniOps，覆盖两个仓库的职责和逐文件改造计划，便于跨仓库评审。InfiniRT 的专属方案见 [运行时设计](03-design.md#infinirt-表示与生命周期) 和 [RT 执行任务](04-execution-plan.md#任务拆分与合并顺序)。

## 阅读顺序

| 文档 | 解决的问题 |
| --- | --- |
| [01 PyTorch 设计与源码导读](01-pytorch-reference.md) | sparse 的含义、五种布局、源码调用链以及需要对齐的语义 |
| [02 两仓库现状与差距](02-repository-audit.md) | 现有类型、生成器、桥接和缓存为何无法直接承载 sparse |
| [03 目标设计与接口契约](03-design.md) | 类型、所有权、校验、输出容量、分派、缓存和 stream 怎样协作 |
| [04 分阶段执行计划](04-execution-plan.md) | 哪个仓库改哪些文件、依赖顺序、阶段出口、风险和决策 |
| [05 验证与验收方案](05-validation.md) | 正确性、内存安全、并发、兼容性和性能怎样验证 |
| [06 D0 接口契约](06-d0-contract.md) | 已冻结的类型/状态/checked entry/动态输出/alias/owner 决议，以及全部 38 项用例走查 |
| [07 RT1 实现记录](07-rt1-implementation.md) | 源码 API、元数据校验边界及 RT1 验证记录 |
| [08 RT2 实现记录](08-rt2-implementation.md) | 公开头生成、安装导出、增量更新及安装消费者验证 |
| [09 RT3 实现记录](09-rt3-implementation.md) | 正式元数据、边界、生命周期测试及 Release / dense / 安装回归 |
| [10 OP1 实现记录](10-op1-implementation.md) | 参数 traits、checked 分派、metadata cache、禁用 sparse tuning 及验证边界 |
| [11 OP2 实现记录](11-op2-implementation.md) | 生成器的 dense/sparse 类型、重载、公开实例化及安装消费者验证 |

## 分析基线

| 对象 | 本次核查基线 |
| --- | --- |
| InfiniOps | `/workspace/InfiniOps`，`3fac0255ef5d61a67d376c410c695d720f764bd8` |
| InfiniRT | `/workspace/InfiniRT`，`cd1d48258f0a6b8d9f66111d5d6cb1664205e78d` |
| PyTorch 源码 | [`v2.9.0`](https://github.com/pytorch/pytorch/tree/v2.9.0)，作为固定比较基线，不宣称最新版本 |
| PyTorch API 文档 | [`2.9` 文档](https://docs.pytorch.org/docs/2.9/sparse.html)；实现细节以对应 tag 源码为准 |
| 核查日期 | 2026-11-04；最终文档复核于 2026-11-05 |

仓库行号指上述分析提交的定位起点。文中“现有”描述已读源码，“建议/拟新增”描述未来工作，“待验证”表示未运行对应后端。D0 于 2026-11-12 冻结首期决议，工作基线与固定源码补读记录见 06；早期建议措辞不覆盖 D0 决议。在线版本文档可能补订，源码 tag 用于复核差异。D0 没有编译两个仓库，也没有运行或实现新的稀疏算子。

## D0 冻结的首个交付边界

- 非 batch、非 hybrid 的二维 COO / CSR；COO 使用 int64 索引，CSR 使用匹配的 int32 或 int64 索引。
- CPU float32 / float64，组件及 dense operands/out 连续、负 stride 拒绝；结构描述可表达其他 value dtype，但计算支持须逐项声明。
- 外部组件构造、组件访问、校验、COO coalesce、COO / CSR 转稠密、二维 COO 与 CSR 互转、COO / CSR × dense 的矩阵乘，以及 CSR sampled_addmm。
- C++ 公共入口、Python 输入桥接、明确的错误和生命周期契约；原生 CPU 路径无需启用 Torch provider，Torch 用于对照及可选 provider。
- 为后续动态输出建立容量/实际 nnz 协议；dense → sparse 排在该协议稳定之后。

首期不含自动求导、稀疏 × 稀疏乘法、稀疏 softmax、通用 reshape/view、隐式布局转换、自动稠密回退、2:4 半结构稀疏，以及“所有设备可用”的承诺。后续范围与限制见执行计划。

## 必须先解决的四个问题

1. `Tensor` 当前就是 RT 的稠密 `TensorView` 别名；不能只增加一个 `layout` 字段而继续复用单 `data/strides` 路径。
2. 生成器以类型名包含 `Tensor` 识别张量，必须显式区分 dense / sparse，防止静默降为 dense。
3. 稀疏结构分析依赖索引内容，不能复用只按 shape / dtype / strides 的缓存。
4. 稀疏结果的实际 nnz 可能运行后才能确定，必须先定义输出容量、结果元数据和异步完成方式。

决策和可检验的交付条件已在 D0 冻结；RT1–RT3 已完成独立 view、RT 安装导出与正式元数据测试，OP1 已完成 Ops 参数、元数据缓存键和 checked 分派基础设施。OP2 已完成生成器类型分类与公开实例化。Python 桥接、内容校验、计算 provider 与动态输出由 OP3 及后续任务交付。供应商库选择留给相应后端任务。
