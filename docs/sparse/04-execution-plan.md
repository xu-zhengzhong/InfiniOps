# 分阶段执行计划

本页列出实现任务；**D0 / P0 文档契约已完成冻结**，见 [D0 决议及走查](06-d0-contract.md)。RT1 及后续代码、构建、运行验证和发布仍未完成。阶段以依赖和验收门槛排序，不给出未经团队规模、硬件和供应商 SDK 确认的日期承诺。

## 阶段与交付范围

| 阶段 | 范围 | 必须达到的出口 |
| --- | --- | --- |
| P0 契约冻结 | RT/Ops 对齐类型、错误、校验、输出、layout 能力和生命周期 | 以本文档的可执行用例评审接口；所有首期限制可被显式拒绝 |
| P1 首条 CPU 链路 | RT COO/CSR view；Ops 参数/生成器/checked import；ToDense、SparseMm | 真正 sparse 输入经 C++ 安装入口和 Python 入口运行，原生 CPU 无 Torch provider 依赖 |
| P2 CPU 最小完整支持 | coalesce、二维 COO↔CSR、固定 pattern sampled_addmm、动态输出 builder | README 首期边界完整通过；结构与数值一致，空张量不被 skip |
| P3 NVIDIA 与异步 | GPU content validation、COO/CSR 基础算子、stream/owner/plan；可选 ATen provider | CPU/NVIDIA 逐算子矩阵，有跨 stream 与生命周期证据；无隐式 densify |
| P4 通用布局扩展 | dense→sparse、CSC、BSR/BSC、hybrid、batch | 每增加一项同时扩展表示/算法/桥接/测试；未完成项仍拒绝 |
| P5 按需求扩展 | sparse×sparse、更多 sparse 算子、autograd、半结构稀疏 | 独立需求与设计；不以 P1–P4 完成暗示这些功能可用 |

P1 是可独立演示的里程碑；P2 完成后才满足本文定义的 CPU 最小完整支持。P3 才将相关设备标记为支持。自动求导和 2:4 不是首期里程碑的前置条件。

## 任务拆分与合并顺序

路径以仓库根为起点；“新增”表示拟议文件，不是当前存在的实现。文件名可在 P0 按仓库命名规范调整，但职责和交付要求不能丢失。

| ID | 仓库 / 主要文件 | 交付 | 依赖与验收 |
| --- | --- | --- | --- |
| D0 | 两库 docs：core-types、compatibility；Ops operator-api-alignment、sparse/06-d0-contract | **已完成契约冻结**：SparseTensorView、校验结果、checked entry、dynamic output、alias/owner | P0；[38 项验证样例逐项走查](06-d0-contract.md#验证样例逐项走查)，不计作实现测试通过 |
| RT1 | 新增 `src/tensor_layout.h`、`src/sparse_tensor_view.h` / `.cc` | 独立布局和组件描述；metadata-only 工厂，不依赖 Torch/稀疏供应商库 | D0；dense `TensorView` 对象布局与 hash/equality 不变 |
| RT2 | `scripts/generate_public_headers.py`、`CMakeLists.txt`、`src/CMakeLists.txt` | umbrella、include 重写、安装导出新类型 | RT1；安装后仅 include `<infini/rt.h>` 可用 |
| RT3 | 新增 `tests/test_sparse_tensor_view.cc`；`tests/CMakeLists.txt`、`tests/install_consumer_smoke.cc` | 元数据/溢出/设备/空组件测试；dense 回归和安装消费者 | RT1–RT2；Release 下错误输入也被拒绝 |
| OP1 | `src/tensor.h` 或新增 sparse alias 头；`src/operator.h`、`src/tuning.h` | 参数 traits、设备/能力检查、sparse metadata cache；默认禁用 sparse tuning | RT2；不同 layout/indices 不误复用；明确 checked entry 状态 |
| OP2 | `scripts/generate_wrappers.py`、`scripts/generate_public_operator_header.py` | 显式 dense/sparse 类型分类、overload/optional/list 保真、公开实例化 | RT2，可与 OP1 并行；新类型不降为 dense，旧 dense 输出稳定 |
| OP3 | `src/pybind11_utils.h`、绑定生成与相应测试 | layout-aware Python 路由、组件 owner、checked import | OP1–OP2；未 coalesced COO 导入不被无声规范化 |
| OP4 | 新增 sparse validation、`src/base/` 中 ToDense / SparseMm 类、`src/native/cpu/ops/` provider | CPU float32/64、COO int64、CSR int32/64；稀疏内容检查和固定 shape 输出 | OP1–OP3；P1 完成 |
| OP5 | 新增 output builder / result metadata、coalesce、转换 providers | Prepare/Execute/Finalize、indices 紧凑打包、Python 显式动态结果返回 | OP4；实际 nnz 和 capacity 不混淆，失败不发布结果 |
| OP6 | `src/base/sparse_sampled_addmm.h`、`scripts/torch_ops.yaml`、专用 CPU provider | canonical sparse 签名、旧 dense 迁移诊断、保留输入结构 | OP2、OP4；可与 OP5 并行，OP5/OP6 均通过后 P2 完成 |
| AT1 | 新增 `src/torch/sparse_tensor.h`、手写 `src/torch/ops/` provider；`scripts/generate_torch_ops.py` | 组件级 ATen 包装、stream guard、输出分配保护；防止旧通用生成路径接管 sparse | OP3–OP6；CPU 路径先验，slot 8 不冲突 |
| GPU1 | `src/native/cuda/` 对应 NVIDIA provider、CMake 依赖声明 | 验证/转换/SpMM/采样；供应商支持矩阵和 workspace；明确索引降位政策 | P2；可先打通 CSR，逐算子推进 COO，未完成组合显式拒绝 |
| GPU2 | `src/handle.h` / plan 集成、Python owner 适配；RT 现有通用资源能力按需补齐 | 跨 stream、资源回收、结构 plan、固定结构 capture 能力 | GPU1；启用 ATen provider 时额外依赖 AT1；仅经验证的 provider 声明 capture-safe |
| EXT1 | RT sparse 表示、Ops layout contract/转换/provider/测试 | CSC、block、hybrid、batch 分批 PR | P3；按单独特性矩阵验收 |

推荐 PR 主线：D0 → RT1/RT2/RT3 → OP1/OP2 → OP3/OP4 → OP5/OP6 → GPU1/GPU2 → EXT1。AT1 是 P2 后的可选 provider 分支；PyTorch 参考测试与启用 ATen provider 是不同要求。OP1 和 OP2 可在 RT 头接口稳定后并行；用同一个配套 RT 安装前缀集成，不能让生成器针对尚未合并的类型发布到用户构建。

每个 PR 附上“上游公共目标、Infini 适配差异、支持矩阵、错误路径、验证记录”。跨 repo 依赖在 PR 描述和 CI 中固定 SHA/安装包，不只写“使用最新 InfiniRT”。不手改 `generated/`，不顺便清理遗留 `src/tensor.cc` 或重构无关 dense 算子。

## 每阶段的算法与资源策略

### P1 / P2：正确性基线

- COO ToDense 先清零输出，再把每个存储项累加到坐标；CPU 能自然定义重复项顺序。CSR ToDense 遍历行区间；非法指针在运算前拒绝。
- COO/CSR × dense 直接遍历稀疏条目累加，复杂度与存储项和输出列数相关；首期不依赖 MKL 等可选库，不生成完整稠密左输入。
- COO coalesce 建立字典序排序和分段求和流程；整型坐标线性化若可能溢出，使用元组比较或经验证的安全宽度，不能无条件 `row*K+col`。
- COO→CSR 在需要时合并后生成 row pointer；CSR→COO 展开行索引。对显式零不额外删项；稠密→稀疏会丢失显式零的原结构，须另行定义。
- sampled_addmm 只计算 input 中已存储的坐标；输出三组件容量已知。先保持无重叠输出契约，不引入隐式 resize。

这些是实现路线，不限定唯一 kernel。P2 要求正确性与资源界限，不预设优于 PyTorch 的性能结论。

### P3：设备支持

NVIDIA 先评估已有 ATen 路径验证语义，再接入原生 provider。供应商稀疏库属于 Ops 的依赖，不加到 RT 通用层。当前规划不固定未经验证的库版本或算法；GPU1 需要核验工具链、index width、dtype、算法确定性、workspace、graph capture 等官方能力。

设备内容校验的同步入口明确付出等待成本；受信结构可以避免重复扫描。若使用 GPU 排序/前缀和，所有临时内存纳入 workspace 查询。首次创建 handle/descriptor 和资源预热放在 capture 之前。

异步 Python 桥接必须证明外部 owner 在 consumer stream 完成前存活。原生 CUDA 调用若借用 PyTorch allocator 的 buffer，需要正确登记使用 stream 或保活到 event 完成。尚无等价机制的后端使用明确同步的受限接口，不把 Python 调用返回视为设备完成。

### P4 / P5：扩展次序

优先 dense→COO/CSR，再按实际消费者选择 CSC 或 block。dense→sparse 首期可只处理二维 scalar values，采用精确定义的 `value != 0` 选择存储项，不隐含 epsilon 剪枝；NaN、正负零和 Inf 要与固定 PyTorch 基线验证。hybrid 中按整个 dense 子块是否全零决定保留，单独增加测试。

batch compressed 不能随意支持每批不同 nnz；需要这种能力时另行设计。block 支持不能仅增加构造工厂，必须同时验证整除、值块顺序、转换和算子支持。

sparse×sparse 的符号/数值两阶段输出、softmax 对隐式元素的特殊语义、values autograd、2:4 packed 表示分别独立评审。它们不通过为已有 op 添加一个枚举就完成。

## 初始能力矩阵

表中是计划目标，不是当前支持状态。D0 冻结的 P2 限制为二维、连续组件、无 batch/hybrid、实数 float32/64；dense operands/out 也必须连续，负 stride 拒绝，所有 value dtype 一致。

| 能力 | CPU P2 | NVIDIA P3 | 其他加速器 |
| --- | --- | --- | --- |
| COO / CSR metadata view | 共享 RT 描述 | 共享 RT 描述 | 仅描述不等于能执行 |
| 完整内容校验 | 目标支持 | 专用设备路径，待实测 | 单独实现/验证 |
| ToDense / SparseMm | COO/CSR；sum | 同范围逐项交付 | 未声明前拒绝 |
| Coalesce / COO↔CSR | 目标支持 | workspace / 动态结果协议 | 未声明前拒绝 |
| sampled_addmm | CSR 固定 pattern | CSR 固定 pattern，待实测 | 未声明前拒绝 |
| index dtype | COO int64；CSR int32/int64 | 逐 provider 核验，不静默窄化 | 按后端矩阵 |
| ATen provider | 可选，slot 8 | 可选，依赖注册及工具链 | 不保证可用 |
| graph capture | 不作为 CPU 交付要求 | 初始关闭；固定结构且单独验收后开放 | 按后端能力 |
| autograd | 不支持 | 不支持 | 不支持 |

## 风险、决策与回退

| 风险 | 推荐决定/行动 | 验收证据 |
| --- | --- | --- |
| 改动 dense 描述符导致 ABI/性能回退 | 新增 sparse 类型；不改 TensorView 布局/hash | RT dense allocation/footprint、installed consumer |
| 生成器擦除类型 | 先完成 typed classification，再导出新签名 | 生成结果编译、Python overload 路由测试 |
| 外部 indices 修改后旧 plan 被复用 | metadata cache 不存结构分析；caller-owned plan 明确结构稳定期 | 同 nnz 不同结构、同地址改写、地址复用回归 |
| out 的 nnz 变化或 ATen 重新分配 | output builder 与显式 Finalize；ATen 必要时组件复制 | 地址/容量/结果槽测试 |
| Python 对象提前释放 | component owner + stream 使用登记/完成事件 | 强制 GC、内存复用、延迟消费测试 |
| Debug assert 被编译掉 | sparse checked entry 的 Release 有效状态契约 | Release 非法输入测试，无越界执行 |
| 类型范围与 index 降位 | 显式 int32/int64 能力，溢出拒绝 | 边界元数据测试，无需超大实际分配 |
| 上游版本与文档不一致 | 固定 tag、记录探针版本、逐组合验证 | API/source/reference 证据表 |
| 性能收益取决于稀疏分布 | 分开测转换/分析/执行；保留用户显式选择 | 分布、密度、宽度和复用次数基准 |

P0 决策已由 D0 冻结：独立 view、先 CPU COO/CSR、checked 状态入口、默认 checked import、caller-owned 动态输出、原生 CPU 不依赖 Torch、初期不缓存结构分析。错误映射、完成状态、特殊值及用例对应见 [D0 契约](06-d0-contract.md)。具体优先硬件和 block 大小由后续真实消费者决定，不阻塞当前契约交付。

如果某后端未达出口，只保持该组合 unsupported，不自动转成 dense 或切换设备。回滚可撤回对应 provider/注册与实验性入口，已稳定的 dense 调用链继续工作。版本发布时把两库配套版本、受支持组合和实验性 API 标记写入发布说明。
