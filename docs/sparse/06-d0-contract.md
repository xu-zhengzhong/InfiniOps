# D0：冻结的稀疏接口契约

状态：**D0 / P0 契约已冻结；RT1 源码实现与 RT2 安装导出已完成，RT3 正式元数据测试集与所有 Ops 后续任务尚未完成**。冻结日期为 2026-11-12。本页把 [目标设计](03-design.md) 中的建议落实为 RT1–EXT1 的实现约束；具体 API 与导出边界见 [RT1 实现记录](07-rt1-implementation.md) 和 [RT2 实现记录](08-rt2-implementation.md)。RT 类型已安装导出，不表示 Ops provider 已注册。与早期方案的推荐措辞有差异时，以本页和两库 core-types / compatibility 中的 D0 条款为准。

本页固定类型职责、可区分的状态、参数语义和执行协议。表中的名字是契约标识，不是可编译声明；具体 C++ 重载拼写、枚举数值和对象二进制布局由实现任务导出并验证，不构成 D0 的 ABI 承诺。实现不得借此合并错误类别或改变所有权、同步和发布规则。

## 决议与实现边界

| 事项 | D0 决议 | 承接任务 |
| --- | --- | --- |
| 表示 | RT 新增独立 `TensorLayout`、`SparseTensorView`、`ValidationResult`；Ops 使用同一 sparse view。现有 `TensorView` / `Tensor` 保持 dense 语义、对象布局及 hash/equality | RT1–RT3、OP1 |
| 首期表示 | 显式二维 shape、连续组件、非 batch/hybrid COO/CSR；COO int64，CSR 两索引组件同为 int32 或 int64 | RT1、RT3 |
| 首期计算 | 原生 CPU float32/float64，所有 value operands/out dtype 一致；dense operands/out 也必须连续，负 stride 拒绝；不依赖 Torch provider | OP4–OP6 |
| 外部输入 | RT 工厂仅校验 metadata；Ops checked import / checked entry 默认完整校验内容。可信来源只能省略重复内容扫描 | OP1、OP3、OP4 |
| 错误 | 外部稀疏错误在 Debug/Release 均返回可检查状态；assert 仅检查内部不变量；Python 在边界翻译异常 | RT1、OP1–OP4 |
| 输出 | 调用者拥有缓冲；固定 shape/pattern 使用 out，动态 nnz 使用 Prepare/Execute/Finalize 与独立结果槽 | OP4–OP6 |
| 执行 | P1/P2 CPU 同步完成；P3 才交付设备异步校验/执行。首期 sparse capture 和在线 tuning 关闭 | OP1、GPU1、GPU2 |
| 缓存与生命周期 | metadata cache 不存结构分析或旧 data 指针；调用者保活所有组件、workspace、结果槽至全部消费者完成 | OP1、OP3、GPU2 |
| 后续布局 | CSC/BSR/BSC、batch/hybrid、dense→sparse 在 P4；autograd、sparse×sparse、2:4 在独立后续范围 | EXT1 / P5 |

这是对 Ops [贡献规范](../../CONTRIBUTING.md) 中 assert 错误处理规则的**稀疏公共边界专用决议**，不改写现有 dense 错误体系。RT 当前贡献规范已经要求外部运行错误返回状态。现有 `runtime::Error` 和 void `Operator::Call` 不被宣称已实现本页协议。

## RT 类型与构造结果

`SparseTensorView` 可复制，拥有逻辑 shape 和组件视图的元数据，借用组件 data；不借用临时 `TensorView::shape()` 的返回值，不延长 buffer 生命周期。工厂不分配组件数据、不读内容、不启动设备 kernel、不同步 stream。元数据自身的主机存储失败也必须报告，不能发布半构造视图。

| 属性 | 冻结规则 |
| --- | --- |
| layout | `TensorLayout` 区分 COO/CSR/CSC/BSR/BSC；初期只允许 COO/CSR 工厂成功 |
| 逻辑维度 | `shape=[M,K]`，`sparse_dim=2`、`dense_dim=batch_dim=0`；不从 max(index) 推断尺寸 |
| COO | `indices=[2,N]`、int64；`values=[N]`；`nnz=N` 包括重复和显式零 |
| CSR | `crow_indices=[M+1]`、`col_indices=[N]`、`values=[N]`；两个索引 dtype 匹配；`nnz=N` 包括显式零 |
| dtype/device | value dtype 与 index dtype 分开查询；所有组件完整 `(device.type, device.index)` 相同；RT 能描述某个 value dtype 不等于 Ops 支持计算 |
| 宽度限制 | 每个逻辑 extent 与 N 不大于选定有符号索引类型最大值；COO 为 INT64_MAX，CSR 为 INT32_MAX 或 INT64_MAX。这是明确的首期表示限制，不自动窄化 |
| 算术 | `M+1`、逻辑 numel、组件元素数/bytes、stride 和地址区间端点分别检查可表示性；不把逻辑 numel 乘 value size 当成 sparse 占用 |
| 空维度 | 任一逻辑维度为零时 N 必须为零；零长度组件可为 null，不能解引用；CSR `[0,K]` 仍须一个 crow 元素，`[M,0]` 仍须 M+1 个 |
| 对齐 | 非空 typed 组件在任何内容访问前检查元素存储类型的自然对齐；不能先解引用 indices 再报告未对齐。provider 更严格的输出/workspace/结果槽要求由 Prepare 报告 |
| 组件访问 | COO 的 `indices()/values()` 要求声明 coalesced；raw 组件入口允许未合并输入。非法 layout/state 访问返回 checked failure。CSR 有 crow/col/values 访问器 |

COO 工厂保留调用者的 coalesced 声明，包括空 COO；省略该属性时默认 false，不自动排序或提升标记。`coalesced=true` 只是一项待验证声明，不是可信证明。Ops 完整校验要求坐标有界；声明 coalesced 时还须字典序严格递增、无重复。未 coalesced COO 可乱序、可重复。

CSR 完整校验要求 crow 首项 0、末项 N、每项在 `[0,N]` 内且非递减，col 有界且每行严格递增。必须先证明 row pointer 范围安全，再按其读取 col/values；不能让校验器本身越界。空 CSR 的所有 crow 元素也须为 0。RT 只凭 metadata 不能证明这些内容条件。

整体 sparse 对象不提供 `data()`、`strides()`、`IsContiguous()`、dense 指针切片或无条件 `T()`。shape、layout、维度分解、nnz、dtype/device、组件元数据和 COO 规范化声明可用于 metadata key；它们不能证明结构内容相同。

## 结果与 Python 异常映射

`ValidationResult` 的逻辑字段为 `{code, component, position?}`：`component` 标识参数/组件（例如 `input.crow_indices`）；`position` 是可用时的出错元素位置，不保证并行校验报告最小位置。结果不借用临时错误字符串。成功的 `code=Success` 不带错误位置。RT 返回下表的 metadata 子集，Ops 保留 RT 的错误类别，并增加内容、能力及执行类别；可采用分层枚举，但跨边界映射必须无损。

| code | 触发 / 责任边界 | Python 映射 |
| --- | --- | --- |
| `Success` | 当前阶段成功；RT 工厂成功仅证明 metadata | 无异常 |
| `InvalidMetadata` | 组件 rank/shape/数量关系错误、非法维度值、操作数或固定输出的 shape/value dtype 不匹配、输入/输出 typed 组件或结果槽指针对齐错误 | `ValueError` |
| `UnsupportedLayout` | 尚未实现的 sparse layout | `NotImplementedError` |
| `UnsupportedCapability` | 有效但超出首期的 rank、batch/hybrid、非连续组件、负 stride、value dtype、设备、reduce、capture 等组合 | `NotImplementedError` |
| `InvalidIndexType` | COO 非 int64、CSR 非匹配 int32/int64 | `TypeError` |
| `DeviceMismatch` | 组件或操作数完整 device 不一致 | `ValueError` |
| `NullData` | 声明非零有效/容量 bytes 的组件指针为空 | `ValueError` |
| `IntegerOverflow` | 元数据算术、地址端点溢出，或 extent/N 超出 index width | `OverflowError` |
| `IndexOutOfBounds` | COO 坐标/CSR col 负值或越界 | `IndexError` |
| `InvalidStructure` | CSR crow 不合法、行内 col 乱序或重复 | `ValueError` |
| `InvalidCoalescedClaim` | 声称 coalesced 的 COO 实际乱序或重复 | `ValueError` |
| `InsufficientCapacity` | 输出组件声明容量不足，包括动态执行发现实际 N 超容量 | `ValueError` |
| `InvalidWorkspace` | workspace 缺失、容量不足、设备或对齐错误 | `ValueError` |
| `Aliasing` | 禁止的输入/输出/workspace/结果槽区间重叠 | `ValueError` |
| `NoProvider` | 此构建没有可执行该请求的 provider | `NotImplementedError` |
| `UnsupportedImplementation` | 显式 implementation index 不存在或不支持请求；不自动改选 | `ValueError` |
| `InvalidState` | 错误组件访问、未 Prepare、复用仍在执行的结果槽、失败结果再次发布等 | `RuntimeError` |
| `StaleStructure` | 能可靠维护版本的 wrapper 检测到 Prepare 后结构/计数相关数据变化 | `RuntimeError` |
| `AutogradUnsupported` | Python 任一输入 requires_grad；提示调用者主动 detach | `NotImplementedError` |
| `AllocationFailure` | 元数据、builder 或 provider 的允许分配失败 | `MemoryError` |
| `ExecutionFailure` | 提交/执行/同步失败或 ATen 框架异常，无法归类为以上错误 | `RuntimeError` |

若同时存在多项错误，只承诺报告安全检查顺序遇到的一项，不固定跨 provider 的诊断优先级。检查顺序必须保证不先读取无效 buffer。Release 不得将这些路径变成 assert、静默成功或越界执行。

动态结果另有 `Pending / Success / Failed` 完成状态，**Pending 不是校验失败或执行成功**。`Failed` 携带上述错误结果，`actual_nnz` 只在完成 `Success` 时有效。设备结果槽使用可复制的状态/位置数据；Python 异常在主机确认完成后构造，不在设备上创建字符串或异常。

## Checked entry 与可信来源

契约级同步入口为 `CheckedCall(inputs, attributes, out, resources) -> ValidationResult`；它不改变已有 dense `Call`。动态操作使用下一节的三阶段入口。C++ 调用方必须检查结果；Python 转换在读取 dense 的 data/stride 前先检查 layout，不依赖两个 `py::object` overload 自动回退。

检查按依赖分层进行：先验证参数 metadata、设备、支持组合和显式 implementation；再验证资源的声明范围、对齐及 alias；需要数据内容的校验只在安全资源和执行依赖已建立后进行。校验失败时不运行计算 kernel；没有成功 provider 时不进入旧的抛异常选择分支。ATen 适配边界局部捕获框架异常并转换状态，不把它们传播给 C++ 调用者。

默认外部 RT view 每次进入 checked 计算都须完整校验；Python checked import 完成的验证可在同一次受控调用内复用。可信模式只允许已验证且结构未改的输入或已成功完成的内部算子输出，仍检查 metadata、设备、能力、输出和资源。裸指针、地址相同、metadata hash 相同、coalesced 标记均不能单独启用可信模式。

P1/P2 的 CPU 入口在返回时完成。P3 的同步 checked 入口明确允许等待校验/执行完成；异步入口返回提交结果、完成句柄和 caller-owned 结果槽，只有完成成功才可消费输出。不能先异步提交校验，再无条件运行可能越界的计算；必须等待其结果或使用经验证的设备端条件执行。尚无该能力的后端返回 unsupported，不能 host 解引用设备指针。

## 输出、容量与发布

固定输出不 resize、不更换 data：ToDense 输出 shape 与 input 相同；SparseMm 的 `[M,K] × [K,P]` 输出 `[M,P]`；sampled_addmm 的 CSR out shape、N 与 input 一致。固定 pattern CSR out 是 **out-only 描述**，只检查 metadata、可写容量和 alias，不能扫描未初始化 indices。执行复制 input pattern 并写 values；完成前不得把 out 当成普通输入。CSR→COO 的 N 已知，也使用精确大小的 out-only 描述。

动态输出的逻辑类型 `SparseOutputBuffer` 记录 layout、目标 shape、dtype/index dtype/device、各组件 `{data, capacity_bytes}`，不持有伪装为有效 nnz 的 capacity-shaped view。`SparseResultMetadata` 是 caller-owned 的完成状态、失败详情及实际 N；它与缓冲和 plan 的生命周期分开。

1. **Prepare**：校验输入与能力，返回组件容量上界、各组件/workspace/结果槽大小及对齐要求，绑定输入元数据、结构稳定期、属性与 implementation。coalesce、COO→CSR 的 N 上界为 input N。P2 必须提供上界模式；count-first 是可选扩展，未实现应明确拒绝。Prepare 不改变输入或发布输出。
2. **Execute**：绑定 caller-owned 缓冲与结果槽，检查容量/alias 后初始化槽为 Pending，提交返回值只表示是否成功提交。若按 Prepare 的上界分配，执行应容纳所有合法结果；容量小于上界的调用在初期直接拒绝，不冒险以“通常够用”继续。成功 count-first 才可使用其精确容量。运行时仍须防御容量溢出，不发生越界写入。执行完成将槽置为 Success 或 Failed。
3. **Finalize**：确认执行完成并检查结果槽。Pending 返回未就绪且不产生 view；调用者可显式等待，Python 的同步 finalize 明确等待并支付回读成本。Failed 返回原错误且不产生 view。Success 检查 `0 <= actual_nnz <= capacity` 及最终组件布局后发布新 view；不复制 values。成功 finalize 可重复取得相同视图，但在下次 Execute 前须结束旧结果的使用；仍 Pending 的槽禁止复用。

容量/metadata 检查在提交前失败不写输出组件；执行或后端失败后缓冲可能部分写入，不承诺事务回滚，固定和动态输出均不可继续作为成功结果使用。结果槽与完成句柄须一致，设备故障/等待失败必须报告 ExecutionFailure，不能永久把故障伪装为 Pending。

COO 必须由 Execute 按实际 N 打包成连续 `[2,N]`。例如 capacity=10、N=6：第二行从第 6 个 index 开始，不能仍从第 10 个开始；打包和临时空间计入 workspace。CSR col/values 可使用有效前缀，crow 始终为 M+1。Finalize 只描述已经写好的布局，不用缩 shape 隐藏错误 stride。

Prepare 到 Execute 完成期间输入不得改变。count-first 还绑定影响计数的 values；可跟踪 owner 的 wrapper 发现变化返回 StaleStructure，裸视图无法检测任意外部改写，由调用者履行前置条件。首次动态流程不支持 graph capture。

Python output builder 保活组件和结果槽，Finalize 后返回新的 `torch.sparse_*` 对象，其组件必须关联原始存储 owner。删除 builder 和原组件变量后结果仍须有效。传入的 sparse out 不被隐式替换元数据；ATen out 若不能证明地址稳定，则用临时结果、校验容量后逐组件复制，并把开销计入资源查询和性能记录。

## Alias、owner 与缓存

检查按同一设备地址空间的声明字节半开区间进行：输入按有效 bytes，动态输出按全部 capacity bytes，固定 out 按其写入范围，workspace 和结果槽按声明使用范围。端点运算溢出返回 IntegerOverflow；零长度区间不重叠。所有写入区间相互不重叠，也不与任何输入区间重叠，包括跨组件、输出与结果槽、workspace 与结果槽。只读输入彼此允许 alias。输入/输出 typed 组件须满足元素自然对齐，结果槽须满足结果元数据类型及 Prepare 报告的对齐；非空范围在读写前检查。组件或结果槽未对齐返回 InvalidMetadata，workspace 未对齐返回 InvalidWorkspace；零长度且不访问的组件免除地址对齐要求。

这些检查不证明任意裸指针的真实分配长度或可写性；调用者保证声明真实、地址有效。所有 sparse 算子初期均不允许 in-place，包括已经 coalesced 的 Coalesce；这相对 PyTorch 可能返回自身是显式 out 适配。

所有 index/value/dense operand/out/workspace/结果槽 owner 保活至最后一个 consumer 完成。producer event 后才能消费，多个 consumer 完成后 join 才能复用或 FreeAsync。CPU/缺少事件能力的后端采用明确同步路径。Python 持有全部组件强引用，NVIDIA 还须经验证的 stream 使用登记或 event 完成队列；只保活到 Python 函数返回不够。

builder 分配部分失败时回收已分配组件，view 不负责释放。owner 析构使用分配时的 backend/device 上下文，并恢复析构线程原状态。跨设备复制逐组件建立依赖并重新检查 metadata，不隐式换设备。

metadata cache 包含布局、几何/维度分解、N、dtype/index dtype、device、组件 shape/strides、规范化声明及算法属性，但不保存 data 指针或结构分析。首期结构分析每次重建；后续 caller-owned plan 必须绑定 owner/结构身份和稳定期。indices、shape、layout、N、block shape 或 coalesced 状态变更使校验证明与结构 plan 失效；地址回收复用也不能复用旧 plan。仅 values 更新可复用纯符号分析，但仍满足流顺序。稀疏在线 tuning 初期禁用。

## 数值与特殊值冻结

Coalesce 对相同坐标求和、按字典序输出唯一坐标，保留合并得到的零。COO→CSR 先达到同样规范形式，首期输出索引固定 int64，声明 int32 输出返回 UnsupportedCapability；不把 out dtype 当成已实现的窄化转换请求。CSR→COO 输出 coalesced、int64 索引 COO，输入 int32 时在这个显式格式转换操作中扩宽。Coalesce 保持 COO int64，sampled_addmm 保持 input 的 CSR index dtype；所有这些操作保持 value dtype。ToDense 和 SparseMm 接受未 coalesced COO 并累加重复项；SparseMm 只接受 `reduce=sum`。空维度/零输出列为合法请求，不能被 skip。sampled_addmm 按存储坐标计算，S17 的结果必须为 `[26,33]`，不按非零 value 重建 pattern。

S18 的 CPU 基线固定为 v2.9.0：外层先复制 input；非空乘积逐项计算 `alpha_cast * dot + beta_cast * input_value`，scalar 转为 value dtype；空 operand 或 N=0 时走 values 乘 beta 的分支。没有 beta=0 或 alpha=0 的数值忽略捷径。因此正常非空乘积中 beta=0、input value=NaN/Inf 时仍传播 NaN；alpha=0、dot=NaN/Inf 时也传播 NaN。内积维 K=0 时只按 beta 缩放 input，不额外计算 alpha 乘空点积。依据为固定版本 [CPU 外层](https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/sparse/SparseBlas.cpp) 和 [CPU kernel](https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/cpu/SampledAddmmKernel.cpp)。

后续测试至少含以下解析样例（每例 pattern 都保留）：单存储项 input=NaN、mat1=mat2=1、alpha=1/beta=0 得 NaN；input=2、mat1=Inf、mat2=1、alpha=0/beta=1 得 NaN；input=2、点积=3、alpha=-2/beta=-1 得 -8；K=0、input=2、alpha=NaN/beta=3 得 6。这是源码走查得出的 CPU 契约，D0 未运行固定版本数值测试。P3 provider 必须验证并满足已公开的规则，不能将不同后端行为混称同一保证；必要时适配或明确拒绝相应组合。浮点归约不承诺位级相同，NaN/Inf 分类和结构须单独比较。

## 验证样例逐项走查

以下每行结论均为**契约已覆盖**，不是实现测试通过。用例定义来自 [05 验证方案](05-validation.md)，后续任务仍须提供运行证据。

| ID | D0 冻结的预期 / 拒绝路径 | 实现验证 |
| --- | --- | --- |
| S01 | 重复 COO 合并 N=3，保留零条目；dense 两处非零 | OP5 |
| S02 | `[1,-1]` 同坐标合并为一个显式零，不为空 | OP5 |
| S03 | 假 coalesced 声明 InvalidCoalescedClaim；未合并输入可接受 | OP4 |
| S04 | COO int32 返回 InvalidIndexType，无隐式转换 | RT3 |
| S05 | CSR 空中间行、尾部空列保留 | OP4 |
| S06 | crow 非零起点/非单调/错误末项 InvalidStructure，compute 不执行 | OP4 |
| S07 | CSR 行内乱序/重复 InvalidStructure，不自动修复 | OP4 |
| S08 | 负或越界坐标 IndexOutOfBounds，验证器及 compute 均不越界 | OP4 |
| S09 | 空 shape/N 合法；crow 长 M+1 且内容全零；不跳过测试 | RT3、OP4 |
| S10 | 单条目、显式零、非方阵和尾部空行列保持显式 shape | RT3、OP4–OP6 |
| S11 | 组件几何 InvalidMetadata、index dtype InvalidIndexType、设备 DeviceMismatch | RT3、OP4 |
| S12 | 非连续/负 stride UnsupportedCapability；alias/workspace/capacity 各有独立失败 | RT3、OP4、OP5 |
| S13 | shape、bytes、端点或索引宽度溢出 IntegerOverflow；无大分配前置 | RT3、OP4 |
| S14 | ToDense 重复累加，输入不变 | OP4 |
| S15 | COO↔CSR 得规范结构，显式零不丢失 | OP5 |
| S16 | SpMM 零/一/多输出列、空内积合法；非 sum UnsupportedCapability | OP4 |
| S17 | sampled_addmm values=[26,33]，两项 pattern 不变 | OP6 |
| S18 | 按上一节 CPU 分支及解析例检验零/负 scalar、NaN/Inf | OP6；GPU1 单独验证 |
| S19 | 区分 UnsupportedLayout / UnsupportedCapability / NoProvider，无 fallback | OP1、OP3、OP4 |
| S20 | P4 延期；首期 dense→sparse/hybrid 拒绝。未来筛选 `value != 0`，±0 不存，NaN/Inf 存 | EXT1 |
| S21 | P4 延期；首期 block/batch 拒绝。未来块整除、每批等 N、N 计块 | EXT1 |
| O01 | capacity=10/N=6 的 COO 连续 `[2,6]`，canary 不变 | OP5 |
| O02 | 精确/上界容量足够才执行；少一项/错误零容量拒绝，失败不发布 | OP5 |
| O03 | 提交成功后仍可 Pending；完成成功才发布，失败可诊断 | GPU1、GPU2 |
| O04 | Prepare 后数据变化禁止；可靠 token 变化 StaleStructure，裸指针由 caller 保证 | OP5、GPU2 |
| O05 | Finalize 结果 owner 链独立于 builder 变量存活 | OP5 |
| C01 | COO/CSR metadata key 不混用 | OP1 |
| C02 | 相同几何不同 indices 不复用结构分析 | OP1、OP4 |
| C03 | 原地改索引/地址回收复用失效，不能只按地址缓存 | OP1、GPU2 |
| C04 | values 更新在顺序保证下允许复用纯符号 plan | OP1、GPU2 |
| R01 | producer event → consumer wait，使用正确 stream | GPU2 |
| R02 | enqueue 后强制 GC 时所有组件保活到完成 | OP3、GPU2 |
| R03 | 多 consumer join 后才能复用 workspace/释放资源 | GPU2 |
| R04 | builder 部分失败回收已分配资源，view 不 double free | OP5 |
| R05 | 析构恢复分配 backend/device，随后恢复线程上下文 | OP3、GPU2 |
| R06 | 首期 capture 拒绝；后续只有单独验收的固定资源 provider 可开放 | GPU2 |
| R07 | ATen 替换存储时临时结果按容量复制回 caller buffer | AT1 |
| R08 | requires_grad 返回 AutogradUnsupported，要求显式 detach | OP3 |

生成/安装/dense 回归的六个出口也保留：RT metadata 与 dense footprint（RT3）、RT installed consumer（RT2/RT3）、Ops 三类生成器及 optional/list 保真（OP2）、Python layout/owner（OP3–OP5）、Ops installed C++ consumer（OP2/OP4）、逐 provider 能力及无 broad skip（OP4–GPU2）。后续性能记录分开计算校验、转换、分析、执行和桥接成本，不预设加速比；D0 不以 dense smoke 的通过替代 sparse 运行证据。

## D0 交付记录

- 更新两库 `docs/api/core-types.md`、`docs/compatibility.md`，以及 Ops `docs/operator-api-alignment.md`；本页是完整决议和 38 项用例的走查记录。
- D0 核查时 Ops HEAD 为 `45fc82f6acb71f1eaba0cfa05f581bf68bd94616`，RT HEAD 为 `cd1d48258f0a6b8d9f66111d5d6cb1664205e78d`；这是文档工作基线，不是已实现 sparse 的配套安装版本。
- 成功补读固定 v2.9.0 的 compressed validator、SparseBlasImpl、SampledAddmmKernel 及 sampled_addmm 外层；证据补充见 [01 参考](01-pytorch-reference.md#d0-固定版本复核)。没有用本机其他版本探针冒充固定版本测试。
- 文档验证：两库 `git diff --check` 通过；13 份修改/新增 Markdown 的相对链接及锚点、代码围栏和末尾换行检查通过；S01–S21、O01–O05、C01–C04、R01–R08 共 38 项无遗漏、无重复。未构建 Doxygen HTML（环境无 `doxygen`）。D0 不构建新 API，不执行尚不存在的 sparse 测试，不标记 P1/P2 已完成。
