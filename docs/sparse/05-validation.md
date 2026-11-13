# 验证与验收方案

以下是后续实现的测试设计，不是运行结果。D0 已完成 [全部 38 项用例的契约走查](06-d0-contract.md#验证样例逐项走查)，并冻结状态、输出和生命周期预期；该记录不替代 RT3/OP4–GPU2 的执行证据。D0 未构建 InfiniOps/InfiniRT，未执行新的稀疏算子。

## 验证原则

同时验证 **结构、数值、内存边界和执行语义**。只比较 `to_dense()` 会漏掉丢失显式零、错误 nnz、乱序、重复及 mask 变化。支持矩阵中的 case 必须执行，不能被通用 Torch 测试的 broad skip 掩盖；不支持组合应有明确拒绝测试。

参考环境固定 PyTorch 版本、编译选项和 backend；记录 `torch.__version__`、完整 repo SHA、SDK/驱动、设备、dtype、index dtype、layout、implementation index。PyTorch 在某组合上不支持时，不将它的异常当作 Infini 通过；使用独立解析结果或标记该组合尚无参考证据。

## 结构与数值用例

| ID | 输入/触发 | 预期与要防止的问题 |
| --- | --- | --- |
| S01 | 文档 01 的重复 COO 例子 | coalesce 得 nnz=3，保留零条目；dense 仅两处非零 |
| S02 | 相同 COO 坐标 values=[1,-1] | 合并值为零但条目仍存在；不能误当空 tensor |
| S03 | COO 乱序、重复、错误 coalesced=true | checked 校验拒绝假声明；普通未 coalesced 输入可按语义处理 |
| S04 | COO indices int32 | RT 借用入口明确拒绝；若上层转换需声明分配，不假装零拷贝 |
| S05 | CSR crow=[0,2,2,3]，col=[1,3,0]，shape=(3,4) | 空中间行正确，尾部空列保留，to_dense 和 SpMM 正确 |
| S06 | CSR crow 首项非零、非单调、末项不等于 nnz | 校验拒绝，运算 kernel 不执行 |
| S07 | CSR 单行 col=[1,0] 或 [0,0] | 完整 invariant 检查拒绝乱序和重复，不自动改结构 |
| S08 | COO / CSR index 为负或等于维度上界 | 明确越界错误，不读取无效 values/operand |
| S09 | nnz=0，shape=(0,K)、(M,0)、(M,K) | 合法空输出；CSR crow 仍为 M+1，元素全零；不被 skip |
| S10 | 单存储项、全显式零、非方阵、尾部空行/列 | 保持逻辑 shape 和有效结构，不靠 max(index) 重建尺寸 |
| S11 | 组件 rank/shape/dtype 不匹配，混合 CPU/GPU 或 device index | metadata 检查在读 buffer 前拒绝 |
| S12 | 非连续组件、负 stride、重叠 out、错误 workspace/capacity | 按首期限制拒绝；不能静默复制或越界 |
| S13 | 极大 shape、nnz、bytes 乘法、int32 上限边界 | 元数据溢出与窄化错误；测试无需分配巨大张量 |
| S14 | `ToDense(COO)` 含重复 | 相加而非最后一次覆盖；输入结构不修改 |
| S15 | `COO→CSR→COO` | coalesced 规范结果结构相等，显式零保留 |
| S16 | COO/CSR SpMM，输出列数 0/1/多列，空内积维 | dense out 形状和值正确；reduce 非 sum 明确拒绝 |
| S17 | sampled_addmm 显式零样例：crow=[0,2]、col=[0,1]、values=[0,2]，alpha=beta=1 | mat1=[[2,3]]、mat2=[[4,5],[6,7]] 时 values=[26,33]，pattern 不变 |
| S18 | sampled_addmm alpha/beta 为 0、负数；values/operands 含 NaN/Inf；空内积 | 按 [D0 固定 CPU 特殊值规则](06-d0-contract.md#数值与特殊值冻结) 和解析样例验证；beta=0 不忽略 input NaN/Inf，GPU 单独验证 |
| S19 | 不同 layout/dtype/rank/设备请求 | 能力诊断准确，无 dense fallback，无数据拷贝到其他设备 |
| S20 | 未来 dense→sparse 含 ±0、NaN/Inf，或 hybrid 全零块 | 明确筛选语义；dense roundtrip 不等于恢复原 sparse pattern |
| S21 | 未来 BSR/BSC block 2×3、维度不整除，batch 的 N 不一致 | 按 block/batch 不变量检查，nnz 计块，不计标量 |

S01–S19 属于 P2/P3 按设备适用的核心矩阵，S20–S21 随 P4 特性交付。

浮点数值比较使用与当前项目一致的 dtype 容差，并记录累加精度和确定性。CPU 双精度小矩阵可用解析结果；GPU 归约次序可能不同，不承诺位级相同。结构比较应精确；若某算子允许不同排列，先按该算子约定规范化后再比，不无条件容忍任意结构。

## 动态输出、缓存与资源用例

| ID | 场景 | 预期 |
| --- | --- | --- |
| O01 | coalesce 容量 10、实际 nnz=6 | COO indices 最终连续 `[2,6]`，第 2 行偏移正确；canary 不变 |
| O02 | capacity 恰好足够/少一项/为零 | 不越界；容量不足返回失败，Finalize 不发布结果 |
| O03 | GPU result slot pending、成功、失败 | 提交成功不冒充完成；回读和构造只在已完成成功后发生 |
| O04 | Prepare 后修改 indices 或 dense count-first 的 values | 契约禁止；可检测的 wrapper token 变化被拒绝，裸指针模式明确由调用方负责 |
| O05 | Python 动态结果完成后删除 builder、原组件变量 | 返回 tensor 的 owner 链仍有效；无局部 for_blob 悬空 |
| C01 | 相同 shape/nnz，先 COO 后 CSR | 不复用错误 layout provider 或 metadata |
| C02 | 相同 shape/nnz/type，不同 indices 内容/地址 | 结果随结构变化，旧分析不被误用 |
| C03 | 原地址原地改 indices、分配器回收后地址重用 | caller-owned plan 需失效/重建；禁用结构缓存路径始终正确 |
| C04 | 仅更新 values | metadata 可复用；纯符号 plan 在生命周期契约内可复用 |
| R01 | 非默认 stream，producer/consumer 不同 | event 建立依赖，结果正确；不出现误用默认 stream |
| R02 | enqueue 后立即释放 Python 引用、强制 GC、制造 allocator 复用压力 | 所有索引与 values 保活到完成，不能只保活 values |
| R03 | 多个 consumer stream、workspace 重用或提前 FreeAsync | join 后才复用/释放；无竞争 |
| R04 | 部分组件分配失败 | 上层 builder/owner 回收已分配资源；view 不 double free |
| R05 | 在不同 thread-local backend/device 上析构 owner | 恢复正确上下文，释放正确设备资源，调用线程状态恢复 |
| R06 | capture 中结构固定、资源预热 vs 临时扩容/host nnz 回读 | 仅声明可 capture 的组合通过；其余在捕获前拒绝 |
| R07 | 输出 ATen wrapper 内部换存储 | 不丢失结果，不让 caller buffer 被绕过；地址/复制策略有测试 |
| R08 | Python 输入需要梯度 | 首期明确拒绝并提示 detach；不能悄悄断图 |

CPU 使用 ASan/UBSan 检查 buffer 边界及无效生命周期；GPU 使用可用的设备内存检查工具与重复压力用例。原生 buffer 的任意外部改写无法由 view 全部检测，测试要验证实现承诺的检查范围，不伪造数据竞争下的安全保证。

## 生成、安装和 dense 回归

| 层 | 现有入口与拟新增测试 | 交付条件 |
| --- | --- | --- |
| RT metadata | `tests/test_core.cc`、`test_tensor_view_allocations.cc`；新增 sparse view 测试 | dense rank 0–8 内联及高 rank 回退不退化，sparse 不隐式转 dense |
| RT 安装包 | `tests/install_consumer_smoke.cc`、`tests/install_consumer/` | 只用安装前缀和 `<infini/rt.h>` 编译运行 |
| Ops codegen | 三个 `tests/test_generate_*` 的 wrapper / torch / public header 测试 | sparse 类型不擦除，optional/list 正确，旧 dense 接口稳定 |
| Ops Python | 新增专用 sparse 测试模块，现有 dense smoke/full suite | layout routing、非 coalesced 导入、动态输出 owner |
| Ops C++ | `tests/test_cpp_api.py` 及 sparse consumer | `<infini/ops.h>` 导出并链接正确，配套 RT 版本明确 |
| provider | 原生 CPU/NVIDIA 与可选 ATen slot 8 | unsupported 清楚；实际支持的 case 不被通用 skip 隐藏 |

共享生成器、dispatcher、Python converter 的改动属于高风险基础设施变更。应按仓库贡献规范运行受影响平台的完整构建与测试；smoke 是早期检查，不能作为这些改动最终验收的唯一证据。

## 后续执行命令模板

下列命令供实现后使用，本次没有执行。先按两个仓库的 build 文档准备依赖。使用独立输出目录和配套安装前缀；不要用当前稠密构建的通过结果代替新稀疏实现的验收。

```bash
cmake -S /workspace/InfiniRT -B /tmp/infini-sparse-rt-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/tmp/infini-sparse-rt-prefix \
  -DWITH_CPU=ON -DINFINI_RT_BUILD_TESTING=ON
cmake --build /tmp/infini-sparse-rt-cpu -j
ctest --test-dir /tmp/infini-sparse-rt-cpu --output-on-failure
cmake --install /tmp/infini-sparse-rt-cpu

cd /workspace/InfiniOps
python -m pip install '.[dev]' --no-build-isolation --no-deps \
  -C cmake.define.INFINI_RT_ROOT=/tmp/infini-sparse-rt-prefix \
  -C cmake.define.WITH_CPU=ON \
  -C cmake.define.WITH_TORCH=OFF
python -m pytest tests/test_generate_wrappers.py \
  tests/test_generate_public_operator_header.py \
  tests/test_generate_torch_ops.py tests/test_cpp_api.py -q
```

专用 sparse 测试文件由 OP4–OP6 创建后加入运行清单及 CI；此处不引用尚不存在的测试作为可运行证据。再运行完整 `python -m pytest tests`。ATen/NVIDIA 各使用独立配置，设置 WITH_TORCH/WITH_NVIDIA，验证 provider 选择并记录工具链。纯原生构建不启用 Torch provider，但 Python 对照测试可依赖测试环境的 torch 包。

## 性能实验与通过标准

性能记录至少分为：描述符构造/复制/哈希、完整校验、排序/格式转换、供应商分析准备、单次 kernel、重复执行、端到端桥接和内存峰值。GPU 用正确的 stream event 计时并预热；动态 nnz 回读及临时组件拷贝计入端到端时间。

输入覆盖均匀随机、极不均匀行长度、带状、block、空行很多、极小/极大 nnz；扫描多个密度、矩阵尺寸、输出列宽和结构复用次数。比较固定 PyTorch sparse 参考、原生实现及相同数据的 dense 基线；不据单一随机矩阵推断稀疏总是更快。

内存核算包括所有组件和 workspace。无 batch 的 COO 主要存储为 `N*S*sizeof(index)+N*prod(D)*sizeof(value)`；CSR 为 `(M+1+N)*sizeof(index)+N*prod(D)*sizeof(value)`。实际分配、对齐和临时空间另记，不能把 `numel*element_size` 当作稀疏占用。

正确性、无越界、Release 拒绝非法输入、公共安装消费和 dense 回归是硬门槛。P2 不设未经测量的“比 PyTorch 快 X%”目标；先形成可复现基线。对 dense footprint 和 host overhead 的允许回归阈值在测量环境稳定后冻结，不能以噪声结果随意宣布通过。

## 交付证据模板

每个阶段提交一份结果表：测试 ID → layout/设备/dtype/index dtype → provider → 命令/环境 → 通过、明确拒绝或未测试 → 关联日志。记录已知限制和失败案例。只有支持矩阵内全部必需 case 通过、没有未解释 skip，并且文档/API/生成产物一致，才把该能力从“计划”改成“支持”。
