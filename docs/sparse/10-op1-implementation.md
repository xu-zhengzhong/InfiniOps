# OP1：稀疏参数与 checked 分派基础设施

状态：**OP1 已完成**。本任务将 RT 的 COO/CSR 描述接入 Ops 参数、元数据检查、缓存键和 provider 选择，并定义可检查的执行状态入口。没有新增真实稀疏计算 provider、Python sparse 桥接或动态输出；OP2 及后续任务仍待实现，P1/P2 尚未完成。

## 配套依赖与文件

本次 Ops 工作基线为 `3564986ae8a4d9b925a6c81f7c569c8e5b7eae5d`，验证包含本次 OP1 工作区改动。配套使用 InfiniRT 提交 `6ffd4e671fce8c591b3b188e30b6dcbb8cd4892e` 的 CPU 安装包，安装前缀为 `/tmp/infinirt-rt3-release/tests/install_consumer_prefix`。Ops 依赖其 RT2/RT3 公共稀疏接口，不将尚无 `SparseTensorView` 的旧 RT 安装包视为配套版本。

| 文件 | 改动 |
| --- | --- |
| `src/tensor.h` | 保留 dense `Tensor` alias，新增 `SparseTensor` / `TensorLayout` alias；显式参数 traits、递归 optional/list 遍历和设备发现 |
| `src/sparse.h` | D0 错误状态、RT 状态转换、metadata-only 公共检查及纯元数据 `SparseMetadataKey` |
| `src/operator.h` | sparse 参数的 Make/CanMake、checked provider 分派、缓存键和 trace 接入；阻止进入旧执行入口 |
| `src/tuning.h` | 拒绝从 sparse 参数包构造旧 dense tuning signature |
| `scripts/generate_public_operator_header.py` | 将新增 provider 相关函数体转成安装头声明，避免消费者翻译单元重新做 provider 选择 |
| `tests/cpp/sparse_operator.cc`、`tests/test_sparse_operator.py` | 使用测试 provider 验证分派、状态、缓存边界、索引改写和调优隔离，Debug/Release 均运行 |
| `tests/test_generate_public_operator_header.py` | 覆盖新增 checked 入口和私有辅助入口的声明化 |

没有手改 `generated/`。公开头生成器这里只补上新增函数体的导出边界；OP2 的 dense/sparse 类型分类、overload/optional/list 签名生成及稀疏公开实例化仍未完成。

## 参数与公共检查

`SparseTensor` 是 `infini::rt::SparseTensorView` 的别名，保持借用组件存储的生命周期约束，不伪装成单一 `data/strides` 的 dense view。参数 traits 区分 dense/sparse，并递归处理 `std::optional`、`std::vector`；参数包中任意位置出现 sparse 类型都会启用相应路径，包括 dense 首参和空的 optional/list。

公共检查只读元数据。它遍历所有已提供的输入/输出，比较完整 device type/index，重新执行 RT sparse metadata 工厂检查，并要求 CPU、float32/float64 且 value dtype 一致。dense operands/out 同样检查连续性、负 stride、非空 null、自然对齐、字节数及地址端点溢出。空组件不被解引用。参数的形状关系、资源范围、alias、索引内容及真实计算由后续 provider 检查。

Ops `ValidationCode` 覆盖 D0 的元数据、内容、能力及执行错误类别。`ValidationResult` 包含 `code`、RT `ValidationComponent`、可选 `position` 与顶层参数序号 `argument`；`FromRtValidation` 保留 RT 错误类别、组件及位置。成功的 RT 构造或 Ops metadata 检查均不证明 COO 坐标合法、coalesced 声明真实或 CSR 结构有效。

## Checked entry 与 provider 约束

源码入口为 `Operator<Key>::CheckedCall(handle, config, args...) noexcept -> ValidationResult`。它先执行公共 metadata/设备/能力检查，再选择 provider。`Config{}` 与 automatic 配置按 provider 顺序选择符合条件的实现；显式 implementation 不存在或不支持请求返回 `UnsupportedImplementation`，不会改选其他实现。自动选择没有符合条件的 provider 返回 `NoProvider`。CPU 以外设备或不支持的 value dtype 在公共检查阶段返回 `UnsupportedCapability`；RT 的 `UnsupportedLayout` 通过状态转换保留。

可执行 provider 必须显式提供 `supports(...)` 或 config-aware `supports(config, ...)`，并提供静态入口：

```cpp
static ValidationResult checked_call(const Handle& handle,
                                     const Config& resolved_config,
                                     const Args&... args);
```

该入口每次调用都获得当前参数和已解析的 implementation index。provider 必须在读取索引或执行计算前完成资源、alias 和内容校验，并在同步完成后返回结果；返回失败后不会继续尝试另一个 provider。仅有旧构造函数或 `operator()` 的 provider 不具备 checked 执行资格。分派层将 `std::bad_alloc` 转为 `AllocationFailure`，其他未归类异常转为 `ExecutionFailure`；provider 应主动返回更具体的错误状态。

`Make` / `CanMake` 仅用于元数据能力探测和构造，不建立内容校验证明，保留原有异常/布尔语义。传入 sparse 参数的旧 `Call` 和通用实例 `operator()` 均拒绝执行并提示使用 `CheckedCall`。dense 调用保留原执行和调优路径。

## 缓存、调优与 trace

`SparseMetadataKey` 保存 layout、逻辑 shape、维度分解、nnz、value/index dtype、完整 device、组件 shape/strides/dtype/device 和 COO coalesced 声明。它不包含 data 指针、内容校验证明或结构分析。混合请求的 cache key 也清除 dense operand 的 data 指针，并记录参数顺序和 optional/list 边界及算法标量属性。

相同元数据、不同 indices 可以有相同 metadata key；这只允许复用纯元数据决策。当前 `CheckedCall` 不复用旧算子实例或结构分析，每次进入 provider 的 `checked_call`，因此同地址索引改写、地址复用和 values 更新均传入当前调用。后续若引入结构 plan，仍须满足 D0 的 owner/结构身份及稳定期约束。

稀疏请求不查询或写入旧 tuning cache，也不进行在线 benchmark。直接调用旧 `TuningSignature::Build` 或 benchmark 辅助入口且参数包含 sparse 类型时会拒绝，空 optional/list 也不会被当成 dense tuning 请求。普通 trace 能识别 sparse-only 请求的设备，详细 trace 增加 layout、shape、nnz、dtype/index dtype 和 COO coalesced 声明；trace 不读取组件数据。

## 验证记录

本次专项测试用当前 Ops 源码头和配套 RT 安装头/库编译，不依赖 Torch 或实际稀疏内核。C++ 检查使用显式条件和退出码，Release 的 `NDEBUG` 不会移除检查。测试 provider 和 tuning manager 替身用于观测选择、内容验证次数和 tuning 访问次数，不代表 OP4 算法已实现。

| 检查 | 结果与证据范围 |
| --- | --- |
| OP1 专项 Debug / Release | 2 passed；两种配置均以 `-Werror` 编译，各运行普通调用与 trace 检查 |
| wrapper / public-header 生成器回归 | 49 passed；新增 provider 函数体不泄漏到安装头 |
| 专项与生成器合并运行 | 51 passed；`git diff --check` 通过 |
| CPU Release 构建与安装 | 最新源码构建、安装通过，未启用 Torch provider |
| 安装 C++ 回归 `tests/test_cpp_api.py` | 10 passed、1 failed、1 skipped；失败与跳过原因见下文 |
| 安装 sparse metadata 消费者 | 仅 include `<infini/ops.h>`，COO 构造、公共 metadata 检查及 metadata key 编译运行通过 |
| OP1 专项 ASan / UBSan | 通过；`-O1 -g -fsanitize=address,undefined -fno-pie -no-pie`，开启 leak 检查，无 sanitizer 报告 |

专项覆盖包括：纯 sparse 和 dense/sparse 混合参数、nested optional/list、无 config 的 dense 首参调用、空 dense list 配合 sparse 参数、optional dense 首参、同 type 不同 index 的设备拒绝、错误 dtype/stride/null/对齐/地址端点、空 COO、明确 implementation 错误、legacy provider 拒绝、异常转状态、实例 `operator()` 拒绝绕过 checked entry，以及启用全局 tuning 时 sparse 不查询/记录/benchmark。合法但不可访问的对齐地址仍可通过 metadata 检查和构造，证明这些路径不读取索引或 values。

D0 C01–C04 的证据来自测试 provider：COO/CSR key 不混用；相同几何不同 indices 每次重新验证；同地址索引改写及存储复用不会沿用旧证明；values 改写能在下一次调用中观察到。没有真实稀疏算子或符号 plan，因此不把这些测试计作 OP4 内容校验、GPU2 plan 生命周期或真实数值计算验收。

安装回归唯一失败是 `test_cpp_configless_calls_use_first_active_implementation`：当前原生 CPU 安装不提供该测试引用的 `Abs`，其自定义 `ConfiglessSelection` 也没有可用的公开 `OperatorFactory` 定义。使用仓库 HEAD 原始 `src/operator.h` 与原始公开头生成器重新生成安装头，复现相同的 `Abs` 未声明 / `OperatorFactory` 不完整编译错误，确认不是 OP1 引入的回归；未修改这一无关测试。跳过项是 CPU 构建没有对应 wrapper 的 Ascend workspace 测试。安装 sparse 消费者只验证元数据接口可见和可用，不覆盖尚未完成的稀疏公开算子实例化。

专项与生成器复现命令：

```sh
INFINI_RT_ROOT=/tmp/infinirt-rt3-release/tests/install_consumer_prefix \
  .venv/bin/python -m pytest tests/test_sparse_operator.py \
  tests/test_generate_wrappers.py tests/test_generate_public_operator_header.py -q

cmake --build /tmp/infiniops-op1-release -j 4
cmake --install /tmp/infiniops-op1-release
INFINI_OPS_INSTALL_PREFIX=/tmp/infiniops-op1-install \
  INFINI_RT_ROOT=/tmp/infinirt-rt3-release/tests/install_consumer_prefix \
  LD_LIBRARY_PATH=/tmp/infinirt-rt3-release/tests/install_consumer_prefix/lib \
  .venv/bin/python -m pytest tests/test_cpp_api.py -q
```

后续按 [执行计划](04-execution-plan.md#任务拆分与合并顺序) 推进 OP2 的生成/实例化、OP3 的 Python checked import 和 OP4 的内容校验及实际 CPU provider。现阶段不能通过此基础设施宣称 ToDense、SparseMm 或其他 sparse 算子可用。
