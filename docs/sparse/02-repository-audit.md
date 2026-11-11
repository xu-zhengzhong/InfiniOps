# 两仓库现状与差距

本页是指定提交的源码审查结果。`Ops/...` 表示 `/workspace/InfiniOps/...`，`RT/...` 表示 `/workspace/InfiniRT/...`；行号为定位起点。未发现适用的 `AGENTS.md`。未构建仓库，不能从“存在声明”推断运行时支持。

## 实际依赖和公共边界

`Ops/src/tensor.h:8` 把 `Tensor` 定义为 `infini::rt::TensorView` 的别名，没有自己的 owning Tensor 或 TensorImpl。主改造入口应在 RT 视图及 Ops 的参数处理。`Ops/src/tensor.cc` 是遗留文件，`Ops/src/CMakeLists.txt:67` 明确将它排除，不能把修改该文件作为生效方案。

两库都通过生成公共头导出 C++ API。Ops 的 Python 接口通过 pybind11 生成；`Ops/docs/compatibility.md:18` 明确没有独立的正式 C API。本方案不增加 C ABI。头文件、动态库和 Ops wheel 需配套构建；保持 dense 类型不变降低兼容风险，但不构成跨版本二进制兼容承诺。

## InfiniRT：表示层和资源层

| 当前事实 | 证据 | 稀疏差距/建议 |
| --- | --- | --- |
| TensorView 只有一个 data 指针、shape/strides、dtype、device | `RT/src/tensor_view.h:36`、`:261` | 无法表达 COO 两组件或 CSR 三组件；新增独立 sparse view |
| view 借用数据，拥有元数据；低 rank 内联，高 rank 堆存储 | `RT/docs/api/core-types.md:42`、`:70` | sparse view 保持相同所有权原则，避免让 dense 对象变大 |
| TensorLike 约束要求 data/strides | `RT/src/tensor_view.h:22`、`:58` | 不为 sparse 伪造整体 data/strides |
| 切片和转置依赖 strides 与指针偏移 | `RT/src/tensor_view.cc:18`、`:67` | sparse 转置/切片需专门语义，不能继承 dense 实现 |
| numel 是逻辑 shape 乘积 | `RT/src/tensor_view.h:228` | 新 API 区分逻辑元素、nnz、capacity、bytes 并检查溢出 |
| hash/equality 不包含指针，只包含几何信息 | `RT/src/tensor_view.h:273`、`:294` | 不能用它证明稀疏结构内容相同 |
| dtype 已含 int32/int64/bool，没有 complex | `RT/src/data_type.h:14` | 索引类型可复用；不承诺 PyTorch 全 value dtype |
| device 含 type 和 index | `RT/src/device.h:13` | 组件必须校验完整设备一致 |
| DeviceGuard 与内存池已存在 | `RT/include/infini/rt/device_guard.h:13`、`RT/include/infini/rt/reusable_memory_pool.h:11` | 复用资源机制；显式覆盖多组件的共同生命周期 |
| runtime backend 选择是 thread-local | `RT/scripts/generate_public_headers.py:676` | 分配、复制、释放不能依赖线程偶然选择的 backend |
| event、async、graph 支持按后端不同 | `RT/docs/backends.md:31`、`RT/docs/api/runtime.md:82` | 每个后端单独声明能力和同步限制 |
| capture 内不能按需扩容，pin 不保证独占 | `RT/src/reusable_memory_pool.cc:85`、`:181` | capture 必须预分配，pin 不代表结构不可变 |

RT 公共入口 `include/infini/rt.h:4` 仅导入生成头。新增类型必须同时接入 `scripts/generate_public_headers.py:119` 的 include 重写、`:145` 的 detail roots、`:199` 的 umbrella 清单，以及 `CMakeLists.txt:389` 的生成和 `src/CMakeLists.txt:168` 的安装链。只在源码树中 include 成功不够。

## InfiniOps：整条调用链的 dense 假设

| 位置 | 当前机制 | 必须补齐的能力 |
| --- | --- | --- |
| `src/pybind11_utils.h:246` | 无条件读取 data_ptr、shape、dtype、device、stride | 先判 layout，再分解 sparse 组件；owner 保活 |
| `src/torch/tensor_.h:93` | 单指针 + shape/strides → ATen dense Tensor | 新增 sparse 工厂组合包装 |
| `src/torch/external_tensor.h:22` | `at::for_blob` 包装外部存储 | 确保所有组件生命周期，不能把外层 sparse 直接 for_blob |
| `scripts/generate_wrappers.py:1265`、`:1567` | 类型名含 Tensor 就可能归一化为 Tensor | 明确 dense/sparse/optional/list 类型分类 |
| `scripts/generate_wrappers.py:709` | 依赖上述分类生成 Python 转换 | sparse converter 和结果协议 |
| `scripts/generate_torch_ops.py:1117`、`:1367` | 只缓存 shape/strides/type，并调用 dense ToAtenTensor | 显式 layout contract，首期手写 sparse provider |
| `scripts/generate_torch_ops.py:1425` | 直接调用 ATen `_out` | 确认稀疏组件是否被替换，容量和结果元数据回传 |
| `scripts/generate_public_operator_header.py:19`、`:45` | public provider/Make 解析包含 dense 匹配 | sparse overload 导出与消费者编译 |
| `src/operator.h:140`、`:155`、`:392` | 设备发现、TensorLike、Make/CanMake 面向 dense | 统一参数 traits，识别稀疏输入和所有设备约束 |
| `src/operator.h:46` | 无 supports 的旧 provider 默认 true | 保证 sparse 不落入旧 dense provider |
| `src/operator.h:73`、`:305` | CacheKey 保存 dense tensor 元数据 | sparse 签名和结构 plan 分离 |
| `src/tuning.h:31`、`:69` | tuning tensor key 只有 shape/dtype | 首期关闭 sparse tuning，后续独立签名 |
| `src/handle.h:13` | 已有 stream/workspace lease 契约 | sparse 临时资源和 capture 延长寿命 |

`SparseTensorView` 这个名称本身含 Tensor，因此生成器改造必须先于把它写入算子签名。不能让源码看起来是 sparse，生成后的公共签名又变回 dense。`generated/` 是构建产物，只修改生成器、源声明或配置。

### 已存在的 sparse_sampled_addmm 为什么不等于支持 sparse

`scripts/torch_ops.yaml:381` 包含 `sparse_sampled_addmm`，`src/base/sparse_sampled_addmm.h:10` 也有抽象基类，但 input/out 仍是单指针 dense Tensor，成员只保存 shape/strides/type。未找到对应的专用原生 sparse provider，通用 Torch 生成路径也只会构造 dense ATen Tensor。

因此现状是“存在名称与基类骨架，但类型和桥接不能表达它需要的 CSR 输入”。应新增明确的 CSR 参数签名、更新生成规则并添加专用测试，不能把已有 YAML 条目当作验收证据。

### 通用测试存在掩盖空缺的路径

`tests/test_torch_ops.py:624` 使用通用 dense 随机输入，参考调用不接受时可能 skip；`:696` 还会跳过零元素输出。这类测试不能证明 sparse support。需要专用 fixture，不能把稀疏参考构造失败当作正常通过。

已有 `tests/test_generate_wrappers.py`、`tests/test_generate_torch_ops.py`、`tests/test_generate_public_operator_header.py`、`tests/test_cpp_api.py:10` 可承接生成及安装消费者验证。新增 sparse case 需要同时比较 layout、indices、values、nnz 和 dense 数值结果。

### 当前不应自动扩大的路径

NineToothed 的 `src/ninetoothed/tensor.h:27` 等包装按 dense component 工作。不能默认认为 Triton、TVM/DLPack、linked provider 已支持多组件 sparse 对象；这些边界应明确拒绝整体 sparse，或另行增加组件级协议。Ops 的现有 Torch provider slot 8 继续保留，新增 provider 不占用其他已用槽位。

## 结论对应的工作边界

| 责任 | InfiniRT | InfiniOps | 调用方/上层框架 |
| --- | --- | --- | --- |
| 非拥有型结构表示 | 主责 | 使用并导出适配类型 | 提供真实组件和逻辑 shape |
| 元数据校验 | 主责 | 算子前置条件补充 | 处理失败结果 |
| indices 内容校验 | 不隐式执行设备 kernel | CPU/device 显式校验 | 可信路径保证结构未修改 |
| coalesce/转换/SpMM | 无 | 主责 | 分配输出与 workspace |
| 内存/stream/event | 通用资源机制 | 正确绑定调用资源 | 保活 owner、建立跨 stream 依赖 |
| PyTorch 导入导出 | 无 libtorch 依赖 | 主责 | 确定同步和 autograd 边界 |
| owning Tensor / autograd | 不新增完整框架 | 首期不实现 | 后续另行设计 |

更具体的类型契约见 [目标设计](03-design.md)，逐文件任务和阶段出口见 [执行计划](04-execution-plan.md)。
