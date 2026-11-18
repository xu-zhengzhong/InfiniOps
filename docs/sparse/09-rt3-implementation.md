# RT3：稀疏元数据正式测试

状态：**RT3 正式测试已完成**。将 RT1 的临时契约探针整理为 InfiniRT 的 CTest 测试，并扩展仅 include `<infini/rt.h>` 的安装消费者。实现与导出接口仍沿用 [RT1](07-rt1-implementation.md) / [RT2](08-rt2-implementation.md)，没有修改 dense 或 sparse 运行时源码。

## 文件与覆盖

实现位于 `/workspace/InfiniRT`：

| 文件 | 改动 |
| --- | --- |
| `tests/test_sparse_tensor_view.cc` | 正式元数据、错误状态、边界、访问、生命周期和零分配测试 |
| `tests/CMakeLists.txt` | 将 `test_sparse_tensor_view` 注册到常规测试集，不依赖加速器设备 |
| `tests/install_consumer_smoke.cc` | 安装入口的错误码/组件/位置、失败清空结果、空 COO/CSR 及组件访问回归 |

测试使用显式条件及退出码，`NDEBUG` 不会移除检查；编译期另检查 sparse/dense 类型边界、复制/移动约束和临时 sparse 对象不可借出组件指针。

| 范围 | 正式用例 |
| --- | --- |
| 支持组合 | 显式二维非方阵 COO、CSR int32/int64，13 种合法 value storage dtype，所有合法 device type/index 的描述 |
| 几何与状态 | 不支持布局、batch/hybrid、组件数量/null、rank/shape/stride rank、索引 dtype/宽度配对、非法 value dtype、COO coalesced 声明及 checked/raw 访问 |
| 设备与存储 | 设备类型/编号不匹配、非法设备描述；各组件非空 null、自然对齐、非连续及负 stride |
| 空与单例 | `(0,0)`、`(0,K)`、`(M,0)`、非空 shape 且 N=0；空组件 null/非负 stride/对齐豁免、CSR crow 仍长 M+1、singleton stride |
| 算术边界 | 逻辑 numel、int32/int64 索引范围、组件 bytes 和地址端点溢出；不实际分配巨大张量；逻辑 numel 不乘 value element size |
| 生命周期 | 元数据复制后原 view/shape 可销毁或修改；copy/move、moved-from 状态、借用组件的工厂自替换、失败清空已有输出、零分配 |

Linux 使用 `mmap(PROT_NONE)` 作为组件地址：合法描述符仍构造成功，若工厂尝试读取索引或 values 会立即失败。其他平台仍运行元数据用例，只将该地址替换为对齐的本地存储。合法设备描述不要求对应 backend 已启用，也不代表该设备能够执行稀疏计算。

测试覆盖 D0 S04、S09–S13 中属于 RT 的元数据部分。S09 的 crow 内容全零、S10 的数值行为、S12 的 alias/workspace/capacity 等仍属于后续 Ops；RT 接受 coalesced 声明不证明索引有序或唯一。

## 验证记录

2026-11-16 在 Linux x86-64、GCC 13.3.0、CMake 4.2.3、CPU 后端完成验证。工作基线为 InfiniRT `27b7c349230a625d0169c87c14397a3d30c33e26`、InfiniOps `80d752b24989619b3b81c8deb588f333ef36510e`；验证包含本次 RT3 工作区改动，不表示基线提交已经包含 RT3。

| 检查 | 结果 |
| --- | --- |
| CPU Release 构建及完整 CTest，含 performance 与安装消费者 | 16/16 通过 |
| CPU Debug 构建及完整 CTest，含安装消费者 | 12/12 通过 |
| `test_sparse_tensor_view` | Release / Debug 均为 877 项检查、0 失败 |
| ASan/UBSan：sparse、dense core、dense allocation | 3/3 通过；稀疏测试 877 项检查、0 失败，无 sanitizer 报告 |
| 安装消费者依赖隔离及 Release 编译 flags | 通过；使用安装头/库，`-DNDEBUG` 下仍执行拒绝检查 |
| clang-format 21.1.8、两库 diff 空白及文档链接/围栏/末尾换行 | 通过 |

Release / Debug 命令：

```sh
cmake -S /workspace/InfiniRT -B /tmp/infinirt-rt3-release \
  -DCMAKE_BUILD_TYPE=Release -DWITH_CPU=ON \
  -DINFINI_RT_BUILD_TESTING=ON -DINFINI_RT_BUILD_PERFORMANCE_TESTING=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build /tmp/infinirt-rt3-release -j 4
ctest --test-dir /tmp/infinirt-rt3-release --output-on-failure
cmake -S /workspace/InfiniRT -B /tmp/infinirt-rt3-debug \
  -DCMAKE_BUILD_TYPE=Debug -DWITH_CPU=ON -DINFINI_RT_BUILD_TESTING=ON
cmake --build /tmp/infinirt-rt3-debug -j 4
ctest --test-dir /tmp/infinirt-rt3-debug --output-on-failure
```

ASan/UBSan 检查正式稀疏测试和相关 dense 回归：

```sh
cmake -S /workspace/InfiniRT -B /tmp/infinirt-rt3-sanitized \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DWITH_CPU=ON -DINFINI_RT_BUILD_TESTING=ON \
  '-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer'
cmake --build /tmp/infinirt-rt3-sanitized \
  --target test_sparse_tensor_view test_core test_tensor_view_allocations -j 4
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 \
  ctest --test-dir /tmp/infinirt-rt3-sanitized --output-on-failure \
  -R '^(test_sparse_tensor_view|test_core|test_tensor_view_allocations)$'
```

dense 回归复用 `test_core`、`test_tensor_view_allocations` 和 performance 测试，覆盖 rank 0–8 内联、高 rank 回退、复制/移动、metadata hash/equality。`perf_tensor_view_footprint` 输出的本机 `sizeof(TensorView)=160` 与 RT1 记录一致；dense 源文件没有改动。基准运行成功不表示已建立跨平台性能回归阈值，也不宣称速度提升。

安装消费者通过现有 CTest fixture 安装、独立配置、编译、链接并运行。Release 编译 flags 含 `-DNDEBUG`，依赖文件没有 InfiniRT 的 `src/`、`include/` 或 `generated/` 头路径；只使用 `/tmp/infinirt-rt3-release/tests/install_consumer_prefix` 的公开头与匹配动态库。

## 后续边界

RT1–RT3 的交付仅覆盖描述、安装和元数据验证。未执行加速器 SDK 构建或设备 kernel，未增加内容校验、Python sparse 桥接、动态输出或 sparse 算子。P1/P2 尚未完成，继续按 [执行计划](04-execution-plan.md#任务拆分与合并顺序) 推进 OP1/OP2 及后续任务。
