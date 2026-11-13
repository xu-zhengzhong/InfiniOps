# RT2：稀疏类型安装导出

状态：**RT2 安装导出已完成**。本任务将 [RT1 的稀疏描述符](07-rt1-implementation.md) 接入 InfiniRT 公开头生成与安装链。安装消费者只需 include `<infini/rt.h>`，并链接同一次构建的 `InfiniRT::infinirt`，即可使用 `TensorLayout`、`SparseTensorView`、`ValidationResult`、`ValidationCode` 和 `ValidationComponent`。

RT3 正式元数据测试集和所有 Ops 后续任务仍未完成。RT2 不改变 RT1 的 metadata-only 校验、非拥有型生命周期或首期二维 COO/CSR 限制，也不增加内容校验、稀疏计算或 Python 桥接。

## 生成与安装链

实现位于 `/workspace/InfiniRT`：

| 文件 | 改动 |
| --- | --- |
| `scripts/generate_public_headers.py` | 将 `tensor_layout.h`、`sparse_tensor_view.h` 加入 detail roots、include 重写规则及生成的 umbrella include 清单 |
| `CMakeLists.txt` | 跟踪 `src/*.h` 和公开头生成脚本的变化，增量 build 自动重新配置并生成公开头 |
| `tests/install_consumer_smoke.cc` | 扩展仅使用 `<infini/rt.h>` 的 COO/CSR 构造、状态类型和组件访问 smoke，覆盖头文件及库符号的安装消费 |

生成器把本地头引用重写为 `<infini/rt/detail/...>`，安装前缀无需 InfiniRT 源码 include 路径。`src/CMakeLists.txt` 已通过 `*.cc` 收集实现、通过 `generated/include/` 整目录安装导出头，因此本次复用原有规则，无需修改；`generated/` 由生成器生成，不手工维护。

顶层 CMake 使用 `GLOB_RECURSE ... CONFIGURE_DEPENDS` 发现源码头集合，并通过 `CMAKE_CONFIGURE_DEPENDS` 跟踪这些头文件和生成脚本的内容变化。修改相关声明后，常规 `cmake --build` 会触发重生成，避免库实现更新而安装声明停留在旧版本。

安装消费者保留原有 dense/runtime 覆盖。稀疏检查位于 runtime 设备探测之前，不会因机器缺少加速器而被提前跳过；检查使用显式条件和返回码，在 Release 下仍执行。

## 验证记录

2026-11-13 在 Linux x86-64、GCC 13.3.0、CPU 后端完成本地验证。工作基线为 InfiniRT `9eed5ff8b3e6eab2c7e75faf43b31c763a633b2d`、InfiniOps `86c68fc2537299fc3f9513768c3dede8969056e5`；以下验证包含本次 RT2 工作区改动，不表示这些基线提交已经包含 RT2。

| 检查 | 结果 |
| --- | --- |
| CPU Release 构建及 CTest（启用 performance tests） | 15/15 通过，含安装消费者、现有 dense 与 performance 测试 |
| 安装消费者依赖隔离 | 编译 flags、链接和依赖文件仅使用测试安装前缀的头与库，无 InfiniRT `src/`、`include/`、`generated/` 依赖 |
| 10 个后端的独立公开头生成 | 10/10 通过；新头导出、include 重写和生成头依赖闭包完整 |
| 增量重生成 | 3/3 通过；分别修改两个新头和生成脚本后，仅执行 build 即自动重配并刷新生成内容 |
| 安装消费者 C++ 格式 | clang-format 21.1.8 检查通过 |
| 文档与空白 | InfiniOps `git diff --check`、7 份修改/新增 Markdown 的相对链接、代码围栏及末尾换行检查通过 |

CPU 构建与测试命令：

```sh
cmake -S /workspace/InfiniRT -B /tmp/infinirt-rt2-release \
  -DCMAKE_BUILD_TYPE=Release -DWITH_CPU=ON \
  -DINFINI_RT_BUILD_TESTING=ON -DINFINI_RT_BUILD_PERFORMANCE_TESTING=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build /tmp/infinirt-rt2-release -j 4
ctest --test-dir /tmp/infinirt-rt2-release --output-on-failure
```

安装消费者使用 `/tmp/infinirt-rt2-release/tests/install_consumer_prefix`，由现有 CTest 安装 fixture 和独立 CMake consumer 配置完成编译、链接及运行。

另外两个验证脚本是本次工作区的临时探针，不是 RT3 正式测试：

```sh
python3 /tmp/infinirt-rt2-backends-check.py
python3 /tmp/infinirt-rt2-incremental-check.py
```

第一个脚本分别为 cpu、nvidia、iluvatar、hygon、thead、metax、mars、moore、cambricon、ascend 生成独立输出，检查两个新头及其本地依赖已正确导出。它不执行加速器 SDK 编译或设备运行。第二个脚本在临时源码副本中完成 CPU Release Ninja 初始构建，依次修改 `sparse_tensor_view.h`、`tensor_layout.h`、`generate_public_headers.py`，验证每次 `cmake --build` 自动重配并刷新对应生成内容，不修改原仓库源码。

## 后续边界

RT2 的安装 smoke 证明公开头与匹配库可被消费，不替代 RT3 的元数据、溢出、设备、空组件及 dense 回归正式测试集。Ops 参数 traits、生成器、checked import、内容校验、动态输出与 provider 按 [执行计划](04-execution-plan.md#任务拆分与合并顺序) 继续实施；P1/P2 尚未完成。
