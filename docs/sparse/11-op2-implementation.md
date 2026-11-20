# OP2 稀疏签名生成与 C++ 安装入口

生成器从每个构造函数与调用重载自身的声明提取参数类型和标量默认值。它识别 Ops 的 `Tensor`、`SparseTensor` 及对应的 RT view，按声明顺序保留 `std::optional` 和 `std::vector` 的嵌套。libclang 遇到解析错误时使用头文件声明，避免未解析的类型使重载丢失。

对包含 sparse 参数的签名，生成器在库内实例化 `CheckedCall`、`Make` 和 `CanMake`。安装端通过 `<infini/ops.h>` 调用，provider 选择发生在库内。`CheckedCall` 返回 OP1 的 `ValidationResult`，保留 provider 的失败状态。Python 绑定和旧 C 包装只选择 dense 重载。Python sparse layout 路由与组件保活属于 OP3，ToDense 和 SparseMm 的计算与内容校验属于 OP4。

## 验证

在 InfiniOps 基线 `be240ac` 上验证工作区改动，配套 InfiniRT 为 `6ffd4e671fce8c591b3b188e30b6dcbb8cd4892e` 的 CPU 安装包。CPU Release 构建开启 Python 绑定和公开实例化，关闭 Torch provider。

| 检查 | 结果 |
| --- | --- |
| wrapper / Torch / public-header 生成器 | 检查同名重载、限定名、相似类型名、容器顺序及默认值。通用 ATen dense schema 拒绝 sparse 参数。三个生成器测试模块及安装消费者共 81 项通过。 |
| dense 生成对照 | 对 569 个有构造和调用声明的手写算子头，对比修改前后的 Python 包装、旧 C 包装、分派和公开实例化，输出一致。 |
| CPU 安装消费者 | 测试 provider 覆盖 sparse 首参、dense 首参、`vector<optional<SparseTensor>>` 和 `optional<vector<SparseTensor>>`。消费者仅包含 `<infini/ops.h>`，在 `-O0`、`-O2` 下链接并运行，检查成功和错误返回。 |
| 构建与安装 | 原生 CPU Release 库与 Python 扩展构建、安装通过。另在独立源码副本中仅选择测试算子，通过常规 CMake 构建、安装和消费链路。 |

CPU 完整测试使用本次安装的 Python 扩展，执行 `python -m pytest tests --devices cpu -q -n 10 --dist worksteal`，得到 **2419 passed、21 failed、7377 skipped**。在相同 RT 和构建配置下重新构建修改前的 `be240ac`，按用例 ID 重跑，21 项失败全部复现。

| 基线同样失败的用例 | 数量 | 原因 |
| --- | ---: | --- |
| CPU Embedding | 15 | 调用包含非连续张量或额外选项，原生 provider 要求 contiguous plain lookup。 |
| GetCutlassMoeMmData | 2 | CPU 安装未导出该 CUTLASS 绑定。 |
| CUDA 编译探针 | 2 | nvcc 在 sm_52 和 sm_80 均报告 NVIDIA device 特化晚于首次使用。 |
| Ascend workspace 主机编译探针 | 1 | workspace 头中的 `std::string` 类型不完整。 |
| configless C++ 安装用例 | 1 | 原生 CPU 安装缺少 `Abs`，用例内的自定义类缺少公开 factory。 |

专项复现时把两个环境变量指向这次构建所用的安装前缀。安装消费者测试在临时目录生成测试算子及其库，使用已安装的 Ops 公共头、运行库和 RT，测试 provider 只用于验证签名和状态传递。

```sh
INFINI_OPS_INSTALL_PREFIX="$ops_prefix" INFINI_RT_ROOT="$rt_prefix" \
  python -m pytest -q tests/test_generate_wrappers.py \
  tests/test_generate_torch_ops.py tests/test_generate_public_operator_header.py \
  tests/test_cpp_api.py::test_installed_sparse_codegen_consumer
```
