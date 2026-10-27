# KVServe runtime API 1：DCU 联调版

分支：`kvserve-dcu-runtime`，基于上游 `v1` 的
`fdc92fc07e706d18ff0d321646f176c5e749096e`。包版本：`0.2.0+dcu.1`。

本版是**待 DCU 实机验证的移植代码**，不是“已在 DCU 跑通”的发布版。
Windows 开发机只能运行 CPU/mock 测试，不能验证 HIP 编译、真实压缩率或 PD 性能。

## 与对方旧 DCU 源码的关系

参考材料为 `KVServe_DCU_source_20261027`：旧 `fused` 基线
`2fadbf2afa24b5495429d92c40755815b20a6b41` 上的工作区快照，没有 Git 历史。

- 沿用其 **ctypes → HIP C ABI → hipCOMP** 路径，不制造 `nvidia.nvcomp` 假包。
- 显式支持 `hipcomp_snappy`、`hipcomp_cascaded`。对方提供的
  `kvserve_hipcomp_cascaded16_tested.cpp` 实际调用 **Cascaded**，虽然导出的函数名
  带 `snappy`。先用 `dcu_cascaded.json` 对齐这份“tested”源码，再对照测试 Snappy。
- Hadamard 保持归一化 FWHT 数学变换和新版 runtime 的随机符号逻辑。
  本版把新版已有的工作线程 PyTorch 蝶形实现用于所有线程，不依赖 Humming，
  也不强制安装另一份 CUDA/FHT 二进制包。**没有移植旧版的融合 TileLang/FHT 性能内核**。
- 保留新版 runtime API 1、逐请求调度所需的多槽接口、部分尾页恢复及 wire profile。
  不搬旧 vLLM connector，不改 SGLang/Mooncake 的网络协议代码。
- 重新整理 C ABI：独立的槽工作区、明确的算法 ID、长度/状态/帧格式检查、
  256 字节对齐、越界和缓冲区别名检查。GPU 工作区和压缩输出上界进入内存预算。
  不照搬旧版全局 `hipMalloc` 缓存、不透明算法映射或拷贝兜底。

## 行为与边界

- `API_VERSION` 仍为 **1**，可供已有 `sglang-kvserve-provider` 适配器调用。
- 原 `default` 仍选 NVIDIA ANS；DCU 必须显式加载下面的 JSON。
  hipCOMP 不提供 ANS，不能只把导入名称换成 hipCOMP 就继续运行 ANS。
- 压缩前的 KV 仍是 **BF16**；量化参数沿用新版默认值。这条完整流水线是有损压缩，
  Snappy/Cascaded 只是其中的无损编解码阶段。
- 不新增“请求并发必须为 1”或“TP 必须为 2”的限制。
  `slot_count=2` 是工作缓冲槽数，不是服务请求并发数。
  SGLang 原适配器的同构 TP、BF16、布局等限制仍保留。
- 不关闭外部 Store/DFS linker，不更改其数据格式。仅 P→D 传输采用此 runtime；
  Store/DFS 与压缩并用的真实请求、缓存命中/复用仍需实机联调。
- **初版 HIP 编解码同步完成**，有主机元数据和 stream synchronize，
  PyTorch FWHT 也未融合。先验证正确性；目前不能承诺 TTFT/吞吐优于原始传输。
- wire profile 版本改为 3；P/D 两端必须用同一版 KVServe 和相同算法/配置，
  不能与旧 DCU 包或原 ANS 端混用。旧 `libkvserve_hipcomp_snappy.so` 不能直接使用。
- DCU 使用 `KVSERVE_COMPRESSION_BACKEND=python`。旧 `cpp` 后端为 CUDA 专用，
  本次没有把该 CUDA 内核移植到 HIP。

## 1. 安装 Python 包

在这份源码根目录执行；沿用你已有的 DCU PyTorch，不要安装 NVIDIA CUDA 依赖：

```bash
python -m pip wheel --no-deps --no-build-isolation . -w dist
python -m pip install --no-deps --force-reinstall dist/kvserve_v1-0.2.0+dcu.1-py3-none-any.whl
```

这个 wheel 包含 Python 代码和 HIP shim **源码**，**不含预编译 HIP .so**。
平台相关的 `.so` 需下一步在 DCU 工具链上编译，不能从 Windows 打包出来。
P/D 两节点均需安装此版本。

## 2. 准备 hipCOMP，编译本版 shim

前提：DTK/HIP 的 `hipcc` 可用，hipCOMP 已有头文件与 `libhipcomp.so`。
仅拿到对方的旧 wrapper `.so` 不够；本版要链接真正的 hipCOMP 底层库。
优先使用对方跑通时的 hipCOMP/DTK 组合；不要假定公开仓库当前版本与 DTK 兼容。
上游算法和构建说明见 [ROCm/hipCOMP-core](https://github.com/ROCm/hipCOMP-core)。

假设安装前缀为 `/opt/hipcomp`，包含 `include/hipcomp/{snappy,cascaded}.h` 和
`lib/libhipcomp.so` 或 `lib64/libhipcomp.so`：

```bash
python -m kvserve_v1.compression.runtime.build_hipcomp \
  --hipcomp-root /opt/hipcomp \
  --output /opt/kvserve-dcu/libkvserve_hipcomp_runtime.so

export KVSERVE_HIPCOMP_LIBRARY=/opt/kvserve-dcu/libkvserve_hipcomp_runtime.so
export KVSERVE_COMPRESSION_BACKEND=python
ldd "$KVSERVE_HIPCOMP_LIBRARY"
```

`/opt/hipcomp` 是示例路径，替换为真实路径。如果没有安装、只有 hipCOMP 源码和
编译目录，可以改用 `--include-dir /path/hipCOMP-core/include` 和
`--library-dir /path/hipCOMP-core/build/lib`（以实际 `.so` 所在目录为准）。
工具还支持 `--hipcc`、`--offload-arch`、`--hipcomp-library`、`--dry-run`。
先用 `--dry-run` 看命令，再实编译。**本工具不编译或安装 hipCOMP 本身**。

每个启动 P/D 的 shell 都需要导出这两个环境变量；若节点 ABI/工具链一致可复制 shim，
但两节点仍要能加载相同 hipCOMP/HIP 依赖。缺依赖时调整 `LD_LIBRARY_PATH`，
不要安装 `nvidia-nvcomp-cu12/cu13` 来解决 DCU 的问题。

## 3. 先单卡自检，再启动 PD

在 DCU 上、此源码根目录执行（需安装 pytest）：

```bash
HIP_VISIBLE_DEVICES=0 python -m pytest tests/runtime/test_dcu_gpu.py -q
```

该文件包括真正的无损字节往返、坏帧拒绝、两个工作线程的独立工作区、
以及 Spark 几何（36 层中的 9 层 FA、每 rank 2 个 KV heads、head_dim=256、
BF16、TP=2 的 rank 0/1、63/64/65 token、K/V 跨槽恢复）。
“TP rank 测试”是单卡运行两个 rank 的配置，不是已验证双卡通信。
在 HIP 设备上缺 `.so` 或算子错误会直接失败，不会伪装成跳过或原样拷贝成功。

同时在你已有 SGLang 源码目录，执行之前的真实适配器自检：

```bash
HIP_VISIBLE_DEVICES=0 python examples/kvserve/check_runtime.py \
  --model-config /path/to/model/config.json --tp-size 2 --tp-rank 0 \
  --mode @/path/to/KVServe/configs/runtime/dcu_cascaded.json
```

再用 `--tp-rank 1` 测一次；必要时对实际使用的第二张卡重复。`/path/to/...`
均需替换为服务器上的路径。配置文件随源码提供，不安装进 site-packages。

在原 P、D 两份启动脚本中，把：

```bash
--kvserve-compression default
```

同时改成：

```bash
--kvserve-compression @/path/to/KVServe/configs/runtime/dcu_cascaded.json
```

其他 master/client/router/Store 参数保持原样。要测 Snappy，两端同时换成
`dcu_snappy.json`。这两个配置仅改变算法和工作块设置，量化参数沿用新版默认值；
`logical_blocks_per_chunk=2` 是便于首轮检查的分块参数，不是只能处理 128 token。
长请求仍由 SGLang 分多块传输。

之后按原流程启动 master → client → P/D → router，检查：

1. 两端启动日志的 codec、wire profile、内存预算一致，且不再导入 nvCOMP/Humming。
2. 63/64/65 token 及长 prompt 的请求成功；多请求并发、反复复用槽无挂起或串数据。
3. Store/DFS 首次写入与后续命中复用成功，原 SWA 传输正常。
4. 与关闭压缩的基线比较输出质量、实际传输字节、TTFT、吞吐、显存。

## 本地验证记录

CPU 测试只验证 Python 配置、FWHT 数值、ABI 绑定/mock、工作区预算和后端选择。
真实 HIP 测试在无设备环境跳过。原 SGLang API 不因本次新增后端而改变。

原始 DCU 快照、现有 SGLang、Mooncake 源码未修改。若实机编译失败，请保留完整
`hipcc` 命令、首条编译错误、hipCOMP 头文件版本和 `ldd` 输出，先解决工具链/API
差异，不要删除状态检查或退回原样拷贝绕过错误。
