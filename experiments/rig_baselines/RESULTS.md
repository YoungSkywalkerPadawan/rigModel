# RigAnything / Puppeteer 实验记录

## 2026-09-17：CPU 准备完成，等待 GPU 单例验证

### 目的

在 UniRig 首轮之后，按用户要求接入另外两套公开绑骨模型，检查它们的关节点能否提供更好的机械旋转轴位置候选。本节只记录源码、依赖、权重和输入准备，不包含 GPU 模型效果。

### 条件

- 数据：首轮冻结的 25 个持续旋转物体、65 根参考轴；完整装配场景，保留所有世界变换。原输入清单 SHA-256 为 `7f78c2904c7a7427194324144543d63d8279a2d310d71ebedc1ea1027874f6eb`。
- RigAnything 上游 commit：`d03cdb21dd134fa81df6b0947522469db3f78bd2`。
- Puppeteer 上游 commit：`1c0f9fc6ad209667a0ec5ceac9b59964938a8b51`；Michelangelo 子模块：`c88949f294fffccf40b191357ce669cae8694fd9`。
- AutoDL：CPU 模式，0.5 核 / 2 GiB；两个独立 Python 3.11.16 venv，PyTorch 2.4.1+cu124。Puppeteer 使用 Transformers 4.46.1、Accelerate 0.28.0、FlashAttention 2.7.3。
- RigAnything 预处理在 AutoDL CPU 完成；Puppeteer 先在 AutoDL 完成 13 例，再用本地 Python 3.12 两个 CPU 进程完成全部 25 例并回传。NumPy 1.26.4、trimesh 4.2.3、mesh2sdf 1.1.0、scikit-image 0.24.0 保持一致。
- 每例种子 12345；RigAnything 原生 1,024 点输入，Puppeteer 原生 depth-7 SDF / marching cubes 后 8,192 点输入。固定坐标逆变换、保留官方后处理。详见 [README.md](README.md)。

### 结果

| 检查 | RigAnything | Puppeteer |
| --- | --- | --- |
| 完整预处理输入 | 25/25，25,600 个表面点 | 25/25，204,800 个表面点 |
| 冻结参考轴 | 65 | 65 |
| 权重大小 | 2,344,595,770 字节 | 主骨架 4,399,390,278 + 辅助编码器 3,934,164,973 字节 |
| 官方权重 SHA-256 | 通过 | 两份均通过 |
| checkpoint 参数键 | 123 | 702，其中 308 个点编码器键 |
| 模型参数数 | 195,369,478 | 490,195,585 |
| 缺失 / 多余 / 形状不匹配参数 | 0 / 0 / 0 | 0 / 0 / 0 |
| 常量及非持久缓冲区 | 已实际构造 | 已实际构造 |
| `pip check` | 通过 | 通过 |
| 最终 CPU 预检 | `READY_FOR_GPU` | `READY_FOR_GPU` |
| 实际 GPU 推理 | 未运行 | 未运行 |

三份权重总计 **10,678,151,021 字节（10.68 GB / 9.94 GiB）**，位于 AutoDL 的 `models/riganything/`、`models/puppeteer/`。每份官方 revision、文件名及 SHA-256 见 [锁定清单](../../configs/rig-baselines.assets.lock.json)，没有权重进入 Git。Puppeteer 完整 skeleton checkpoint 已包含点编码器全部参数；辅助 checkpoint 仍下载校验并保留，推理时避免重复装载后覆盖。

本地生成的 Puppeteer 输入与 AutoDL 已完成的 **13/13 例**逐数组核验，编码器点、解码坐标点、法线及全部数值坐标变换完全相同。完整 25 例输入压缩包 4,529,056 字节，SHA-256 为 `2263c251182fdaaf1201cffa952c202b980e4424532e2fe26797cb0d5fa44038`。原服务器部分结果保留在 `data/user25_puppeteer_autodl_cpu_partial_20260917`。

适配的 4 项 CPU 合约测试通过：固定旋转及逆变换、Puppeteer 两级逆变换、任意节点编号的骨段图验证、原数值后处理在无图形上下文下正常运行。模型结构审计还实际验证了 Puppeteer 的原生 token 解码合约，且确认 RigAnything 代码实际使用 50 个扩散步。没有执行任何学习网络前向。

两次最终 CPU 预检记录的共同源码身份为 `82c7c648766bca11c695ec65f4f6e4e575be85b0c9de7552170d5b12a5e1c8d5`；新增权重清单 SHA-256 为 `6182cfeb54e691b32e737defc727556e19ff59b36edcbe665dd514f232cfa508`。

| 预处理清单 | SHA-256 |
| --- | --- |
| `data/user25_riganything/inputs.json` | `e8a0643cf832a18ffb8995920edc0b60e8447f0923337f12e13c39a57ae4b081` |
| `data/user25_puppeteer/inputs.json` | `ee8c8a8c04982786162d2bbbd3f0a7d6ea73c4f8c60a9cc4063bb18bee3b1ccf` |

原始证据位于服务器 `setup/<model>-preflight.json`、对应日志、`<model>-model-audit.json`、`<model>-environment.freeze.txt`、`cpu_preprocessing_comparison.json` 和 `baselines-cpu-contracts.log`。准备证据归档为 `setup/baselines_preparation_20260917.tar.gz`，包含逐文件 SHA-256 清单。本地副本保存在 Git 忽略的 `outputs/baselines_preparation_20260917/`，仅克隆 Git 的机器需从归档恢复。

证据归档共 17 个文件、17,618 字节，SHA-256 为 `42afcc9834eb2e90d5d94c29c09743f81016360ef70a7faef3ad4280c015f52a`。

### 结论

CPU 可验证的准备已通过，可以通知用户开启 GPU。下一步先各运行一个物体，验证真实模型前向、CUDA 内核、坐标导出及评分流程；通过后跑同一批 25 例。运行命令已准备在 `scripts/run_rig_baselines.sh`。

现在没有两模型的位置精度、耗时或显存结果，不能据此认为它们优于 UniRig、子件中心或旧 v29，也不能把绑骨父子关系当成机械装配结构。

### 局限

- Puppeteer 复用 PyTorch 2.4.1 / CUDA 12.4，与官方的 PyTorch 2.1.1 / CUDA 11.8 不同；GPU 兼容性及数值差异需要开卡后确认。CPU 的结构审计不等于前向验证。
- 两模型使用各自原生采样及归一化，Puppeteer 还会 SDF 重建；本轮不是完全相同输入张量下的模型消融。
- 评估沿用参考方向和最佳节点 oracle，仅表示位置候选覆盖上限；未实现自动节点关联或新的 VLM 方向融合。65 根轴始终保留在全量评分分母中。
- 本地与服务器预处理已对比 13 例，其余 12 例只有本地生成和完整输入校验，未额外重复计算 SDF。
- RigAnything 的 Adobe Research License 限定非商业研究用途；Puppeteer 及其子模块分别保留各自许可。
- 项目成功、停止标准和后续投入仍为 TODO，未代替用户设定。
