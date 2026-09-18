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

## 2026-09-18：GPU 推理与三模型对比

### 目的

用户开启 GPU 后，实际检验 RigAnything 和 Puppeteer 的原生骨架节点能否定位机械旋转轴，沿用已完成的 UniRig 和旧几何基准做位置对比。先各跑地球仪单例，通过后各跑完整 25 例；没有根据结果修改采样配置或挑选随机种子。

### 条件

使用上述冻结源码、权重、25 物体输入及 65 根参考轴；部署提交 `2cff0b2f133a79794732663c6237835d5725d601`。RTX 4090 24 GiB，PyTorch 2.4.1+cu124，两个独立环境及版本清单沿用 CPU 准备结果。RigAnything 为 BF16 autocast、最多 64 节点、实际 50 扩散步；Puppeteer 为 FP16 autocast、diverse-pose joint-token、1 beam、128 级离散坐标。每例 seed 12345，完整装配，世界坐标逆变换和官方数值后处理不变。

真实单例及全量 GPU 前向均成功，Puppeteer 的 FlashAttention 在当前环境可执行；没有改变网络结构、模型权重或推理适配。只在独立的 `experiments/rig_comparison/` 新增离线报告生成器。

### 结果

| 模型 | 成功物体 | 计分轴 | 总节点 | 逐例耗时之和 | 逐例耗时中位数 | 峰值已分配显存 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UniRig（前轮结果） | 25/25 | 65/65 | 296 | 31.67 秒 | 1.032 秒 | 3.903 GiB |
| RigAnything | 25/25 | 65/65 | 282 | 34.46 秒 | 0.722 秒 | 1.867 GiB |
| Puppeteer | 25/25 | 65/65 | 96 | 9.12 秒 | 0.333 秒 | 2.826 GiB |

逐例计时含输入传输、生成和导出，不包含首次模型加载，也不包含前置 CPU 预处理；不是完整任务墙钟耗时。显存为 PyTorch `max_memory_allocated`，不等于进程总 GPU 占用。两模型地球仪单例与完整批次对应例的世界坐标节点逐项完全一致。

固定参考方向，计算节点到参考轴线的垂距，除以冻结子件尺度：

| 位置来源 | 误差 ≤ 1% | 误差 ≤ 2% | 归一化位置误差中位数 |
| --- | ---: | ---: | ---: |
| UniRig 最佳节点（oracle） | 7/65（10.77%） | 13/65（20.00%） | 32.77% |
| RigAnything 最佳节点（oracle） | 1/65（1.54%） | 6/65（9.23%） | 31.75% |
| Puppeteer 最佳节点（oracle） | 7/65（10.77%） | 17/65（26.15%） | 48.49% |
| 旧基准子件中心 | 39/65（60.00%） | 49/65（75.38%） | 0.63% |
| 旧 v29 位置锚点 | 56/65（86.15%） | 63/65（96.92%） | 0.18% |

RigAnything 的 1% 命中只有望远镜 `azimuth_spin`。Puppeteer 命中地球仪 `globe_spin`、旋转木马 `carousel_spin`、风车 `rotor_spin`、闸阀 `handwheel_spin`、办公椅 `seat_swivel`、永动仪 `orbit_spin` 和切割台 `table_spin`。三模型节点联合，在 1% 阈值下覆盖 **10/65** 根轴，全部已被 v29 命中；对 v29 的新增覆盖为 **0/65**。联合覆盖仍用参考标签选点，不能称为已实现的融合系统。

### 结论

本批资产和当前公开预训练配置下，RigAnything 与 Puppeteer 的离散节点同样不足以直接替换已有原点方案。Puppeteer 在 2% 阈值下比 UniRig 多覆盖 4 根轴，但总体仍低于子件中心和 v29，也未补足 v29 在 1% 阈值下的失败轴。

这轮完成了用户要求的两套模型实验。现有证据回答的是“原生离散节点作为机械轴心候选”的效果，不决定整个绑骨研究方向结束，也未排除骨段插值、几何精修、微调或专门机械轴心模型。

### 核验与归档

两模型 50 例均通过原始文件哈希、原世界坐标还原、骨段索引、GLB 原几何和节点变换保留，以及骨架节点标记位置核验。独立报告生成器使用叉积形式重算三模型全部 **195 个逐轴结果**，与原评分 CSV 一致；同时校验同一资产和参考清单、对照 v29 及子件中心的位置口径。

- RigAnything 全量：`outputs/riganything_user25_20260918T015736Z`；单例：`outputs/riganything_user25_20260918T015629Z`。完整 run identity：`847ab3be50383ad0dd77f8703576f131623f61752f5598fd0d1a14244e7bff05`。
- Puppeteer 全量：`outputs/puppeteer_user25_20260918T015928Z`；单例：`outputs/puppeteer_user25_20260918T015658Z`。完整 run identity：`3088f1f0cda8e48088e314e73904eef66aaead9763c9fdff2f1958548ce63c29`。
- 两模型推理 source identity 均为 `82c7c648766bca11c695ec65f4f6e4e575be85b0c9de7552170d5b12a5e1c8d5`，与前一天 CPU 预检一致。
- 服务器报告：`outputs/rig_comparison_user25_20260918/report.html`；`comparison.json` 含完整逐轴独立计算和运行来源。使用方法见 [报告说明](../rig_comparison/README.md)。
- 服务器归档：`setup/baselines_user25_gpu_20260918.tar.gz`，132,999,531 字节，SHA-256：`79bbeb39a9d4444fdb3104b247bf272914eb7cfe62c46b5611629dc810654191`。归档包含 511 个载荷文件及逐文件 SHA-256 清单。
- 归档保留两次单例、两次全量、前轮 UniRig 全量、三模型报告、原生候选 NPZ/JSON/GLB、逐轴 CSV、运行日志和参数、新模型预处理输入、冻结参考及推理源码包；大权重和完整原始场景不重复打包。
- 本地解包：`D:/project/urdfrelated/rigModel/outputs/baselines_user25_20260918/`，其中 `outputs/rig_comparison_user25_20260918/report.html` 为自包含交互报告；相邻运行目录提供 GLB / JSON 链接目标。下载后验证整体 SHA-256 及全部 511 个载荷文件哈希。

浏览器已检查地球仪单轴模型切换、最佳节点高亮，以及赛车四轴的三模型叠加、俯视和垂距显示；无浏览器脚本错误。页面显示的数字与独立计算结果一致。

### 局限

本轮每个模型只运行一个公开 checkpoint 和一个随机种子，输入采样和模型原生处理不同，不能当作完全受控的模型架构消融。UniRig 首轮沿用 Z-up 输入，本轮两模型适配使用固定 Y-up 旋转；没有搜索朝向或其他超参数来选最好分数。Puppeteer 当前环境已通过真实 GPU 运行，但没有和官方另一套 CUDA 环境做逐数值等价验证。

指标只比较参考方向下的位置候选集合；未实现自动节点关联、VLM 方向融合、机械运动约束或完整 URDF 生成。参考标签来自旧几何整理，训练集与本批资产是否重合未排除。大文件和权重保持在 Git 忽略目录；项目成功标准、停止标准及后续投入仍留 TODO。
