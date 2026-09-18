# rigModel

使用预训练绑骨模型为机械旋转关节生成位置候选。

## 研究状态

- **为什么做**：尝试用绑骨模型补充已有脚本和神经网络的关节原点定位。
- **核心问题**：UniRig、RigAnything 和 Puppeteer 生成的关节点是否接近自有物体的机械旋转轴？
- **成功标准**：TODO；本轮先沿用子件尺度 1% / 2% 的探索性位置阈值，尚非产品验收标准。
- **停止标准**：TODO。
- **预计投入**：沿用 25 个既有持续旋转物体，新增 RigAnything / Puppeteer 两套预训练模型实验；后续投入 TODO。
- **当前阶段**：三套预训练模型均完成同一批 25 物体 GPU 推理、65 轴评分和坐标核验；新增 RigAnything / Puppeteer 各 25/25 成功。
- **最新结论**：固定参考方向，用真值选择最佳节点，UniRig / RigAnything / Puppeteer 在 1% 阈值分别覆盖 7/65、1/65、7/65；子件中心为 39/65，旧 v29 为 56/65。三模型联合仅覆盖 10/65，未补足 v29 的失败轴。当前预训练节点候选配置不足以替换已有原点方案；不据此决定整个研究方向结束。

## 本轮范围

用户确认的 25 个物体、65 根 continuous 参考轴。整物体只预测一次骨架；65 根轴分别用于离线分析。第 18 个仅含有限旋转的压汁器不进入本轮。保留原装配姿态，不采用预测骨架父子表替换机械装配图。

`third_party/UniRig/` 保存官方源码和配置，版本及逐文件哈希见 [UPSTREAM.json](UPSTREAM.json)。新接入的 RigAnything、Puppeteer 及其必要子模块见 [UPSTREAM-rig-baselines.json](UPSTREAM-rig-baselines.json)。适配代码分别放在 `experiments/unirig/` 和 `experiments/rig_baselines/`，不修改官方网络、tokenizer 或采样器。不训练，不把参考关节输入模型。只输出骨架候选；RigAnything 保留官方生成函数自带的采样点蒙皮头，不运行全网格蒙皮或动画阶段。

## 使用

AutoDL 准备说明：[docs/AUTODL.md](docs/AUTODL.md)。实验设计与局限：[experiments/unirig/README.md](experiments/unirig/README.md)。实际状态：[experiments/unirig/RESULTS.md](experiments/unirig/RESULTS.md)。

```bash
bash scripts/setup_autodl.sh
/root/autodl-tmp/envs/unirig/bin/python experiments/unirig/prepare.py \
  --source /root/autodl-tmp/continuous-eval-20260914/source/user \
  --reference /root/autodl-tmp/geometric-axis-eval-20260915/prepared/ground_truth.json
/root/autodl-tmp/envs/unirig/bin/python experiments/unirig/preflight.py
# 用户开 GPU 后才执行：
bash scripts/run_gpu.sh
```

生成每例 `candidates.json`、模型坐标骨架 NPZ、世界坐标叠加 GLB、耗时/显存/错误记录，以及离线候选覆盖率。真值选出的最佳节点仅代表候选上限；当前没有自动节点关联结果，也未接入新的 VLM 方向预测。

数据、权重、视频和运行产物不进入 Git，服务器本地目录和校验清单用于复现。仓库不保存 SSH 密码或其他凭据。

## 新增 RigAnything / Puppeteer

两套适配复用首轮冻结的 25 物体 / 65 轴，分别采用各自官方输入预处理。三份权重共 10.68 GB 已下载到 AutoDL 并通过官方 SHA-256 校验。设计、依赖差异及命令见 [适配说明](experiments/rig_baselines/README.md)，CPU 和 GPU 实际结果见 [实验记录](experiments/rig_baselines/RESULTS.md)。2026-09-18 已完成两个单例及两批完整推理；查看现有结果不需要重新开启 GPU。

```bash
bash scripts/run_rig_baselines.sh riganything --case 0001-antique-globe
bash scripts/run_rig_baselines.sh puppeteer --case 0001-antique-globe
# 上述单例通过后：
bash scripts/run_rig_baselines.sh riganything
bash scripts/run_rig_baselines.sh puppeteer
```

## 三模型对比结果

| 位置候选 | 误差 ≤ 1% | 误差 ≤ 2% |
| --- | ---: | ---: |
| UniRig 最佳节点 | 7/65 | 13/65 |
| RigAnything 最佳节点 | 1/65 | 6/65 |
| Puppeteer 最佳节点 | 7/65 | 17/65 |
| 子件中心 | 39/65 | 49/65 |
| 旧 v29 位置锚点 | 56/65 | 63/65 |

模型节点由参考标签选取，属于候选覆盖上限，非自动关联准确率。误差按冻结子件尺度归一化；只评位置，不重新预测方向。

本地完整归档位于 `outputs/baselines_user25_20260918/`；其中 `outputs/rig_comparison_user25_20260918/report.html` 可离线切换三个骨架、逐轴高亮最佳节点并下载叠加 GLB。报告生成器和复现命令见 [统一对比说明](experiments/rig_comparison/README.md)。本地运行产物被 Git 忽略，完整证据与 SHA-256 见 [GPU 实验记录](experiments/rig_baselines/RESULTS.md#2026-09-18gpu-推理与三模型对比)。

## 已完成的 GPU 实验

2026-09-17，RTX 4090。实际生成 296 个骨架节点，25 例处理耗时之和 31.67 秒（不含首次模型加载），PyTorch 峰值已分配显存 3.90 GiB。完整数字、评估口径和归档 SHA-256 见 [实验记录](experiments/unirig/RESULTS.md#2026-09-17gpu-推理与位置评估)。

本地结果目录：`outputs/unirig_user25_20260917/`。打开其中 `results/report.html` 可离线旋转点云、查看预测骨架和参考轴、逐轴高亮最佳节点；每例同时保存叠加 GLB 和候选 JSON。该目录被 Git 忽略，在仅克隆仓库的机器上需从归档恢复。

## 上游许可

保留 UniRig 的 [根许可](third_party/UniRig/LICENSE) 和所有源码文件头；其中 Michelangelo 派生源码带有 GPL-3.0-or-later 声明，不能将整个依赖树统一宣称为 MIT。下载模型的来源及版本见 [configs/assets.lock.json](configs/assets.lock.json)。

RigAnything 的 [Adobe Research License](third_party/RigAnything/LICENSE.md) 限定非商业研究用途。Puppeteer [根许可](third_party/Puppeteer/LICENSE) 为 Apache-2.0，其 [Michelangelo 子模块许可](third_party/Puppeteer/skeleton/third_partys/Michelangelo/LICENSE) 单独保留。新增权重来源、版本和哈希见 [权重清单](configs/rig-baselines.assets.lock.json)。
