# rigModel

使用预训练绑骨模型为机械旋转关节生成位置候选。

## 研究状态

- **为什么做**：尝试用绑骨模型补充已有脚本和神经网络的关节原点定位。
- **核心问题**：UniRig 生成的关节点是否接近自有物体的机械旋转轴？
- **成功标准**：TODO；本轮先沿用子件尺度 1% / 2% 的探索性位置阈值，尚非产品验收标准。
- **停止标准**：TODO。
- **预计投入**：本轮准备并推理 25 个既有持续旋转物体；后续投入 TODO。
- **当前阶段**：CPU 准备和预检已通过，等待用户开启 GPU。
- **最新结论**：25 个物体输入和 65 根参考轴核对完成；585 个权重张量严格匹配，状态为 READY_FOR_GPU。尚未进行 GPU 推理，暂无模型准确率。

## 本轮范围

用户确认的 25 个物体、65 根 continuous 参考轴。整物体只预测一次骨架；65 根轴分别用于离线分析。第 18 个仅含有限旋转的压汁器不进入本轮。保留原装配姿态，不采用预测骨架父子表替换机械装配图。

`third_party/UniRig/` 保存官方源码和配置，版本及逐文件哈希见 [UPSTREAM.json](UPSTREAM.json)。适配代码单独放在 `experiments/unirig/`，不修改官方网络、tokenizer 或采样器。不运行蒙皮、不训练，不把参考关节输入模型。

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

## 上游许可

保留 UniRig 的 [根许可](third_party/UniRig/LICENSE) 和所有源码文件头；其中 Michelangelo 派生源码带有 GPL-3.0-or-later 声明，不能将整个依赖树统一宣称为 MIT。下载模型的来源及版本见 [configs/assets.lock.json](configs/assets.lock.json)。
