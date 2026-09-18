# 三套绑骨模型的统一位置对比

读取已完成的 UniRig、RigAnything、Puppeteer 运行目录，在 CPU 上生成自包含交互报告。不会运行模型或改变推理输入。与推理适配放在不同目录，使新增报告代码不改变推理源码身份。

比较同一批 25 个物体、65 根参考轴。程序校验来源资产哈希、输入清单、完整运行及坐标审计结果；用叉积独立重算节点到参考轴线的距离，再与每个模型原逐轴 CSV 校对。旧 v29 的位置来自首轮保留的原始记录，采用相同等价锚点口径。

```bash
cd /root/autodl-tmp/rigModel
/root/autodl-tmp/envs/unirig/bin/python experiments/rig_comparison/build_report.py \
  --unirig outputs/unirig_user25_20260917T103010Z \
  --riganything outputs/riganything_user25_20260918T015736Z \
  --puppeteer outputs/puppeteer_user25_20260918T015928Z \
  --output outputs/rig_comparison_user25_20260918
```

输出目录必须尚不存在，以免覆盖已归档报告。`report.html` 内嵌点云、骨架和评分；可切换模型、物体和视角，点击轴行显示各模型最佳节点及到参考轴的垂距。GLB / JSON 下载链接指向相邻原运行目录，归档时应保留这些相对目录。`comparison.json` 保存逐轴独立重算值、汇总、运行身份和互补覆盖清单。

灰色点云统一使用 UniRig 预处理的原装配抽样，仅用于对齐展示；各模型实际使用自身原生采样与归一化。Puppeteer 骨段按原图显示，不强制变为 UniRig 式父节点表。

候选及其联合覆盖都以参考标签挑选最佳节点，属于候选上限，不是自动节点关联准确率。1% / 2% 阈值按冻结子件尺度归一化；本轮没有重新预测方向。实际结论与归档信息见 [实验记录](../rig_baselines/RESULTS.md)。
