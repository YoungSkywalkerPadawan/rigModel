# UniRig 关节点位置候选实验

## 输入与模型

- 25 个自有装配物体，65 根已有 continuous 参考轴，包含 mimic 从动轴；不输入参考轴、运动类型或机械父子关系。
- 完整装配 `scene.glb`，数值坐标为已有 URDF 参考姿态、米、右手 Z-up。直接读世界变换后的几何，避免 Blender 将这批数值 Z-up GLB 再按标准 Y-up 文件旋转一次。
- 沿用官方 AABB 中心和最大边长归一化到 `[-1,1]` 的公式；显式记录中心和半最大边长，输出按逆变换恢复。沿用官方 SamplerMix 的 65,536 点 / 最多 8,192 顶点样本配置。
- 本轮保留原三角网格，关闭官方 shell 默认的 50,000 面简化，以保留轴孔细节。此输入预处理差异必须随结果报告，不能将其称为官方命令完全复现。
- 官方骨架 checkpoint：Articulation-XL2.0，256 坐标量化。使用官方 15 beams、top-k 5、top-p 0.95、temperature 1.5、max-new-tokens 2048，seed 12345。
- 只需要 OPT config，网络参数均来自 UniRig checkpoint；不需要下载 OPT 语言模型权重或蒙皮权重。骨架直接经官方 `UniRigAR.predict_step` 生成，跳过 Lightning 数据编排和 Blender 导出。

## 离线评估

固定参考方向，计算每个预测节点到参考轴线的垂直距离，再除以旧评估冻结的 `child_scale_m`。沿轴方向平移不影响该指标。每轴最近的节点是 oracle，只反映候选覆盖能力；可同时输出旧基准子件中心的距离作简单对照。不能把 oracle 当自动关联精度。

轴标签与推理输入分文件。准备阶段核对每个源 GLB / joints JSON 与旧评估哈希，推理入口仅读取 `inputs.json`、点云和坐标变换。失败和缺失预测保留在全部 65 根轴的分母中。

## 限制

当前没有给骨架节点关联机械部件，未融合 VLM 方向，未进行自动 URDF 导出或完整运动可行性验证。骨架节点数量、沿骨段位置和动画蒙皮质量不能直接视为机械轴心精度。256 级整物体坐标量化对小部件可能产生较大相对位置误差。

公开 checkpoint 的训练数据与本批资产重合尚未排除。参考标签来自已有几何拟合和整理，非实物测量。不能将一轮预训练推理结果外推为所有机械关节的精度结论。

验证：`python experiments/unirig/test_contracts.py`；CPU 完整准备验证：`python experiments/unirig/preflight.py`。CPU 预检只检查输入、依赖、哈希及 mmap/meta 权重键/形状；GPU 上的完整前向和 CUDA 内核需开卡后实测。
