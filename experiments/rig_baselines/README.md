# RigAnything / Puppeteer 机械轴位置候选实验

沿用 UniRig 首轮已冻结的 25 个自有持续旋转物体、65 根参考轴，以及 `experiments/unirig/score.py` 的位置指标。每个整物体生成一次骨架；不输入参考轴、部件关联、关节类型或机械父子关系。两套模型都先完成 CPU 准备，用户开 GPU 后才运行。

## 模型与范围

| 项目 | 推理配置 | 输入 |
| --- | --- | --- |
| [RigAnything](https://github.com/Isabella98Liu/RigAnything) | 官方 `RigARDiffusion`，最多 64 个关节点；实际代码固定 50 个扩散采样步，虽然 YAML 写了 300；BF16 autocast | 完整装配表面均匀采样、补齐到 1,024 点，采样点包围盒中心和最大绝对坐标归一化 |
| [Puppeteer](https://github.com/Seed3D/Puppeteer) | 官方 diverse-pose checkpoint；`joint_token`、`seq_shuffle`、1 beam、128 级坐标、最多 100 骨段设置；FP16 autocast | 官方 depth-7 SDF / marching cubes，8,192 表面点及面法线，保留官方 FP16 输入处理和两级输出逆变换 |

每例固定种子 12345，表面采样显式传入种子，便于复现。未进行网格简化。RigAnything 仍执行原生成函数中对采样点的蒙皮头；跳过全体原网格顶点蒙皮及 Blender 导出。Puppeteer 只准备骨架阶段，不加载动画和蒙皮模型。

源码保存在 `third_party/`，逐文件校验和、上游 commit、必要的 Michelangelo 子模块及未纳入的示例资源/其他子模块见 [UPSTREAM-rig-baselines.json](../../UPSTREAM-rig-baselines.json)。上游文本源码保持原样；适配只在本目录。

## 坐标与输出

原场景是米制、右手 Z-up。先将所有几何应用各自世界变换并合成完整装配，再用固定旋转 `(x,y,z) -> (x,z,-y)` 转为 Y-up。RigAnything 的官方 glTF 导入和后续轴变换对应 Y-up；Puppeteer 的渲染/数据约定也使用 Y-up。旋转写入每例 `frame.json`，模型输出通过相应逆变换恢复到原世界坐标。

Puppeteer 的输入编码器收到约 `[-1,1]` 的点坐标，而解码骨架位于 `[-0.5,0.5)`；输出还原严格保留官方 demo 中 `joints * pc_scale + pc_center`、再 `/ scale + center` 的两级步骤。不能省掉其中一级。RigAnything 的归一化来自采样点范围，不能复用 UniRig 全网格包围盒的归一化参数。

每例保存 `candidates.json`、`skeleton_model.npz` 和 `skeleton_world.glb`，记录原文件哈希、坐标变换、骨段索引、耗时、峰值显存和异常。Puppeteer 保存官方后处理后的图及原始骨段/生成 tokens，不将其强制改成按编号排序的树。它们都只是位置候选，不是机械装配结构。

Puppeteer 的 `save_utils.py` 同时导入 OpenGL 渲染器。适配通过 AST 从已校验的官方文件载入四个完整的数值后处理函数，保留其原函数体、默认阈值和连通性修复，避免为了数值后处理创建窗口或依赖服务器图形上下文。未替换后处理算法。

## 环境及离线运行

AutoDL 根目录：`/root/autodl-tmp/rigModel`。两个独立 Python 3.11 venv：`/root/autodl-tmp/envs/riganything`、`/root/autodl-tmp/envs/puppeteer`；兼容的大依赖从已有 UniRig 环境只读链接。先安装新版本包，再链接其余依赖，避免修改旧环境。

Puppeteer 固定官方要求的 Transformers 4.46.1 / Accelerate 0.28.0，但复用本机 PyTorch 2.4.1+cu124 / FlashAttention 2.7.3；这与论文的 PyTorch 2.1.1 / CUDA 11.8 不同。CPU 预检只能验证导入和参数结构，CUDA 内核及数值差异须在开卡后的单例烟测中验证。

本轮 Puppeteer SDF 输入使用 `prepare_local.py` 在本地两个 CPU 进程生成后回传，避免 AutoDL 半核 CPU 的长时间预处理。核心仍直接调用 `prepare.py` 中同一函数和官方 `MeshProcessor`；固定 NumPy 1.26.4、trimesh 4.2.3、mesh2sdf 1.1.0、scikit-image 0.24.0。本地 Python 为 3.12，AutoDL 推理环境为 3.11。已和 AutoDL 先行完成的 13 例逐数组对比，点坐标、法线与数值变换全部完全一致。

```bash
bash scripts/setup_rig_baselines.sh
/root/autodl-tmp/envs/unirig/bin/python scripts/download_assets.py \
  --lock configs/rig-baselines.assets.lock.json --endpoint https://hf-mirror.com
for model in riganything puppeteer; do
  /root/autodl-tmp/envs/$model/bin/python experiments/rig_baselines/prepare.py --model "$model"
  /root/autodl-tmp/envs/$model/bin/python experiments/rig_baselines/preflight.py --model "$model"
done

# 只有用户确认 GPU 已开启后才运行，先单例再完整批次：
bash scripts/run_rig_baselines.sh riganything --case 0001-antique-globe
bash scripts/run_rig_baselines.sh puppeteer --case 0001-antique-globe
bash scripts/run_rig_baselines.sh riganything
bash scripts/run_rig_baselines.sh puppeteer
```

大 checkpoint 用 mmap 加载，先创建 meta 参数再严格装载，避免初始化一份随机大模型。RigAnything 的 CPU 构造只临时调整 attention 构造函数的设备参数，原位置编码和非持久缓存正常创建。Puppeteer 用同一官方结构构造点编码器，要求完整 skeleton checkpoint 严格包含其权重，避免二次加载后又被覆盖；官方 Michelangelo 权重仍单独下载、校验并保留。CPU 结构审计临时使用 eager 构造 Transformer，GPU 推理仍使用官方 FlashAttention；CPU 不运行该解码前向。

## 评估与局限

固定参考方向，计算最佳节点到参考轴线的垂距，再除以冻结 `child_scale_m`。位置阈值为 1% / 2%；失败或缺失预测留在全部 65 轴分母中。参考标签挑出的最佳节点为候选覆盖上限，不是自动关联精度。未融合 VLM 方向，也不推断绑骨拓扑等于机械连接关系。

不同模型的原生采样、量化和 marching-cubes 预处理不同，不能将此轮称为输入张量完全一致的受控模型消融。标签来自已有几何整理，训练数据与本批资产是否重合未排除。

RigAnything 代码和权重采用 Adobe Research License，限定非商业研究用途；Puppeteer 根许可为 Apache-2.0，其 Michelangelo 子模块的许可另行保留，不能把整个依赖树统一宣称为 Apache-2.0。所有权重、资产及运行结果保持在 Git 忽略目录。

验证入口：`python experiments/rig_baselines/test_contracts.py`。实际准备状态见 [RESULTS.md](RESULTS.md)。
