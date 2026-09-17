# UniRig 接口核查

2026-09-17。官方源码固定于 `6793c6640ff01c8fb389f3993434124bb43d2933`。

- 网络和骨架 tokenizer 可直接调用；`DetokenizeOutput.joints` 为三维节点，`parents` 为骨架父节点索引，与机械部件 ID 无对应保证。
- 官方 `ARWriter` 的 user mode 只导出 FBX，不写骨架 NPZ。适配入口直接保存 `predict_step` 输出，避免只拿到最终文件而丢失原始候选。
- `AugmentAffine` 在推理时根据网格 AABB 做统一平移缩放；适配端记录该变换并显式还原。
- `SamplerMix` 当前源码按三角形叉积平方范数抽样；本轮直接使用该源码，不将它静默改为通常的面积权重。
- 官方骨架配置需要 FlashAttention 2，几何编码器内部也使用 PyTorch SDPA。CPU 检查时仅以 eager 模式构造 meta 网络核对形状；GPU 运行仍使用官方 FlashAttention 配置。
- 模型权重约 1.44 GB；独立 Fourier buffers 不在 checkpoint 中，使用 `accelerate.init_empty_weights(include_buffers=False)` 保留其确定性初始化，防止全 meta 构造丢失这些 buffer。
- 当前公开骨架权重训练于 Articulation-XL2.0，不能把论文其他训练配置的成绩作为该权重的机械轴精度。

来源：[UniRig 官方仓库](https://github.com/VAST-AI-Research/UniRig)、[官方模型卡](https://huggingface.co/VAST-AI/UniRig)。
