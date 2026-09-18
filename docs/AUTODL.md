# AutoDL 准备与运行

本轮目录 `/root/autodl-tmp/rigModel`；Python 为 `/root/autodl-tmp/envs/unirig/bin/python`。服务器连接凭据由用户在会话中提供，不写进脚本或仓库。

## 准备

`scripts/setup_autodl.sh` 复用既有 `/root/autodl-tmp/envs/uniphysgen` 的兼容 CUDA 依赖，在独立 venv 中固定 UniRig 依赖。共享包只从旧环境读取，新版本安装到新环境；不要删除作为共享依赖来源的旧环境。实际完整版本表在服务器 `setup/environment.freeze.txt`。

公开权重和 OPT 配置按 `configs/assets.lock.json` 固定 revision 下载，前者验证官方 LFS SHA-256，后者验证 Git blob SHA-1。下载可用 `HF_ENDPOINT=https://hf-mirror.com`；推理强制离线，不发送 Hub token。

源码和脚本通过独立代码包上传；原数据位于 `/root/autodl-tmp/continuous-eval-20260914/source/user`，不重复上传。准备输入输出位于 `data/user25`。原参考为 `/root/autodl-tmp/geometric-axis-eval-20260915/prepared/ground_truth.json`，筛出 65 条自有轴。

```bash
cd /root/autodl-tmp/rigModel
bash scripts/setup_autodl.sh
/root/autodl-tmp/envs/unirig/bin/python experiments/unirig/prepare.py \
  --source /root/autodl-tmp/continuous-eval-20260914/source/user \
  --reference /root/autodl-tmp/geometric-axis-eval-20260915/prepared/ground_truth.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/autodl-tmp/envs/unirig/bin/python experiments/unirig/preflight.py
```

`setup/preflight.json` 的 `READY_FOR_GPU` 仅代表 CPU 可验证项目通过，不代表已通过 GPU 前向。失败需看实际日志，不能以文件存在判断成功。

2 GiB CPU 实例中，哈希校验会缓存整个 1.44 GB checkpoint。下载验证函数在校验后对该文件释放干净页缓存，再运行依赖导入和 mmap/meta 形状审计。不会清理其他项目文件或系统全局缓存。可选 Open3D/Blender 导出不在本骨架适配环境中；叠加 GLB 使用 trimesh。

## GPU 启动

用户确认开卡后，先用一个物体做完整前向、坐标导出和 CUDA 内核验证，再跑全部。未开卡时运行入口会明确报错，不会回退到 CPU 推理。

```bash
bash scripts/run_gpu.sh --case 0001-antique-globe
bash scripts/run_gpu.sh
```

每次使用独立时间戳目录，已有运行不会覆盖。单例失败有错误文件；全量运行逐例继续，汇总保存每次成功和失败。脚本默认调用独立离线 oracle 评分。预测文件与参考标签相互独立。

本轮不自动开 GPU、不创建定时任务，也不在等待期间消耗 GPU 推理资源。

## 运行后核验与报告

2026-09-17 已完成单例及全部 25 例运行；无需为查看结果重新执行 GPU 推理。以下步骤只读取既有输出，在 CPU 上执行：

```bash
cd /root/autodl-tmp/rigModel
run_dir=outputs/unirig_user25_20260917T103010Z
/root/autodl-tmp/envs/unirig/bin/python experiments/unirig/audit_run.py --run "$run_dir"
/root/autodl-tmp/envs/unirig/bin/python experiments/unirig/build_report.py --run "$run_dir" \
  --v29-records /root/autodl-tmp/geometric-axis-eval-20260915/runs/cpu_v1/generic_v29/records
```

`audit_run.py` 需要原始场景仍在输入清单记录的位置；`build_report.py` 需要准备好的点云、变换及标签。生成的 `report.html` 自包含点云和骨架，可离线查看；相对链接的 GLB/JSON 需与每例目录一起保存。

归档 `setup/unirig_user25_gpu_20260917.tar.gz` 含完整输出及逐文件哈希，未包含大权重和完整预处理点云。具体指标和归档指纹见 [实验记录](../experiments/unirig/RESULTS.md)。

## RigAnything / Puppeteer 准备与启动

2026-09-18 已完成两模型单例和完整 25 例，全部通过 GPU 前向及输出审计。当前只需查看已有结果，无需再次开 GPU。三模型交互报告生成命令见 [统一对比说明](../experiments/rig_comparison/README.md)，结果及归档指纹见 [GPU 记录](../experiments/rig_baselines/RESULTS.md#2026-09-18gpu-推理与三模型对比)。以下命令保留用于后续复现。

新增环境分别位于 `/root/autodl-tmp/envs/riganything`、`/root/autodl-tmp/envs/puppeteer`，从现有 UniRig 环境只读链接兼容依赖。源码和三份权重固定版本及 SHA-256，见 `UPSTREAM-rig-baselines.json` 和 `configs/rig-baselines.assets.lock.json`。依赖实际版本保存在 `setup/riganything-environment.freeze.txt`、`setup/puppeteer-environment.freeze.txt`。

```bash
cd /root/autodl-tmp/rigModel
bash scripts/setup_rig_baselines.sh
/root/autodl-tmp/envs/unirig/bin/python scripts/download_assets.py \
  --lock configs/rig-baselines.assets.lock.json --endpoint https://hf-mirror.com
for model in riganything puppeteer; do
  /root/autodl-tmp/envs/$model/bin/python experiments/rig_baselines/prepare.py --model "$model"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/autodl-tmp/envs/$model/bin/python \
    experiments/rig_baselines/preflight.py --model "$model"
done
```

本轮上述准备已完成；Puppeteer 的完整预处理输入由本地 CPU 生成并回传，无需重新跑 SDF。`data/user25_riganything` 和 `data/user25_puppeteer` 各有 25 例；`setup/<model>-preflight.json` 的状态、输入哈希及源码身份必须匹配当前文件。原 AutoDL 先完成的 13 例保存在 `data/user25_puppeteer_autodl_cpu_partial_20260917`，无需删除。

用户开启 GPU 后，依次运行两个单例，再运行完整批次：

```bash
bash scripts/run_rig_baselines.sh riganything --case 0001-antique-globe
bash scripts/run_rig_baselines.sh puppeteer --case 0001-antique-globe
bash scripts/run_rig_baselines.sh riganything
bash scripts/run_rig_baselines.sh puppeteer
```

入口强制离线、重新校验模型权重、检查冻结输入，并保存到新的 `outputs/<model>_user25_<UTC时间戳>`。实际推理后会调用共用离线位置评分和输出坐标/场景审计。CPU 预检不包含网络前向；本轮实际 GPU 前向已验证当前环境可执行，但不表示和另一套官方环境逐数值等价。详细适配及限制见 [说明](../experiments/rig_baselines/README.md)，已验证状态见 [记录](../experiments/rig_baselines/RESULTS.md)。
