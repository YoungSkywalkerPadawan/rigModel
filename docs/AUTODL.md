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
