"""GPU-only skeleton candidates. Never loads mechanical reference axes."""
import argparse
from datetime import datetime, timezone
import os
import random
import time
import traceback
from pathlib import Path
import numpy as np
from common import ROOT, read, write, sha, restore, validate_skeleton, source_identity


def run(data, output, case=None):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU is not enabled. Ask the user to enable AutoDL GPU before running.')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('The official bf16 inference configuration needs a compatible GPU.')
    from runtime import construct, generation_config
    from export import export_overlay
    rows = read(data/'inputs.json')
    if case:
        rows = [row for row in rows if case in (row['asset_id'], row['case_name'])]
        if len(rows) != 1:
            raise ValueError('Case must match exactly one case_name or asset_id')
    prepared = read(data/'READY.json')
    if sha(data/'inputs.json') != prepared['files']['inputs.json']:
        raise ValueError('Input manifest changed after preparation')
    import sys
    sys.path.insert(0, str(ROOT/'scripts'))
    from download_assets import verify
    for item in read(ROOT/'configs/assets.lock.json')['files']:
        if not verify(ROOT/item['destination'], item):
            raise ValueError('Missing or corrupted model asset')
    # New run directory only: preserve prior runs and prevent mixed configurations.
    output.mkdir(parents=True, exist_ok=False)
    identity = dict(source=source_identity(), inputs_sha256=sha(data/'inputs.json'),
                    assets_lock_sha256=sha(ROOT/'configs/assets.lock.json'), generation=generation_config(),
                    seed=12345, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                    started_at=datetime.now(timezone.utc).isoformat(), cases=[r['asset_id'] for r in rows])
    write(output/'run_config.json', identity)
    run_hash = sha(output/'run_config.json')
    torch.set_float32_matmul_precision('high')
    model, audit = construct()
    write(output/'model_audit.json', audit)
    records = []
    for row in rows:
        started = time.monotonic()
        directory = output/row['asset_id']
        directory.mkdir()
        result = dict(asset_id=row['asset_id'], case_name=row['case_name'], status='running', run_identity=run_hash)
        try:
            for key in ['points', 'frame']:
                if sha(data/row[key]) != row[key+'_sha256']:
                    raise ValueError('Input file changed: '+key)
            with np.load(data/row['points'], allow_pickle=False) as arrays:
                vertices = torch.from_numpy(arrays['vertices']).to('cuda')
                normals = torch.from_numpy(arrays['normals']).to('cuda')
            random.seed(row['seed'])
            np.random.seed(row['seed'])
            torch.manual_seed(row['seed'])
            torch.cuda.manual_seed_all(row['seed'])
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model.predict_step(dict(vertices=vertices, normals=normals,
                    path=[row['case_name']], cls=['articulationxl'], generate_kwargs=generation_config()))[0]
            torch.cuda.synchronize()
            validate_skeleton(prediction.joints, prediction.parents)
            # Official detokenize validates EOS and strips BOS/EOS from .tokens.
            frame = read(data/row['frame'])
            joints = restore(prediction.joints, frame)
            parents = [None if p is None else int(p) for p in prediction.parents]
            np.savez_compressed(directory/'skeleton_model.npz', joints=prediction.joints,
                parents=np.array([-1 if p is None else p for p in parents]), tokens=prediction.tokens)
            write(directory/'candidates.json', dict(frame='source_world_metres_Z_up',
                joints_world=joints.tolist(), parents=parents, names=prediction.names,
                source_sha256=row['source_sha256'], normalization=frame,
                role='Unassociated rig nodes, not predicted mechanical joint identities'))
            export_overlay(row['source_glb'], joints, parents, directory/'skeleton_world.glb')
            result.update(status='success', joints=len(joints), tokens=len(prediction.tokens),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
        except Exception as exc:
            result.update(status='failed', error=type(exc).__name__+': '+str(exc))
            (directory/'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        result['seconds'] = time.monotonic()-started
        write(directory/'status.json', result)
        records.append(result)
        write(output/'summary.json', dict(planned=len(rows), completed=len(records),
              successes=sum(r['status']=='success' for r in records), records=records))
        print(result, flush=True)
        torch.cuda.empty_cache()
    return 0 if all(r['status']=='success' for r in records) else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT/'data/user25')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case')
    a = parser.parse_args()
    raise SystemExit(run(a.data.resolve(), a.output.resolve(), a.case))
