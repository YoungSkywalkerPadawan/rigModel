"""GPU-only inference on frozen, label-free inputs; independent scoring is a separate command."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import time
import traceback
import numpy as np
import trimesh
from shared import ROOT, read, restore, sha, source_identity, validate_graph, write


def export_scene(source, joints, bones, destination):
    scene = trimesh.load(source, force='scene', process=False)
    radius = max(float(np.max(scene.extents)) * .006, 1e-6)
    for index, point in enumerate(joints):
        marker = trimesh.creation.icosphere(subdivisions=1, radius=radius)
        marker.visual.vertex_colors = [255, 166, 70, 255]
        transform = np.eye(4); transform[:3, 3] = point
        scene.add_geometry(marker, node_name=f'rig_candidate_{index}', transform=transform)
    for index, (parent, child) in enumerate(bones):
        if np.linalg.norm(joints[parent]-joints[child]) < 1e-9:
            continue
        bone = trimesh.creation.cylinder(radius=radius*.3, segment=[joints[parent], joints[child]], sections=8)
        bone.visual.vertex_colors = [255, 190, 100, 255]
        scene.add_geometry(bone, node_name=f'rig_bone_{index}')
    scene.export(destination)


def run(model, output, case=None):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    import torch
    from runtime import Runtime
    if not torch.cuda.is_available():
        raise RuntimeError('GPU is not enabled; ask the user to enable it after CPU preparation is complete')
    data = ROOT/f'data/user25_{model}'
    ready = read(ROOT/f'setup/{model}-preflight.json')
    current_source = source_identity()
    if ready['status'] != 'READY_FOR_GPU' or ready['source_identity'] != current_source or ready['inputs_sha256'] != sha(data/'inputs.json'):
        raise ValueError('The prepared inputs or source changed; rerun CPU preflight')
    for asset in read(ROOT/'configs/rig-baselines.assets.lock.json')['files']:
        if f'models/{model}/' in asset['destination']:
            from preflight import verify_asset
            verify_asset(asset)
    rows = read(data/'inputs.json')
    if case:
        rows = [r for r in rows if r['case_name'] == case or r['asset_id'] == case]
        if len(rows) != 1:
            raise ValueError('Case must match exactly one frozen case name or asset id')
    output.mkdir(parents=True, exist_ok=False)
    configuration = dict(model=model, source=current_source, inputs_sha256=sha(data/'inputs.json'),
                         asset_lock_sha256=sha(ROOT/'configs/rig-baselines.assets.lock.json'), seed=12345,
                         started_at=datetime.now(timezone.utc).isoformat(), gpu=torch.cuda.get_device_name(0),
                         torch=torch.__version__, cases=[r['asset_id'] for r in rows],
                         generation='official RigAnything 64-joint maximum, 50 diffusion steps, BF16 autocast' if model == 'riganything'
                         else 'official Puppeteer diverse-pose, joint_token, seq_shuffle, 1 beam, 128 bins, FP16 autocast')
    write(output/'run_config.json', configuration)
    identity = sha(output/'run_config.json')
    runtime = Runtime(model)
    records = []
    for row in rows:
        start = time.monotonic()
        folder = output/row['asset_id']; folder.mkdir()
        record = dict(asset_id=row['asset_id'], case_name=row['case_name'], model=model, run_identity=identity)
        try:
            assert sha(row['source_glb']) == row['source_sha256']
            assert sha(data/row['points']) == row['points_sha256'] and sha(data/row['frame']) == row['frame_sha256']
            frame = read(data/row['frame'])
            with np.load(data/row['points'], allow_pickle=False) as loaded:
                arrays = {key: loaded[key] for key in loaded.files}
            torch.cuda.reset_peak_memory_stats()
            joints, bones, raw = runtime.predict(arrays)
            validate_graph(joints, bones)
            world = restore(joints, frame)
            np.savez_compressed(folder/'skeleton_model.npz', joints=joints, bones=bones, **raw)
            write(folder/'candidates.json', dict(model=model, asset_id=row['asset_id'], source_sha256=row['source_sha256'],
                 normalization=frame, joints_world=world.tolist(), bones=bones.tolist(),
                 note='Candidate nodes only; no mechanical part association or inferred reference direction'))
            export_scene(row['source_glb'], world, bones, folder/'skeleton_world.glb')
            torch.cuda.synchronize()
            record.update(status='success', joints=len(joints), bones=len(bones), peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        except Exception as error:
            record.update(status='failed', error=repr(error))
            (folder/'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        record['seconds'] = time.monotonic()-start
        write(folder/'status.json', record)
        records.append(record)
        write(output/'summary.json', dict(planned=len(rows), completed=len(records), successes=sum(r['status']=='success' for r in records), records=records))
        print(record, flush=True)
    return 0 if all(r['status']=='success' for r in records) else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['riganything', 'puppeteer'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case')
    args = parser.parse_args()
    raise SystemExit(run(args.model, args.output.resolve(), args.case))
