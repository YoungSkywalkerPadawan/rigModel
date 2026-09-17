"""Check output identity, model/world coordinates and exported candidate markers."""
import argparse
from pathlib import Path
import numpy as np
import trimesh
from common import ROOT, read, write, sha, restore, validate_skeleton


def audit(data, run):
    inputs = {r['asset_id']: r for r in read(data/'inputs.json')}
    configuration = read(run/'run_config.json')
    summary = read(run/'summary.json')
    assert summary['completed'] == summary['planned'] == len(configuration['cases'])
    assert {r['asset_id'] for r in summary['records']} == set(configuration['cases'])
    errors = []; checked = []
    for record in summary['records']:
        identity = record['asset_id']
        if record['status'] != 'success':
            errors.append(dict(asset_id=identity, error=record.get('error')))
            continue
        row = inputs[identity]
        folder = run/identity
        assert record['run_identity'] == sha(run/'run_config.json')
        output = read(folder/'candidates.json')
        assert output['source_sha256'] == row['source_sha256'] == sha(row['source_glb'])
        joints = np.array(output['joints_world'])
        validate_skeleton(joints, output['parents'])
        with np.load(folder/'skeleton_model.npz', allow_pickle=False) as arrays:
            np.testing.assert_allclose(restore(arrays['joints'], output['normalization']), joints, atol=1e-10)
        scene = trimesh.load_scene(folder/'skeleton_world.glb', process=False)
        original = trimesh.load_scene(row['source_glb'], process=False)
        for node in original.graph.nodes_geometry:
            before, old = original.graph[node]
            after, new = scene.graph[node]
            np.testing.assert_allclose(before, after, atol=1e-7)
            np.testing.assert_allclose(original.geometry[old].vertices, scene.geometry[new].vertices, atol=1e-7)
            np.testing.assert_array_equal(original.geometry[old].faces, scene.geometry[new].faces)
        for index, point in enumerate(joints):
            transform, geometry = scene.graph[f'rig_candidate_{index}']
            marker = trimesh.transform_points(scene.geometry[geometry].vertices, transform).mean(axis=0)
            np.testing.assert_allclose(marker, point, atol=1e-7, rtol=1e-6)
        checked.append(dict(asset_id=identity, joints=len(joints), glb_bytes=(folder/'skeleton_world.glb').stat().st_size))
    result = dict(status='PASS' if not errors else 'HAS_INFERENCE_FAILURES', verified=len(checked),
                  planned=summary['planned'], checked=checked, inference_failures=errors)
    write(run/'audit.json', result)
    print(result, flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT/'data/user25')
    parser.add_argument('--run', type=Path, required=True)
    a = parser.parse_args()
    result = audit(a.data.resolve(), a.run.resolve())
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
