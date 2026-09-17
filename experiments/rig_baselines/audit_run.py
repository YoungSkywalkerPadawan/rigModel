"""Verify actual GPU candidates against their raw decoder arrays and original source scenes."""
import argparse
from pathlib import Path
import numpy as np
import trimesh
from shared import ROOT, read, restore, sha, validate_graph, write


def audit(name, run):
    data = ROOT/f'data/user25_{name}'
    rows = {r['asset_id']: r for r in read(data/'inputs.json')}
    summary, config = read(run/'summary.json'), read(run/'run_config.json')
    assert summary['completed'] == summary['planned'] == len(config['cases'])
    assert {r['asset_id'] for r in summary['records']} == set(config['cases'])
    verified, failures = [], []
    for status in summary['records']:
        identity = status['asset_id']
        if status['status'] != 'success':
            failures.append(identity)
            continue
        row = rows[identity]
        folder = run/identity
        candidates = read(folder/'candidates.json')
        assert status['run_identity'] == sha(run/'run_config.json')
        assert sha(row['source_glb']) == row['source_sha256'] == candidates['source_sha256']
        world, bones = np.asarray(candidates['joints_world']), np.asarray(candidates['bones'], dtype=int).reshape(-1, 2)
        validate_graph(world, bones)
        with np.load(folder/'skeleton_model.npz', allow_pickle=False) as raw:
            np.testing.assert_allclose(restore(raw['joints'], candidates['normalization']), world, atol=1e-10)
            np.testing.assert_array_equal(raw['bones'], bones)
        original = trimesh.load(row['source_glb'], force='scene', process=False)
        scene = trimesh.load(folder/'skeleton_world.glb', force='scene', process=False)
        for node in original.graph.nodes_geometry:
            before, old = original.graph[node]
            after, new = scene.graph[node]
            np.testing.assert_allclose(before, after, atol=1e-7)
            np.testing.assert_allclose(original.geometry[old].vertices, scene.geometry[new].vertices, atol=1e-7)
            np.testing.assert_array_equal(original.geometry[old].faces, scene.geometry[new].faces)
        for index, point in enumerate(world):
            transform, geometry = scene.graph[f'rig_candidate_{index}']
            actual = trimesh.transform_points(scene.geometry[geometry].vertices, transform).mean(axis=0)
            np.testing.assert_allclose(actual, point, atol=1e-7, rtol=1e-6)
        verified.append(identity)
    result = dict(status='PASS' if not failures else 'HAS_INFERENCE_FAILURES', model=name,
                  verified=len(verified), planned=summary['planned'], verified_assets=verified, failed_assets=failures)
    write(run/'audit.json', result)
    print(result)
    return 0 if not failures else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['riganything', 'puppeteer'], required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(audit(args.model, args.run.resolve()))
