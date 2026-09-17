"""CPU-only checks. READY_FOR_GPU does not mean a GPU forward pass has happened."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from shared import ROOT, normalize_world, read, restore, sha, source_identity, write


def verify_asset(item):
    path = ROOT/item['destination']
    assert path.stat().st_size == item['bytes'], str(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    assert digest == item['sha256'], str(path)


def preflight(name):
    data = ROOT/f'data/user25_{name}'
    for project in read(ROOT/'UPSTREAM-rig-baselines.json')['projects']:
        for relative, identity in project['files'].items():
            assert sha(ROOT/project['destination']/relative) == identity['sha256'], relative
    assets = [a for a in read(ROOT/'configs/rig-baselines.assets.lock.json')['files'] if f'models/{name}/' in a['destination']]
    for item in assets:
        verify_asset(item)
        print('VERIFIED', item['destination'], flush=True)
    original_assets = read(ROOT/'configs/assets.lock.json')['files']
    opt = next(a for a in original_assets if a['destination'].endswith('opt-350m/config.json'))
    content = (ROOT/opt['destination']).read_bytes()
    assert hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest() == opt['git_blob_sha1']
    manifest = read(data/'READY.json')
    for key, digest in manifest['files'].items():
        assert sha(data/key) == digest
    rows = read(data/'inputs.json')
    assert len(rows) == 25 and len(read(data/'reference_axes.json')) == 65
    for row in rows:
        assert row['model'] == name and sha(row['source_glb']) == row['source_sha256']
        for key in ('points', 'frame'):
            assert sha(data/row[key]) == row[key+'_sha256']
        frame = read(data/row['frame'])
        with np.load(data/row['points'], allow_pickle=False) as arrays:
            expected_points = 1024 if name == 'riganything' else 8192
            for key in ('encoder_points', 'vertices', 'normals'):
                assert arrays[key].shape == (expected_points, 3) and np.isfinite(arrays[key]).all()
            np.testing.assert_allclose(normalize_world(restore(arrays['vertices'], frame), frame), arrays['vertices'], atol=1e-8)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       HF_HUB_DISABLE_TELEMETRY='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True, env=environment)
    audit_file = ROOT/f'setup/{name}-model-audit.json'
    subprocess.run([sys.executable, str(Path(__file__).with_name('runtime.py')), '--model', name,
                    '--audit-output', str(audit_file)], check=True, env=environment)
    audit = read(audit_file)
    assert not audit['gpu_inference_run'] and audit['all_parameter_shapes_match']
    result = dict(status='READY_FOR_GPU', model=name, inputs=25, reference_axes=65,
                  verified_assets=len(assets), asset_bytes=sum(a['bytes'] for a in assets),
                  source_identity=source_identity(), inputs_sha256=sha(data/'inputs.json'),
                  asset_lock_sha256=sha(ROOT/'configs/rig-baselines.assets.lock.json'),
                  python=sys.version, model_structure=audit, gpu_inference_run=False)
    write(ROOT/f'setup/{name}-preflight.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['riganything', 'puppeteer'], required=True)
    args = parser.parse_args()
    preflight(args.model)
