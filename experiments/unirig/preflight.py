"""CPU readiness audit. No model forward or CUDA inference is performed."""
import argparse
import importlib
import importlib.metadata
import subprocess
import sys
from pathlib import Path
import numpy as np
from common import ROOT, read, write, sha, source_identity


def main(data, output):
    sys.path.insert(0, str(ROOT/'scripts'))
    from download_assets import verify
    for asset in read(ROOT/'configs/assets.lock.json')['files']:
        if not verify(ROOT/asset['destination'], asset):
            raise ValueError('Missing or incorrect model asset: ' + asset['destination'])
    ready = read(data/'READY.json')
    for name, digest in ready['files'].items():
        if sha(data/name) != digest:
            raise ValueError('Prepared metadata hash mismatch: '+name)
    rows = read(data/'inputs.json')
    for row in rows:
        for field in ['points', 'frame']:
            if sha(data/row[field]) != row[field+'_sha256']:
                raise ValueError('Prepared input hash mismatch')
        if sha(row['source_glb']) != row['source_sha256']:
            raise ValueError('Source mesh changed')
        with np.load(data/row['points'], allow_pickle=False) as points:
            for name in ['vertices', 'normals']:
                if points[name].shape != (65536, 3) or not np.isfinite(points[name]).all():
                    raise ValueError('Invalid input point arrays')
    versions = {}
    for package, module in [('torch','torch'), ('transformers','transformers'), ('flash-attn','flash_attn'),
                            ('torch-scatter','torch_scatter'), ('torch-cluster','torch_cluster'), ('spconv-cu120','spconv.pytorch'),
                            ('lightning','lightning'), ('timm','timm'), ('numpy','numpy')]:
        subprocess.run([sys.executable, '-c', f'import importlib; importlib.import_module({module!r})'], check=True)
        versions[package] = importlib.metadata.version(package)
        print('IMPORT_OK', package, versions[package], flush=True)
    import torch
    from runtime import construct
    audit = construct(cpu_audit=True)
    result = dict(status='READY_FOR_GPU', assets=len(rows), reference_axes=ready['reference_axes'],
                  model=audit, versions=versions, source_identity=source_identity(), inputs_sha256=sha(data/'inputs.json'),
                  gpu_available=torch.cuda.is_available(), gpu_inference_run=False,
                  note='CPU mmap/meta architecture audit only. CUDA kernels and full model forward await GPU.')
    write(output, result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT/'data/user25')
    parser.add_argument('--output', type=Path, default=ROOT/'setup/preflight.json')
    a = parser.parse_args()
    main(a.data.resolve(), a.output.resolve())
