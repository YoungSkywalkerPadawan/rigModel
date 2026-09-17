import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / 'third_party/UniRig'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def restore(points, metadata):
    return np.asarray(points, dtype=np.float64) * metadata['scale_m'] + np.asarray(metadata['center_world'])


def point_axis_errors(points, origin, direction):
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    delta = np.asarray(points, dtype=np.float64) - np.asarray(origin)
    return np.linalg.norm(delta - (delta @ direction)[..., None] * direction, axis=-1)


def validate_skeleton(joints, parents):
    joints = np.asarray(joints)
    if joints.ndim != 2 or joints.shape[1] != 3 or len(joints) == 0 or not np.isfinite(joints).all():
        raise ValueError('No finite skeleton joints')
    if len(parents) != len(joints) or parents[0] not in (None, -1):
        raise ValueError('Invalid skeleton root or parent count')
    for index, parent in enumerate(parents[1:], 1):
        if parent is None or not isinstance(parent, (int, np.integer)) or not 0 <= parent < index:
            raise ValueError('Invalid skeleton topology')


def source_identity():
    paths = [ROOT / 'configs/assets.lock.json', ROOT / 'UPSTREAM.json']
    paths += sorted((ROOT / 'experiments/unirig').glob('*.py'))
    paths += sorted((UPSTREAM / 'src').rglob('*.py'))
    paths += sorted((UPSTREAM / 'configs').rglob('*.yaml'))
    return hashlib.sha256(json.dumps({str(p.relative_to(ROOT)): sha(p) for p in paths}, sort_keys=True).encode()).hexdigest()
