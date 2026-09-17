"""Data contracts shared by the two new skeleton baselines; no model imports."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORLD_TO_MODEL = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def restore(points, frame):
    return (np.asarray(points, dtype=np.float64) * frame['scale_m']) @ np.asarray(frame['model_to_world_rotation']) + frame['center_world']


def normalize_world(points, frame):
    return (np.asarray(points, dtype=np.float64) - frame['center_world']) @ np.asarray(frame['model_to_world_rotation']).T / frame['scale_m']


def validate_graph(joints, bones):
    joints, bones = np.asarray(joints), np.asarray(bones)
    if joints.ndim != 2 or joints.shape[1] != 3 or len(joints) == 0 or not np.isfinite(joints).all():
        raise ValueError('No finite joint candidates')
    if bones.ndim != 2 or bones.shape[1] != 2 or not np.issubdtype(bones.dtype, np.integer):
        raise ValueError('Invalid bone array')
    if len(bones) and (bones.min() < 0 or bones.max() >= len(joints) or np.any(bones[:, 0] == bones[:, 1])):
        raise ValueError('Invalid bone endpoints')


def source_identity():
    paths = [ROOT/'UPSTREAM-rig-baselines.json', ROOT/'configs/rig-baselines.assets.lock.json']
    paths += sorted((ROOT/'experiments/rig_baselines').glob('*.py'))
    for project in read(ROOT/'UPSTREAM-rig-baselines.json')['projects']:
        paths += [ROOT/project['destination']/name for name in project['files']]
    return hashlib.sha256(json.dumps({p.relative_to(ROOT).as_posix(): sha(p) for p in paths}, sort_keys=True).encode()).hexdigest()
