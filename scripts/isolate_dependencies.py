"""Expose cached dependency packages without importing the old UniPhysGen app.

Creates links ONLY in the new UniRig environment; the source environment is read-only.
Existing UniRig-local packages take precedence. Re-run setup to rebuild dependencies.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import sys


def normalized(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def isolate(base, target):
    base, target = base.resolve(), target.resolve()
    if base == target or not (target/'pyvenv.cfg').is_file():
        raise ValueError('Target must be a distinct venv')
    source = base/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages'
    if not source.is_dir():
        raise ValueError('No matching Python site-packages in the cached base')
    destination = target/'lib'/source.parent.name/'site-packages'
    overrides = {normalized(d.metadata['Name']) for d in importlib.metadata.distributions(path=[str(destination)])}
    linked, skipped = [], []
    for item in source.iterdir():
        out = destination/item.name
        if out.exists() or out.is_symlink():
            continue
        if any(name in item.name.lower() for name in ['uniphysgen', 'open3d']) or item.suffix in ('.pth', '.egg-link') or item.name.startswith('__editable__'):
            skipped.append(item.name)
            continue
        if item.name.endswith('.dist-info'):
            name = normalized(item.name.split('-')[0])
            if name in overrides:
                skipped.append(item.name)
                continue
        out.symlink_to(item, target_is_directory=item.is_dir())
        linked.append(item.name)
    config = target/'pyvenv.cfg'
    config.write_text(config.read_text().replace('include-system-site-packages = true', 'include-system-site-packages = false'))
    print(json.dumps(dict(base=str(base), target=str(target), linked=linked, skipped=skipped), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    args = parser.parse_args()
    isolate(args.base, args.target)
