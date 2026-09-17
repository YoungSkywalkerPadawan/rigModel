"""Run the same Puppeteer preprocessing on local CPU workers, retaining canonical server paths."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from prepare import assemble, prepare_one
from shared import read, sha, write


def one_case(row, source, output):
    start = time.monotonic()
    folder = output/'inputs'/row['asset_id']
    # Only the read path changes; source geometry and all numerical preprocessing are verified/shared.
    local = dict(row, source_glb=str(source/row['asset_id']/'scene.glb'))
    arrays, frame, detail = prepare_one('puppeteer', assemble(local), 12345)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/'points.npz', **arrays)
    write(folder/'frame.json', frame)
    prepared = dict(asset_id=row['asset_id'], case_name=row['case_name'], source_glb=row['source_glb'],
                    source_sha256=row['source_sha256'], model='puppeteer', seed=12345, input_points=8192,
                    points=(folder/'points.npz').relative_to(output).as_posix(), points_sha256=sha(folder/'points.npz'),
                    frame=(folder/'frame.json').relative_to(output).as_posix(), frame_sha256=sha(folder/'frame.json'),
                    preprocessing=detail, preparation_seconds=time.monotonic()-start,
                    preparation_platform='local CPU', preparation_python=sys.version)
    write(folder/'prepared.json', prepared)
    return prepared


def prepare_local(source, original, output, workers):
    frozen = read(original/'READY.json')
    for name in ('inputs.json', 'reference_axes.json'):
        assert sha(original/name) == frozen['files'][name]
    original_rows = read(original/'inputs.json')
    assert len(original_rows) == 25 and len(read(original/'reference_axes.json')) == 65
    rows = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(one_case, row, source, output) for row in original_rows]
        for future in as_completed(pending):
            row = future.result()
            rows[row['asset_id']] = row
            print('PREPARED_LOCAL', len(rows), '/25', row['case_name'], round(row['preparation_seconds'], 2), flush=True)
    ordered = [rows[row['asset_id']] for row in original_rows]
    write(output/'inputs.json', ordered)
    shutil.copyfile(original/'reference_axes.json', output/'reference_axes.json')
    write(output/'READY.json', dict(status='INPUTS_READY', model='puppeteer', assets=25, reference_axes=65,
          source_inputs_sha256=sha(original/'inputs.json'), seed=12345, labels_in_points=False,
          preparation_platform='local CPU', workers=workers,
          files={name: sha(output/name) for name in ['inputs.json', 'reference_axes.json']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('Use 1 to 8 bounded local CPU workers')
    prepare_local(args.source_root.resolve(), args.original.resolve(), args.output.resolve(), args.workers)
