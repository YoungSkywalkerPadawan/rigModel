"""Offline oracle coverage, NOT automatic mechanical joint prediction accuracy."""
import argparse
import csv
from pathlib import Path
import numpy as np
from common import ROOT, read, write, sha, point_axis_errors


def score(data, predictions, output):
    refs = read(data/'reference_axes.json')
    manifest = read(data/'READY.json')
    if sha(data/'reference_axes.json') != manifest['files']['reference_axes.json']:
        raise ValueError('Reference hash mismatch')
    rows = []
    for ref in refs:
        # Existing frozen asset_key is source-prefixed; resolve only exact asset IDs.
        asset_id = ref.get('asset_id') or ref['asset_key'].split('/')[-1]
        row = dict(sample_id=ref['sample_id'], asset_id=asset_id, joint_name=ref['joint_name'],
                   status='missing_prediction', oracle_index=None, oracle_offset_fraction=None,
                   oracle_pass_1pct=False, oracle_pass_2pct=False,
                   center_offset_fraction=float(point_axis_errors(np.array(ref['center_world']),
                     ref['origin_world'],ref['direction_world']))/ref['child_scale_m'])
        candidate_file = predictions/asset_id/'candidates.json'
        status_file = predictions/asset_id/'status.json'
        if candidate_file.exists() and status_file.exists() and read(status_file)['status']=='success':
            prediction = read(candidate_file)
            joints = np.array(prediction['joints_world'])
            errors = point_axis_errors(joints, ref['origin_world'], ref['direction_world'])/ref['child_scale_m']
            best = int(np.argmin(errors))
            row.update(status='scored', oracle_index=best, oracle_offset_fraction=float(errors[best]),
                       oracle_pass_1pct=bool(errors[best]<=0.01), oracle_pass_2pct=bool(errors[best]<=0.02))
        rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'per_axis.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = dict(metric='Oracle best node at fixed reference direction; NOT deployable accuracy', planned=len(rows),
                   scored=sum(r['status']=='scored' for r in rows),
                   oracle_pass_1pct=sum(r['oracle_pass_1pct'] for r in rows),
                   oracle_pass_2pct=sum(r['oracle_pass_2pct'] for r in rows),
                   center_pass_1pct=sum(r['center_offset_fraction']<=0.01 for r in rows))
    write(output/'summary.json', summary)
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT/'data/user25')
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    score(a.data.resolve(), a.predictions.resolve(), a.output.resolve())
