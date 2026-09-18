"""Compare actual frozen GPU runs; independently recompute every offset using cross products."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = ('unirig', 'riganything', 'puppeteer')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def offsets(points, origin, direction, scale):
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    return np.linalg.norm(np.cross(np.asarray(points)-origin, direction), axis=-1)/scale


def build(root, runs, output):
    original = root/'data/user25'
    refs, inputs = read(original/'reference_axes.json'), read(original/'inputs.json')
    assert len(refs) == 65 and len(inputs) == 25
    ids = {row['asset_id'] for row in inputs}
    frozen_sources = {r['asset_id']: r['source_sha256'] for r in inputs}
    records, predictions, provenance, performance, scored_rows = {}, {}, {}, {}, {}
    for model, run in runs.items():
        summary, config, audit = read(run/'summary.json'), read(run/'run_config.json'), read(run/'audit.json')
        assert summary['completed'] == summary['planned'] == 25 and set(config['cases']) == ids
        assert {r['asset_id'] for r in summary['records']} == ids
        assert audit['status'] == 'PASS', 'Resolve coordinate/export audit failures before publishing the report'
        data = root/('data/user25' if model == 'unirig' else f'data/user25_{model}')
        assert sha(data/'reference_axes.json') == sha(original/'reference_axes.json')
        assert sha(data/'inputs.json') == config['inputs_sha256']
        assert {r['asset_id']: r['source_sha256'] for r in read(data/'inputs.json')} == frozen_sources
        records[model] = {r['asset_id']: r for r in summary['records']}
        predictions[model] = {}
        scored_rows[model] = {r['sample_id']: r for r in csv.DictReader((run/'scores/per_axis.csv').open(encoding='utf-8-sig'))}
        assert set(scored_rows[model]) == {r['sample_id'] for r in refs}
        for asset, record in records[model].items():
            if record['status'] == 'success':
                prediction = read(run/asset/'candidates.json')
                assert prediction['source_sha256'] == frozen_sources[asset]
                predictions[model][asset] = prediction
        performance[model] = dict(successes=summary['successes'], planned=25,
            total_nodes=sum(r.get('joints', 0) for r in summary['records']),
            seconds_sum=sum(r['seconds'] for r in summary['records']),
            seconds_median=float(np.median([r['seconds'] for r in summary['records']])),
            peak_allocated_gib=max(r.get('peak_allocated_gib', 0) for r in summary['records']))
        provenance[model] = dict(run=str(run), run_config_sha256=sha(run/'run_config.json'),
            source_identity=config['source'], started_at=config['started_at'], gpu=config['gpu'],
            inputs_sha256=config['inputs_sha256'])
    prior = read(runs['unirig']/'comparison.json')['baseline_records']
    axes = []
    for ref in refs:
        asset = ref['asset_key'].split('/')[-1]
        origin, direction = np.asarray(ref['origin_world']), np.asarray(ref['direction_world'])
        center = np.asarray(ref['center_world'])
        scale = ref['child_scale_m']
        errors = {'center': float(offsets(center, origin, direction, scale))}
        best = {}
        for model in MODELS:
            prediction = predictions[model].get(asset)
            values = offsets(prediction['joints_world'], origin, direction, scale) if prediction else None
            best[model] = int(np.argmin(values)) if values is not None else None
            errors[model] = float(values[best[model]]) if values is not None else None
            saved = scored_rows[model][ref['sample_id']]
            if values is not None:
                np.testing.assert_allclose(errors[model], float(saved['oracle_offset_fraction']), atol=1e-10, rtol=1e-8)
                assert best[model] == int(saved['oracle_index'])
            else:
                assert saved['status'] != 'scored'
            np.testing.assert_allclose(errors['center'], float(saved['center_offset_fraction']), atol=1e-10)
        old = prior[ref['sample_id']]['result']
        pivot, vector = np.asarray(old['pivot']), np.asarray(old['axis'])
        vector = vector/np.linalg.norm(vector)
        pivot = pivot + np.dot(center-pivot, vector)*vector
        errors['v29'] = float(offsets(pivot, origin, direction, scale))
        axes.append(dict(sample_id=ref['sample_id'], asset_id=asset, name=ref['joint_name'],
                         errors=errors, oracle_indices=best, origin_world=origin.tolist(),
                         direction_world=direction.tolist(), child_center_world=center.tolist(), child_scale_m=scale))
    metrics = {}
    for key in (*MODELS, 'center', 'v29'):
        values = [a['errors'][key] for a in axes if a['errors'][key] is not None]
        metrics[key] = dict(available=len(values), pass_1pct=sum(v <= .01 for v in values),
                           pass_2pct=sum(v <= .02 for v in values), median_pct=float(np.median(values)*100) if values else None)
    hits = {key: {a['sample_id'] for a in axes if a['errors'][key] is not None and a['errors'][key] <= .01}
            for key in (*MODELS, 'center', 'v29')}
    union = set.union(*(hits[key] for key in MODELS))
    complement = {key: sorted(hits[key]-hits['v29']) for key in MODELS}
    result = dict(planned=65, metrics=metrics, performance=performance, provenance=provenance,
                  gains_over_v29_1pct=complement, model_union_pass_1pct=len(union),
                  union_gains_over_v29_1pct=sorted(union-hits['v29']), independent_cross_product_verification='PASS',
                  note='Fixed reference direction; oracle nodes are selected with reference labels, not automatic association accuracy.')
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for row in inputs:
        asset = row['asset_id']
        frame = read(original/row['frame'])
        center, scale = np.asarray(frame['center_world']), frame['scale_m']
        with np.load(original/row['points'], allow_pickle=False) as points:
            cloud = points['vertices'][np.linspace(0, len(points['vertices'])-1, 2200, dtype=int)]
        skeletons = {}
        for model in MODELS:
            prediction = predictions[model].get(asset)
            if prediction:
                joints = ((np.asarray(prediction['joints_world'])-center)/scale).tolist()
                bones = ([[p, i] for i, p in enumerate(prediction['parents']) if p is not None and p >= 0]
                         if model == 'unirig' else prediction['bones'])
            else:
                joints, bones = [], []
            links = {name: Path(os.path.relpath(runs[model]/asset/name, output)).as_posix()
                     for name in ('candidates.json', 'skeleton_world.glb')}
            skeletons[model] = dict(joints=joints, bones=bones, status=records[model][asset], links=links)
        display_axes = []
        for axis in [a for a in axes if a['asset_id'] == asset]:
            direction = np.asarray(axis['direction_world']); direction /= np.linalg.norm(direction)
            origin = np.asarray(axis['origin_world'])
            anchor = origin + np.dot(np.asarray(axis['child_center_world'])-origin, direction)*direction
            display_axes.append(dict(name=axis['name'], errors=axis['errors'], oracle_indices=axis['oracle_indices'],
                                     origin=((anchor-center)/scale).tolist(), direction=direction.tolist(),
                                     length=axis['child_scale_m']/scale))
        cases.append(dict(id=asset, name=row['case_name'], cloud=cloud.round(6).tolist(), skeletons=skeletons, axes=display_axes))
    write(output/'comparison.json', dict(summary=result, axes=axes))
    payload = json.dumps(dict(summary=result, cases=cases), ensure_ascii=False, allow_nan=False).replace('</', '<\\/')
    template = Path(__file__).with_name('report_template.html').read_text(encoding='utf-8')
    (output/'report.html').write_text(template.replace('__PAYLOAD__', payload), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    for model in MODELS:
        parser.add_argument('--'+model, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.root.resolve(), {m: getattr(args, m).resolve() for m in MODELS}, args.output.resolve())
