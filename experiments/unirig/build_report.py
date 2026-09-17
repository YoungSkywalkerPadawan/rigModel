"""Self-contained report of actual candidates, fixed-direction scores and baselines."""
import argparse
import csv
from pathlib import Path
import numpy as np
from common import ROOT, read, write, sha, point_axis_errors


def build(data, run, baseline=None):
    refs = read(data/'reference_axes.json')
    inputs = read(data/'inputs.json')
    records = read(run/'summary.json')['records']
    scores = {r['sample_id']: r for r in csv.DictReader((run/'scores/per_axis.csv').open(encoding='utf-8-sig'))}
    old = {}
    if baseline:
        for path in baseline.glob('*.json'):
            value = read(path)
            if value['sample_id'].startswith('user/'):
                value['record_sha256'] = sha(path)
                old[value['sample_id']] = value
    cases = []; comparison = []
    for row in inputs:
        asset_id = row['asset_id']
        status = next(r for r in records if r['asset_id']==asset_id)
        frame = read(data/row['frame'])
        center, scale = np.array(frame['center_world']), frame['scale_m']
        with np.load(data/row['points'], allow_pickle=False) as points:
            cloud = points['vertices'][np.linspace(0, len(points['vertices'])-1, 1800, dtype=int)]
        prediction = read(run/asset_id/'candidates.json') if status['status']=='success' else None
        joints = (np.array(prediction['joints_world'])-center)/scale if prediction else np.zeros((0,3))
        axes = []
        for ref in refs:
            if ref['asset_key'].split('/')[-1] != asset_id:
                continue
            score = scores[ref['sample_id']]
            origin = np.array(ref['origin_world']); direction = np.array(ref['direction_world'])
            direction /= np.linalg.norm(direction)
            child_center = np.array(ref['center_world'])
            anchor = origin + np.dot(child_center-origin,direction)*direction
            v29_error = None
            if ref['sample_id'] in old and old[ref['sample_id']].get('result'):
                prior = old[ref['sample_id']]['result']
                pivot, vector = np.array(prior['pivot']), np.array(prior['axis'])
                vector /= np.linalg.norm(vector)
                # Same anchor gauge as the previous benchmark before substituting GT direction.
                pivot = pivot + np.dot(child_center-pivot, vector)*vector
                v29_error = float(point_axis_errors(pivot,origin,direction)/ref['child_scale_m'])
            current = float(score['oracle_offset_fraction']) if score['oracle_offset_fraction'] else None
            axis = dict(name=ref['joint_name'], origin=((anchor-center)/scale).tolist(), direction=direction.tolist(),
                        length=ref['child_scale_m']/scale, oracle=current,
                        oracle_index=int(score['oracle_index']) if score['oracle_index'] else None,
                        center=float(score['center_offset_fraction']), v29=v29_error)
            axes.append(axis)
            comparison.append(dict(sample_id=ref['sample_id'],case_name=row['case_name'],**axis))
        cases.append(dict(id=asset_id,name=row['case_name'],cloud=cloud.round(5).tolist(),joints=joints.tolist(),
                          parents=prediction['parents'] if prediction else [], axes=axes, status=status))
    def stats(key):
        values=[a[key] for a in comparison if a[key] is not None]
        return dict(available=len(values), pass_1pct=sum(v<=.01 for v in values),pass_2pct=sum(v<=.02 for v in values),
                    median_pct=float(np.median(values)*100) if values else None)
    result=dict(planned=len(refs),metrics={k:stats(k) for k in ['oracle','center','v29']},
                seconds_sum=sum(r['seconds'] for r in records), seconds_median=float(np.median([r['seconds'] for r in records])),
                peak_allocated_gib=max(r.get('peak_allocated_gib',0) for r in records),
                total_nodes=sum(r.get('joints',0) for r in records),
                note='Fixed reference direction. Oracle selects the best predicted node using reference labels; not automatic prediction accuracy.')
    write(run/'comparison.json',dict(summary=result,axes=comparison,baseline_records=old))
    import json
    payload=json.dumps(dict(summary=result,cases=cases),ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    html=(Path(__file__).with_name('report_template.html').read_text(encoding='utf-8')).replace('__PAYLOAD__',payload)
    (run/'report.html').write_text(html,encoding='utf-8')
    print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=ROOT/'data/user25')
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--v29-records',type=Path)
    a=parser.parse_args(); build(a.data.resolve(),a.run.resolve(),a.v29_records)
