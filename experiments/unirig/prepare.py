"""Prepare label-free whole-assembly points; retain labels in a separate file."""
import argparse
import gc
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import trimesh
from common import ROOT, UPSTREAM, read, write, sha, restore


def prepare(source, reference, output):
    sys.path.insert(0, str(UPSTREAM))
    from src.data.sampler import SamplerConfig, get_sampler
    source = source.resolve()
    output.mkdir(parents=True, exist_ok=True)
    refs = [r for r in read(reference) if r['source'] == 'user_continuous_26']
    manifest = read(source / 'manifest.json')
    assets = sorted([a for a in manifest['assets'] if a['continuous_joint_count'] > 0], key=lambda a: a['robot_name'])
    if len(assets) != 25 or len(refs) != 65:
        raise ValueError('Expected the confirmed 25 assets / 65 reference axes')
    sampler = get_sampler(SamplerConfig(method='mix', num_samples=65536, vertex_samples=8192, kwargs={}))
    rows = []
    for index, item in enumerate(assets):
        folder = source / 'assets' / item['id']
        metadata = read(folder / 'asset.json')
        if metadata['source']['up_axis'] != 'Z':
            raise ValueError('This experiment expects the curated Z-up world frame')
        path = folder / 'scene.glb'
        asset_refs = [r for r in refs if r['asset_key'].split('/')[-1] == item['id']]
        if len(asset_refs) != item['continuous_joint_count']:
            raise ValueError('Frozen reference identity/count mismatch')
        mesh_hash, label_hash = sha(path), sha(folder/'joints.json')
        if any(r['source_scene_sha256'] != mesh_hash or r['source_joints_sha256'] != label_hash for r in asset_refs):
            raise ValueError('Source differs from the previous frozen benchmark')
        scene = trimesh.load_scene(path, process=False)
        meshes = []
        for node in scene.graph.nodes_geometry:
            transform, geometry = scene.graph[node]
            mesh = scene.geometry[geometry].copy()
            mesh.apply_transform(transform)
            meshes.append(mesh)
        mesh = trimesh.util.concatenate(meshes)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        bounds = np.stack([vertices.min(axis=0), vertices.max(axis=0)])
        if not np.allclose(bounds, metadata['bounds'], atol=1e-6, rtol=1e-5):
            raise ValueError('Scene world bounds mismatch: ' + item['id'])
        center = bounds.mean(axis=0)
        scale = float(np.max(bounds[1] - bounds[0]) / 2)
        if not np.isfinite(vertices).all() or scale <= 0:
            raise ValueError('Invalid mesh')
        # The curated source is already Z-up, matching Blender internal coordinates.
        # Read numeric glTF coordinates directly; no implicit Y/Z swap or per-part recentering.
        normalized = ((vertices - center) / scale).astype(np.float32)
        asset = SimpleNamespace(vertices=normalized, faces=np.asarray(mesh.faces),
                                vertex_normals=np.asarray(mesh.vertex_normals, dtype=np.float32),
                                face_normals=np.asarray(mesh.face_normals, dtype=np.float32), vertex_groups={})
        np.random.seed(12345)
        points = sampler.sample(asset)
        out = output / 'inputs' / item['id']
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / 'points.npz', vertices=points.vertices.astype(np.float32), normals=points.normals.astype(np.float32))
        frame = dict(center_world=center.tolist(), scale_m=scale, source_up='Z', model_up='Z', unit='metre', rotation='identity')
        error = float(np.max(np.abs(restore(normalized, frame) - vertices)))
        if error > max(1e-7, scale * 2e-7):
            raise ValueError('Normalization round trip failed')
        write(out / 'frame.json', frame)
        row = dict(asset_id=item['id'], case_name=item['robot_name'], source_glb=str(path), source_sha256=mesh_hash,
                   points=str((out/'points.npz').relative_to(output)), points_sha256=sha(out/'points.npz'),
                   frame=str((out/'frame.json').relative_to(output)), frame_sha256=sha(out/'frame.json'),
                   source_vertices=len(vertices), source_faces=len(mesh.faces), input_points=65536,
                   coordinate_roundtrip_max_m=error, seed=12345)
        rows.append(row)
        print('PREPARED', index+1, '/', len(assets), item['robot_name'], flush=True)
        del scene, meshes, mesh, vertices, asset, points
        gc.collect()
    # References are never read by the inference entry point.
    write(output / 'reference_axes.json', refs)
    write(output / 'inputs.json', rows)
    write(output / 'READY.json', dict(status='INPUTS_READY', assets=len(rows), reference_axes=len(refs),
          source_manifest_sha256=sha(source/'manifest.json'), original_reference_sha256=sha(reference),
          files={name: sha(output/name) for name in ['inputs.json', 'reference_axes.json']},
          decimation=False, normals='trimesh', sampler='unmodified official UniRig SamplerMix', labels_in_points=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'data/user25')
    a = parser.parse_args()
    prepare(a.source, a.reference, a.output)
