"""Freeze label-free whole-object inputs using each model's own preprocessing."""
import argparse
import gc
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import trimesh
from shared import ROOT, WORLD_TO_MODEL, normalize_world, read, restore, sha, write


def assemble(row):
    if sha(row['source_glb']) != row['source_sha256']:
        raise ValueError('The frozen source scene changed')
    scene = trimesh.load(row['source_glb'], force='scene', process=False)
    parts = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        mesh = scene.geometry[name].copy()
        mesh.apply_transform(transform)
        parts.append(mesh)
    mesh = trimesh.util.concatenate(parts)
    if len(mesh.vertices) != row['source_vertices'] or len(mesh.faces) != row['source_faces']:
        raise ValueError('Source assembly vertex/face count changed')
    transform = np.eye(4)
    transform[:3, :3] = WORLD_TO_MODEL.T
    mesh.apply_transform(transform)
    return mesh


def prepare_one(model, mesh, seed):
    np.random.seed(seed)
    if model == 'riganything':
        # Official inference processes each mesh and uses sample_surface_even, then pads to 1024.
        shuffled = np.arange(len(mesh.vertices))
        np.random.shuffle(shuffled)
        points, faces = trimesh.sample.sample_surface_even(mesh, 1024, seed=seed)
        indices = np.random.choice(len(points), 1024, replace=len(points) < 1024)
        points = points[indices].astype(np.float32)
        normals = mesh.face_normals[faces[indices]].astype(np.float32)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        center = (points.max(axis=0) + points.min(axis=0)) / 2
        points -= center
        scale = float(np.abs(points).max())
        points /= scale
        frame = dict(scale_m=scale, center_world=(center @ WORLD_TO_MODEL.T).tolist())
        encoder = points
        detail = dict(surface_sampler='trimesh.sample_surface_even', surface_samples=1024, marching_cubes=False)
    else:
        sys.path.insert(0, str(ROOT/'third_party/Puppeteer/skeleton'))
        from utils.mesh_to_pc import MeshProcessor
        from data_utils.save_npz import normalize_to_unit_cube
        # Match the official demo: depth-7 SDF/marching cubes, surface normals, fp16 preprocessing.
        watertight = MeshProcessor.convert_to_watertight(mesh, octree_depth=7)
        sampled, faces = trimesh.sample.sample_surface(watertight, 8192, seed=seed)
        original = np.concatenate([sampled, watertight.face_normals[faces]], axis=1, dtype=np.float16)
        encoder, center, multiplier = normalize_to_unit_cube(original[:, :3].copy(), scale_factor=.9995)
        encoder, normals = encoder.astype(np.float32), original[:, 3:].astype(np.float32)
        pc_center = (encoder.max(axis=0) + encoder.min(axis=0)) / 2
        pc_scale = float((encoder.max(axis=0) - encoder.min(axis=0)).max() + 1e-5)
        # The demo stores all transform_params as float32 before undoing the two normalizations.
        center = center.astype(np.float32)
        multiplier, pc_scale = float(np.float32(multiplier)), float(np.float32(pc_scale))
        points = ((encoder - pc_center) / pc_scale).astype(np.float32)
        frame = dict(scale_m=pc_scale/multiplier,
                     center_world=((pc_center/multiplier + center) @ WORLD_TO_MODEL.T).tolist(),
                     official_transform_params=[*center.tolist(), multiplier, *pc_center.tolist(), pc_scale])
        detail = dict(surface_sampler='trimesh.sample_surface', surface_samples=8192, marching_cubes=True,
                      octree_depth=7, processed_vertices=len(watertight.vertices), processed_faces=len(watertight.faces))
    frame.update(source_up='Z', model_up='Y', unit='metre', model_to_world_rotation=WORLD_TO_MODEL.T.tolist())
    if not np.isfinite(encoder).all() or not np.isfinite(normals).all() or not (np.linalg.norm(normals, axis=1) > .99).all():
        raise ValueError('Nonfinite points or nonunit normals')
    if frame['scale_m'] <= 0:
        raise ValueError('Degenerate shape normalization')
    np.testing.assert_allclose(normalize_world(restore(points, frame), frame), points, atol=1e-9)
    return dict(encoder_points=encoder, vertices=points, normals=normals), frame, detail


def prepare(model, original, output, seed=12345):
    frozen = read(original/'READY.json')
    for name in ('inputs.json', 'reference_axes.json'):
        if sha(original/name) != frozen['files'][name]:
            raise ValueError('Frozen UniRig benchmark identity changed')
    inputs = read(original/'inputs.json')
    refs = read(original/'reference_axes.json')
    if len(inputs) != 25 or len(refs) != 65:
        raise ValueError('Expected the confirmed 25 objects and 65 axes')
    rows = []
    for index, old in enumerate(inputs):
        start = time.monotonic()
        folder = output/'inputs'/old['asset_id']
        state = folder/'prepared.json'
        if state.exists():
            row = read(state)
            assert row['source_sha256'] == old['source_sha256'] and row['model'] == model and row['seed'] == seed
            assert sha(output/row['points']) == row['points_sha256'] and sha(output/row['frame']) == row['frame_sha256']
        else:
            mesh = assemble(old)
            arrays, frame, detail = prepare_one(model, mesh, seed)
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder/'points.npz', **arrays)
            write(folder/'frame.json', frame)
            row = dict(asset_id=old['asset_id'], case_name=old['case_name'], source_glb=old['source_glb'],
                       source_sha256=old['source_sha256'], model=model, seed=seed, input_points=len(arrays['vertices']),
                       points=(folder/'points.npz').relative_to(output).as_posix(), points_sha256=sha(folder/'points.npz'),
                       frame=(folder/'frame.json').relative_to(output).as_posix(), frame_sha256=sha(folder/'frame.json'),
                       preprocessing=detail, preparation_seconds=time.monotonic()-start)
            write(state, row)
            del mesh, arrays
            gc.collect()
        rows.append(row)
        print('PREPARED', model, index+1, '/', len(inputs), old['case_name'], flush=True)
    write(output/'inputs.json', rows)
    shutil.copyfile(original/'reference_axes.json', output/'reference_axes.json')
    write(output/'READY.json', dict(status='INPUTS_READY', model=model, assets=len(rows), reference_axes=len(refs),
          source_inputs_sha256=sha(original/'inputs.json'), seed=seed, labels_in_points=False,
          files={name: sha(output/name) for name in ['inputs.json', 'reference_axes.json']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['riganything', 'puppeteer'], required=True)
    parser.add_argument('--original', type=Path, default=ROOT/'data/user25')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    prepare(args.model, args.original.resolve(), (args.output or ROOT/f'data/user25_{args.model}').resolve())
