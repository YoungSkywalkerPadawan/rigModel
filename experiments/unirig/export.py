from pathlib import Path
import numpy as np
import trimesh


def export_overlay(source, joints, parents, output):
    scene = trimesh.load_scene(source, process=False)
    radius = max(float(np.linalg.norm(scene.extents))*0.004, 1e-5)
    for i, point in enumerate(joints):
        marker = trimesh.creation.icosphere(subdivisions=1, radius=radius)
        marker.apply_translation(point)
        marker.visual.vertex_colors = [255, 75, 45, 255]
        scene.add_geometry(marker, node_name=f'rig_candidate_{i}', geom_name=f'rig_candidate_{i}')
        parent = parents[i]
        if parent is not None and np.linalg.norm(point-joints[parent]) > 1e-9:
            bone = trimesh.creation.cylinder(radius=radius*0.3, segment=[joints[parent], point], sections=8)
            bone.visual.vertex_colors = [255, 190, 30, 255]
            scene.add_geometry(bone, node_name=f'rig_bone_{i}', geom_name=f'rig_bone_{i}')
    Path(output).write_bytes(scene.export(file_type='glb'))
