from collections import defaultdict

import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation

from mesh_loader import load_obj
from primitives_loader import PrimitiveSpec
from poses_loader import PoseRecord

_mesh_cache: dict = {}


def _rgba(color: tuple) -> list:
    return [int(c * 255) for c in color] + [255]


def _quat_xyzw(rot_matrix: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(rot_matrix).as_quat()  # scipy returns xyzw


def _log_primitive(spec: PrimitiveSpec, translation: np.ndarray, rot_matrix: np.ndarray):
    entity = f"primitive/{spec.id}"
    color = _rgba(spec.color)
    labels = [spec.label] if spec.label else None

    if spec.type == "sphere":
        rr.log(
            entity,
            rr.Ellipsoids3D(
                centers=[translation.tolist()],
                radii=[spec.params["radius"]],
                colors=[color],
                labels=labels,
            ),
        )

    elif spec.type == "cylinder":
        # Capsule3D starts at 'translations' and extends for 'lengths' along local +Z.
        # Offset start by -length/2 along the world-space axis so the pose translation
        # is the cylinder's center.
        length = spec.params["length"]
        radius = spec.params["radius"]
        axis = rot_matrix @ np.array([0.0, 0.0, 1.0])
        start = (translation - (length / 2.0) * axis).tolist()
        quat = _quat_xyzw(rot_matrix)
        rr.log(
            entity,
            rr.Capsules3D(
                lengths=[length],
                radii=[radius],
                translations=[start],
                quaternions=[rr.Quaternion(xyzw=quat)],
                colors=[color],
                labels=labels,
            ),
        )

    elif spec.type == "line_segment":
        length = spec.params["length"]
        axis = rot_matrix @ np.array([0.0, 0.0, 1.0])
        start = (translation - (length / 2.0) * axis).tolist()
        end = (translation + (length / 2.0) * axis).tolist()
        rr.log(
            entity,
            rr.LineStrips3D(
                strips=[[start, end]],
                colors=[color[:3]],
            ),
        )

    elif spec.type == "arrow":
        length = spec.params["length"]
        vector = (rot_matrix @ np.array([0.0, 0.0, length])).tolist()
        rr.log(
            entity,
            rr.Arrows3D(
                origins=[translation.tolist()],
                vectors=[vector],
                colors=[color],
                labels=labels,
            ),
        )

    elif spec.type == "box":
        # Box geometry is in local space; Transform3D positions it in world space.
        quat = _quat_xyzw(rot_matrix)
        rr.log(
            entity,
            rr.Transform3D(
                translation=translation.tolist(),
                rotation=rr.Quaternion(xyzw=quat),
            ),
        )
        rr.log(
            entity,
            rr.Boxes3D(
                centers=[[0.0, 0.0, 0.0]],
                half_sizes=[[
                    spec.params["size_x"] / 2.0,
                    spec.params["size_y"] / 2.0,
                    spec.params["size_z"] / 2.0,
                ]],
                colors=[color],
                labels=labels,
            ),
        )

    elif spec.type == "mesh":
        mesh_path = spec.params["mesh_path"]
        scale = spec.params.get("scale", 1.0)
        if mesh_path not in _mesh_cache:
            print(f"Loading mesh: {mesh_path}")
            _mesh_cache[mesh_path] = load_obj(mesh_path)
        verts, tris, normals = _mesh_cache[mesh_path]
        if scale != 1.0:
            verts = verts * scale
        quat = _quat_xyzw(rot_matrix)
        rr.log(
            entity,
            rr.Transform3D(
                translation=translation.tolist(),
                rotation=rr.Quaternion(xyzw=quat),
            ),
        )
        rr.log(
            entity,
            rr.Mesh3D(
                vertex_positions=verts,
                triangle_indices=tris,
                vertex_normals=normals,
                albedo_factor=[*color[:3], 1.0],
            ),
        )
        # Mesh3D has no label field, so attach the caption via a child entity
        # (which inherits the parent's Transform3D above).
        if labels:
            rr.log(
                f"{entity}/label",
                rr.Points3D(
                    positions=[[0.0, 0.0, 0.0]],
                    colors=[color],
                    labels=labels,
                    radii=[0.0],
                ),
            )


def build_scene(primitives: dict, poses: list):
    """Log all poses to Rerun, one timestep at a time."""
    by_time: dict = defaultdict(list)
    for pose in poses:
        by_time[pose.timestamp].append(pose)

    previously_active: set = set()

    for timestamp in sorted(by_time.keys()):
        rr.set_time("time", duration=timestamp)
        group = by_time[timestamp]
        active_ids = {pose.primitive_id for pose in group}

        # Primitives absent at this timestamp get cleared so they disappear.
        for prim_id in previously_active - active_ids:
            rr.log(f"primitive/{prim_id}", rr.Clear(recursive=True))

        for pose in group:
            if pose.primitive_id not in primitives:
                print(f"Warning: primitive_id={pose.primitive_id} not in primitives table, skipping")
                continue
            _log_primitive(primitives[pose.primitive_id], pose.translation, pose.rotation_matrix)

        previously_active = active_ids
