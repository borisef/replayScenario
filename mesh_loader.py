import numpy as np


def load_obj(path: str) -> tuple:
    """Parse a Wavefront OBJ file for use with rr.Mesh3D.

    Returns:
        vertices:         (N, 3) float32 — vertex positions
        triangle_indices: (M, 3) uint32  — 0-based triangle indices
        normals:          (N, 3) float32 — per-vertex normals
    """
    positions = []
    norm_list = []
    face_tris = []  # list of 3-tuples of (v_idx, vn_idx) pairs

    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("vn "):
                parts = line.split()
                norm_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                tokens = line.split()[1:]
                corners = []
                for token in tokens:
                    sub = token.split("/")
                    v_idx = int(sub[0]) - 1
                    vn_idx = int(sub[2]) - 1 if len(sub) >= 3 and sub[2] != "" else None
                    corners.append((v_idx, vn_idx))
                # Fan-triangulate polygons: (0,1,2), (0,2,3), ...
                for i in range(1, len(corners) - 1):
                    face_tris.append((corners[0], corners[i], corners[i + 1]))

    # Remap: each unique (v_idx, vn_idx) pair becomes one new vertex
    key_to_new = {}
    new_pos = []
    new_nrm = []
    triangles = []

    fallback_normal = [0.0, 0.0, 1.0]

    for tri in face_tris:
        idx3 = []
        for (vi, ni) in tri:
            key = (vi, ni)
            if key not in key_to_new:
                key_to_new[key] = len(new_pos)
                new_pos.append(positions[vi])
                if ni is not None and ni < len(norm_list):
                    new_nrm.append(norm_list[ni])
                else:
                    new_nrm.append(fallback_normal)
            idx3.append(key_to_new[key])
        triangles.append(idx3)

    vertices = np.array(new_pos, dtype=np.float32)
    triangle_indices = np.array(triangles, dtype=np.uint32)
    normals = np.array(new_nrm, dtype=np.float32)

    return vertices, triangle_indices, normals
