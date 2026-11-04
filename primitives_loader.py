from dataclasses import dataclass
import pandas as pd


@dataclass
class PrimitiveSpec:
    id: int
    type: str
    params: dict
    color: tuple   # (r, g, b) floats 0-1
    label: str


def _get_float(row, col: str, default: float = 0.0) -> float:
    val = row.get(col, default)
    if pd.isna(val):
        return default
    return float(val)


def load_primitives(path: str) -> dict:
    """Return dict mapping primitive id -> PrimitiveSpec."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    primitives = {}
    for _, row in df.iterrows():
        pid = int(row["id"])
        ptype = str(row["type"]).strip().lower()

        color = (
            _get_float(row, "color_r", 0.8),
            _get_float(row, "color_g", 0.8),
            _get_float(row, "color_b", 0.8),
        )

        label = ""
        if "label" in df.columns:
            raw = row.get("label", "")
            label = "" if pd.isna(raw) else str(raw)

        params: dict = {}
        if ptype == "sphere":
            params["radius"] = _get_float(row, "radius", 1.0)
        elif ptype == "cylinder":
            params["radius"] = _get_float(row, "radius", 0.5)
            params["length"] = _get_float(row, "length", 1.0)
        elif ptype == "line_segment":
            params["length"] = _get_float(row, "length", 1.0)
        elif ptype == "arrow":
            params["length"] = _get_float(row, "length", 1.0)
            params["shaft_radius"] = _get_float(row, "shaft_radius", 0.05)
            params["tip_radius"] = _get_float(row, "tip_radius", 0.1)
        elif ptype == "box":
            params["size_x"] = _get_float(row, "size_x", 1.0)
            params["size_y"] = _get_float(row, "size_y", 1.0)
            params["size_z"] = _get_float(row, "size_z", 1.0)
        elif ptype == "mesh":
            if "mesh_path" not in df.columns:
                print(f"Warning: mesh primitive id={pid} requires 'mesh_path' column, skipping")
                continue
            raw_path = row.get("mesh_path", "")
            if pd.isna(raw_path) or str(raw_path).strip() == "":
                print(f"Warning: mesh primitive id={pid} has no mesh_path, skipping")
                continue
            params["mesh_path"] = str(raw_path).strip()
            params["scale"] = _get_float(row, "scale", 1.0)
        else:
            print(f"Warning: unknown primitive type '{ptype}' for id={pid}, skipping")
            continue

        primitives[pid] = PrimitiveSpec(
            id=pid, type=ptype, params=params, color=color, label=label
        )

    return primitives
