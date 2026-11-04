from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class PoseRecord:
    timestamp: float
    primitive_id: int
    translation: np.ndarray    # shape (3,)
    rotation_matrix: np.ndarray  # shape (3, 3)


def _parse_timestamp(val) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return pd.Timestamp(val).timestamp()


def load_poses(path: str) -> list:
    """Return list of PoseRecord sorted by timestamp."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    cols = set(df.columns)

    if {"rx", "ry", "rz"}.issubset(cols):
        rot_format = "rvec"
    elif {"yaw", "pitch", "roll"}.issubset(cols):
        rot_format = "ypr"
    else:
        raise ValueError(
            f"Cannot detect rotation format. "
            f"Expected columns 'rx,ry,rz' (Rodrigues) or 'yaw,pitch,roll' (Euler). "
            f"Found: {sorted(cols)}"
        )

    records = []
    for _, row in df.iterrows():
        ts = _parse_timestamp(row["timestamp"])
        pid = int(row["primitive_id"])
        translation = np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])])

        if rot_format == "rvec":
            rvec = np.array([float(row["rx"]), float(row["ry"]), float(row["rz"])])
            rot = Rotation.from_rotvec(rvec)
        else:
            # ZYX convention: yaw around Z, pitch around Y, roll around X
            rot = Rotation.from_euler(
                "ZYX",
                [float(row["yaw"]), float(row["pitch"]), float(row["roll"])],
            )

        records.append(
            PoseRecord(
                timestamp=ts,
                primitive_id=pid,
                translation=translation,
                rotation_matrix=rot.as_matrix(),
            )
        )

    records.sort(key=lambda r: r.timestamp)
    return records
