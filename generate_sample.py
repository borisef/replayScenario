"""
Generates sample_poses.csv with a smooth 10-second animation:
  - Object 1 (sphere):   orbits in a circle, gently rising and falling
  - Object 2 (cylinder): swings back and forth while spinning end-over-end
  - Object 6 (mesh):     plane flies a banked circular orbit at altitude
"""

import numpy as np
import pandas as pd

N_FRAMES = 200          # frames total
DURATION = 10.0         # seconds
t = np.linspace(0, DURATION, N_FRAMES)

rows = []

for ts in t:
    phase = 2 * np.pi * ts / DURATION   # 0 → 2π over the full clip

    # --- sphere: circular orbit + gentle bob ---
    tx1 = 3.0 * np.cos(phase)
    ty1 = 3.0 * np.sin(phase)
    tz1 = 0.8 * np.sin(2 * phase)       # bobs up/down twice per orbit
    rows.append([ts, 1, tx1, ty1, tz1, 0.0, 0.0, 0.0])

    # --- cylinder: pendulum swing + full end-over-end spin ---
    tx2 = 5.0 * np.sin(phase)           # swings ±5 along X
    ty2 = 0.0
    tz2 = 0.0
    yaw2 = phase * 1.5                  # 1.5 full spins over the clip
    pitch2 = (np.pi / 3) * np.sin(phase * 2)   # nodding motion
    rows.append([ts, 2, tx2, ty2, tz2, yaw2, pitch2, 0.0])

    # --- plane (mesh): banked circular orbit at altitude ---
    r_plane = 8.0
    alt = 5.0
    tx6 = r_plane * np.cos(phase * 1.5)
    ty6 = r_plane * np.sin(phase * 1.5)
    tz6 = alt + 1.5 * np.sin(phase)
    yaw6 = phase * 1.5 + np.pi / 2     # nose tangent to orbit circle
    pitch6 = 0.1 * np.sin(phase)       # gentle pitch oscillation
    roll6 = 0.3 * np.sin(phase)        # banking
    rows.append([ts, 6, tx6, ty6, tz6, yaw6, pitch6, roll6])

df = pd.DataFrame(
    rows,
    columns=["timestamp", "primitive_id", "tx", "ty", "tz", "yaw", "pitch", "roll"],
)
OUT_PATH = "data/scene/sample_poses.csv"
df.to_csv(OUT_PATH, index=False, float_format="%.6f")
print(f"Written {OUT_PATH}  ({len(df)} rows, {N_FRAMES} frames)")
