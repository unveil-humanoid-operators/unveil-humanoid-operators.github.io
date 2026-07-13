"""
generate_shoulder_only_demo.py
------------------------------
Synthetic G1 CSV: every joint frozen at exercising frame 0 except the right
shoulder pitch joint, which oscillates as a sine wave. Used by the bar plot's
GENDER panel left strip when "shoulder_rom" / "shoulder_mean_vel" is the
selected feature.

Output: assets/demos/g1_csv/shoulder_only_demo.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
BASE = HERE / "g1_csv" / "exercising.csv"
OUT  = HERE / "g1_csv" / "shoulder_only_demo.csv"

FPS         = 120
DURATION_S  = 3.0
CYCLES      = 2
AMP_DEG     = 55              # ±55° arm swing — visible from front-3/4 view

N_FRAMES = int(FPS * DURATION_S)
src = pd.read_csv(BASE)
frame0 = src.iloc[0].copy()

frame0["root_translateX"] = 0.0
frame0["root_translateY"] = 0.0
frame0["root_rotateX"]    = 0.0
frame0["root_rotateY"]    = 0.0
frame0["root_rotateZ"]    = 0.0

base = float(frame0["right_shoulder_pitch_joint_dof"])

rows = []
for i in range(N_FRAMES):
    row = frame0.copy()
    row["Frame"] = i
    phase = 2 * np.pi * (i / N_FRAMES) * CYCLES
    row["right_shoulder_pitch_joint_dof"] = base + AMP_DEG * np.sin(phase)
    rows.append(row)

out = pd.DataFrame(rows, columns=src.columns)
out.to_csv(OUT, index=False)
print(f"wrote {OUT} ({len(out)} frames @ {FPS} fps, ±{AMP_DEG} deg shoulder pitch)")
