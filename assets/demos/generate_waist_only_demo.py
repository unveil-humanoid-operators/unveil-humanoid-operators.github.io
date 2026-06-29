"""
generate_waist_only_demo.py
---------------------------
Build a synthetic G1 CSV where every joint is frozen at frame 0 of an existing
clip *except* the waist_yaw joint, which oscillates as a sine wave. Used by
the bar plot's AGE panel left strip to illustrate waist range-of-motion.

Output: assets/demos/g1_csv/waist_only_demo.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
BASE = HERE / "g1_csv" / "exercising.csv"
OUT  = HERE / "g1_csv" / "waist_only_demo.csv"

FPS         = 120
DURATION_S  = 3.0
CYCLES      = 2
AMP_DEG     = 45                                # ±45° yaw — visible torso twist

N_FRAMES = int(FPS * DURATION_S)
src = pd.read_csv(BASE)
frame0 = src.iloc[0].copy()

# Center the body and zero the root rotation so the synthetic view is canonical.
frame0["root_translateX"] = 0.0
frame0["root_translateY"] = 0.0
frame0["root_rotateX"]    = 0.0
frame0["root_rotateY"]    = 0.0
frame0["root_rotateZ"]    = 0.0

rows = []
for i in range(N_FRAMES):
    row = frame0.copy()
    row["Frame"] = i
    phase = 2 * np.pi * (i / N_FRAMES) * CYCLES
    delta = AMP_DEG * np.sin(phase)
    row["waist_yaw_joint_dof"] = float(frame0["waist_yaw_joint_dof"]) + delta
    rows.append(row)

out_df = pd.DataFrame(rows, columns=src.columns)
out_df.to_csv(OUT, index=False)
print(f"wrote {OUT}  ({len(out_df)} frames @ {FPS} fps, ±{AMP_DEG} deg waist yaw)")
