"""
generate_ankle_only_demo.py
---------------------------
Build a synthetic G1 CSV where every joint is frozen at frame 0 of an existing
clip *except* the two ankle pitch joints, which oscillate as a sine wave.

Used by the bar plot's gender-panel left strip — the viewer renders a full
upright G1 with only the ankle moving, while an SVG overlay draws an ROM arc
on top.

Output: assets/demos/g1_csv/ankle_only_demo.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
BASE = HERE / "g1_csv" / "exercising.csv"     # source for the rest pose
OUT  = HERE / "g1_csv" / "ankle_only_demo.csv"

FPS         = 120
DURATION_S  = 3.0
CYCLES      = 2                                # full back-and-forth cycles
AMP_DEG     = 25                               # ±25° → ~50° total ROM

N_FRAMES = int(FPS * DURATION_S)

src = pd.read_csv(BASE)
frame0 = src.iloc[0].copy()

# Normalize: zero out horizontal root translation so the robot stands centered.
# Keep the vertical (Z) component so the pelvis stays at ~78cm above ground.
# Zero out root rotation entirely so the robot faces +X (we'll point the
# camera from +X side; this gives a clean lateral view of the ankle hinge).
frame0["root_translateX"] = 0.0
frame0["root_translateY"] = 0.0
frame0["root_rotateX"]    = 0.0
frame0["root_rotateY"]    = 0.0
frame0["root_rotateZ"]    = 0.0

cols = list(src.columns)
rows = []
for i in range(N_FRAMES):
    row = frame0.copy()
    row["Frame"] = i
    # Only the RIGHT ankle pitch oscillates; left ankle stays at its frame-0
    # value so the user's attention isn't split between two moving feet — the
    # bar plot's arc overlay tracks the right ankle.
    phase = 2 * np.pi * (i / N_FRAMES) * CYCLES
    delta = AMP_DEG * np.sin(phase)
    row["right_ankle_pitch_joint_dof"] = float(frame0["right_ankle_pitch_joint_dof"]) + delta
    rows.append(row)

out_df = pd.DataFrame(rows, columns=cols)
out_df.to_csv(OUT, index=False)
print(f"wrote {OUT}  ({len(out_df)} frames @ {FPS} fps, ±{AMP_DEG} deg ankle pitch)")
