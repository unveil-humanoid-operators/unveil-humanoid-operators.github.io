"""
generate_root_squat_demo.py
---------------------------
Build a synthetic G1 CSV where every joint is frozen at frame 0 of an existing
clip and the body's root translateZ oscillates as a sine wave — the robot
floats up and down as a rigid block. Illustrates root_translate_rom (height
panel) without distorting limb angles.

Output: assets/demos/g1_csv/root_squat_demo.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
BASE = HERE / "g1_csv" / "exercising.csv"
OUT  = HERE / "g1_csv" / "root_squat_demo.csv"

FPS        = 120
DURATION_S = 3.0
CYCLES     = 2
AMP_CM     = 10.0                                  # ±10 cm vertical sweep

N_FRAMES = int(FPS * DURATION_S)
src = pd.read_csv(BASE)
frame0 = src.iloc[0].copy()

# Lock horizontal position + rotation to canonical so the side camera sees a
# pure vertical bounce.
frame0["root_translateX"] = 0.0
frame0["root_translateY"] = 0.0
frame0["root_rotateX"]    = 0.0
frame0["root_rotateY"]    = 0.0
frame0["root_rotateZ"]    = 0.0

base_z_cm = float(frame0["root_translateZ"])

rows = []
for i in range(N_FRAMES):
    row = frame0.copy()
    row["Frame"] = i
    phase = 2 * np.pi * (i / N_FRAMES) * CYCLES
    row["root_translateZ"] = base_z_cm + AMP_CM * np.sin(phase)
    rows.append(row)

out_df = pd.DataFrame(rows, columns=src.columns)
out_df.to_csv(OUT, index=False)
print(f"wrote {OUT}  ({len(out_df)} frames @ {FPS} fps, ±{AMP_CM} cm root Z)")
