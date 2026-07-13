"""
generate_ankle_ramp_demo.py
---------------------------
Synthetic G1 CSV: right ankle pitch oscillates with TIME-VARYING amplitude
so the angular velocity profile has clear peaks. The amplitude follows a
smooth sin² envelope (5° → 30° → 5° over the clip), and the foot oscillates
at the same 1.5 s period as the other illustrations. This gives the weight
panel's overlay a meaningful "peak velocity so far" curve to track.

Output: assets/demos/g1_csv/ankle_ramp_demo.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
BASE = HERE / "g1_csv" / "exercising.csv"
OUT  = HERE / "g1_csv" / "ankle_ramp_demo.csv"

FPS         = 120
DURATION_S  = 3.0
CYCLES      = 2
AMP_MIN_DEG = 5
AMP_MAX_DEG = 30

N_FRAMES = int(FPS * DURATION_S)
src = pd.read_csv(BASE)
frame0 = src.iloc[0].copy()

# Canonical: zero horizontal/yaw root.
frame0["root_translateX"] = 0.0
frame0["root_translateY"] = 0.0
frame0["root_rotateX"]    = 0.0
frame0["root_rotateY"]    = 0.0
frame0["root_rotateZ"]    = 0.0

base_pitch_deg = float(frame0["right_ankle_pitch_joint_dof"])

rows = []
for i in range(N_FRAMES):
    row = frame0.copy()
    row["Frame"] = i
    t = i / FPS                          # seconds
    env  = np.sin(np.pi * t / DURATION_S) ** 2     # 0 → 1 → 0 over the clip
    amp  = AMP_MIN_DEG + (AMP_MAX_DEG - AMP_MIN_DEG) * env
    phase = 2 * np.pi * (i / N_FRAMES) * CYCLES
    row["right_ankle_pitch_joint_dof"] = base_pitch_deg + amp * np.sin(phase)
    rows.append(row)

out_df = pd.DataFrame(rows, columns=src.columns)
out_df.to_csv(OUT, index=False)
print(f"wrote {OUT}  ({len(out_df)} frames @ {FPS} fps, ankle amp {AMP_MIN_DEG}->{AMP_MAX_DEG}°)")
