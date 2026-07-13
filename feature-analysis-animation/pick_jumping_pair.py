"""
pick_jumping_pair.py
--------------------
Find a heavy/light operator pair for the 'jumping' demo where the heavy
operator's specific clip is *genuinely slower* than the light operator's
specific clip — not just the population aggregate. This is what makes the
red-vs-blue strip-plot curves show the expected gap.

Strategy:
  1. From the metadata parquet, take every actor with at least one G1 clip
     whose content_type_of_movement is 'jumping' and whose g1 csv file exists.
  2. For each actor, pick their longest jumping clip and compute its per-frame
     3D root speed average (matches what twin_view.html renders).
  3. Take the heaviest 25% of actors as the heavy pool and lightest 25% as the
     light pool.
  4. Score (heavy, light) pairs by:
       weight_gap  =  light.weight - heavy.weight    (we want < 0, light is lighter)
       speed_gap   =  light.speed  - heavy.speed     (we want > 0, light is faster)
       both clips have ≥ 200 frames
  5. Pick the pair with the largest combined |weight_gap| × speed_gap.

Then copy the two CSVs into data/trajectories/jumping_heavy.csv and
jumping_light.csv, and patch pairs.json's "jumping" entry in-place.

Usage:
  python pick_jumping_pair.py --bones-seed C:/Users/sihat/Downloads/bones-seed
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def avg_3d_root_speed(csv_path: Path) -> tuple[float, int]:
    """Return (avg |Δp_3D|/Δt in m/s, n_frames) for a G1 CSV."""
    df = pd.read_csv(csv_path, usecols=["root_translateX", "root_translateY", "root_translateZ"])
    xyz = df.to_numpy() * 0.01  # cm → m
    if len(xyz) < 2:
        return float("nan"), len(xyz)
    diffs = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    avg = diffs.mean() * 120.0   # source CSV is 120 fps
    return float(avg), len(xyz)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bones-seed", required=True, type=Path)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "data")
    ap.add_argument("--min-frames", type=int, default=200)
    ap.add_argument("--min-weight-gap", type=float, default=30.0,
                    help="Minimum |heavy − light| weight gap (kg).")
    args = ap.parse_args()

    bs = args.bones_seed.resolve()
    out = args.out.resolve()
    (out / "trajectories").mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(bs / "metadata" / "seed_metadata_v003.parquet")
    jp = meta[meta["content_type_of_movement"] == "jumping"].copy()
    print(f"jumping clips total: {len(jp)}, unique actors: {jp['actor_uid'].nunique()}")

    # Take each actor's longest available G1 clip (excluding mirror _M).
    cand = jp[~jp["move_g1_mujoco_path"].astype(str).str.endswith("_M.csv")].copy()
    cand["abs_csv"] = cand["move_g1_mujoco_path"].apply(lambda r: bs / r if isinstance(r, str) else None)
    cand = cand[cand["abs_csv"].apply(lambda p: p is not None and p.exists())]
    cand["nf"] = cand["move_duration_frames"].astype(int, errors="ignore")
    cand = cand[cand["nf"] >= args.min_frames]
    cand.sort_values(["actor_uid", "nf"], ascending=[True, False], inplace=True)
    longest = cand.drop_duplicates("actor_uid", keep="first").reset_index(drop=True)
    print(f"actors with usable clip (≥{args.min_frames} frames): {len(longest)}")

    # Compute per-clip 3D root avg speed for each chosen clip.
    print("computing per-clip avg speeds…")
    speeds = []
    for r in longest.itertuples():
        s, n = avg_3d_root_speed(r.abs_csv)
        speeds.append((s, n))
    longest["clip_speed"] = [s for s, _ in speeds]
    longest["clip_nf_actual"] = [n for _, n in speeds]

    longest = longest.dropna(subset=["clip_speed", "actor_weight_kg"])
    longest = longest.sort_values("actor_weight_kg").reset_index(drop=True)
    print(f"actor weight range: {longest['actor_weight_kg'].min():.0f}–{longest['actor_weight_kg'].max():.0f} kg")
    print(f"clip speed range:  {longest['clip_speed'].min():.2f}–{longest['clip_speed'].max():.2f} m/s")

    # Keep only clips whose per-frame avg speed is in a "typical jump" band so
    # we don't compare a tiny look-around-jump to a 2 m obstacle vault. Range
    # was eyeballed from the speed histogram: median ~0.25 m/s, IQR ~0.18-0.36.
    SPEED_LO, SPEED_HI = 0.20, 0.45
    longest = longest[(longest["clip_speed"] >= SPEED_LO) &
                      (longest["clip_speed"] <= SPEED_HI)].reset_index(drop=True)
    print(f"after speed band [{SPEED_LO}, {SPEED_HI}] m/s: {len(longest)} actors remain")

    n = len(longest)
    heavy_pool = longest.iloc[int(n * 0.75):].copy()    # top 25% by weight
    light_pool = longest.iloc[: int(n * 0.25)].copy()    # bottom 25%

    # Of those, pick the pair whose speed-and-weight ordering most cleanly
    # demonstrates heavier-is-slower. Score: weight_gap × speed_gap, where
    # both gaps are positive (heavy heavier, light faster). Clips are already
    # within a comparable speed band, so this finds the cleanest example.
    best = None
    for h in heavy_pool.itertuples():
        for l in light_pool.itertuples():
            wgap = h.actor_weight_kg - l.actor_weight_kg
            if wgap < args.min_weight_gap:
                continue
            sgap = l.clip_speed - h.clip_speed
            if sgap <= 0:
                continue
            score = wgap * sgap
            if best is None or score > best[0]:
                best = (score, h, l)
    if best is None:
        raise SystemExit(
            f"no pair found with heavy slower than light by ≥{args.min_weight_gap} kg gap; "
            "try lowering --min-weight-gap or --min-frames")
    score, h, l = best
    print()
    print(f"chosen heavy: A{h.actor_uid[1:]}  {h.actor_weight_kg:.0f} kg "
          f"{h.actor_gender} {h.actor_height_cm:.0f} cm   "
          f"clip {h.abs_csv.name}  speed={h.clip_speed:.3f} m/s  nf={int(h.clip_nf_actual)}")
    print(f"chosen light: A{l.actor_uid[1:]}  {l.actor_weight_kg:.0f} kg "
          f"{l.actor_gender} {l.actor_height_cm:.0f} cm   "
          f"clip {l.abs_csv.name}  speed={l.clip_speed:.3f} m/s  nf={int(l.clip_nf_actual)}")
    print(f"weight gap = {h.actor_weight_kg - l.actor_weight_kg:.0f} kg, "
          f"speed gap (light − heavy) = {l.clip_speed - h.clip_speed:.3f} m/s (heavy slower)")

    dest_h = out / "trajectories" / "jumping_heavy.csv"
    dest_l = out / "trajectories" / "jumping_light.csv"
    shutil.copyfile(h.abs_csv, dest_h)
    shutil.copyfile(l.abs_csv, dest_l)
    print(f"\nwrote {dest_h}")
    print(f"wrote {dest_l}")

    # Patch pairs.json's 'jumping' entry so twin_view.html picks these up.
    pairs_path = out / "pairs.json"
    pairs = json.loads(pairs_path.read_text())
    pairs["jumping"] = {
        "heavy": {
            "uid": h.actor_uid, "weight": float(h.actor_weight_kg),
            "height": float(h.actor_height_cm), "age": float(h.actor_age_yr),
            "gender": str(h.actor_gender), "root_avg_speed": float(h.clip_speed) * 100,
            "csv": "data/trajectories/jumping_heavy.csv",
            "source_csv": str(Path(h.abs_csv).relative_to(bs)).replace("\\", "/"),
            "n_frames": int(h.clip_nf_actual),
        },
        "light": {
            "uid": l.actor_uid, "weight": float(l.actor_weight_kg),
            "height": float(l.actor_height_cm), "age": float(l.actor_age_yr),
            "gender": str(l.actor_gender), "root_avg_speed": float(l.clip_speed) * 100,
            "csv": "data/trajectories/jumping_light.csv",
            "source_csv": str(Path(l.abs_csv).relative_to(bs)).replace("\\", "/"),
            "n_frames": int(l.clip_nf_actual),
        },
    }
    pairs_path.write_text(json.dumps(pairs, indent=2))
    print(f"updated {pairs_path}")


if __name__ == "__main__":
    main()
