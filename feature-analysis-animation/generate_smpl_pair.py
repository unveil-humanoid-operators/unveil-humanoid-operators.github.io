"""
generate_smpl_pair.py
---------------------
Fit two SMPL meshes that match the heavy/light G1 trajectories in pairs.json.

Each side uses its OWN source BVH (mapped from the chosen G1 csv via the
metadata parquet), so the top-row SMPL and bottom-row G1 in twin_view.html
play the same motion per column. Body-shape parameters are overridden so
visually the left mesh reads as "heavier" and the right as "lighter".

Pose fitting: BVH2SMPL's VPoser pipeline with --free-arms (per project memory
note) and a relaxed z-prior so the body actually crouches at the jump apex.

Outputs into ./data/smpl/  (binary format compatible with smpl_engine/):
    heavy_verts.bin     heavy operator's BVH, override weight `--heavy-weight`
    light_verts.bin     light operator's BVH, override weight `--light-weight`
    faces.bin           shared SMPL triangle indices

Usage:
  python generate_smpl_pair.py \
      --bones-seed  C:/Users/sihat/Downloads/bones-seed \
      --bvh2smpl-src C:/Users/sihat/Downloads/BVH2SMPL/src \
      --vposer-dir  C:/Users/sihat/Downloads/BVH2SMPL/vposer_v1_0 \
      --device cuda \
      --heavy-weight 110  --light-weight 60
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np


GENDER_MODEL = {
    "M": "basicmodel_m_lbs_10_207_0_v1.0.0.pkl",
    "F": "basicModel_f_lbs_10_207_0_v1.0.0.pkl",
    "male":   "basicmodel_m_lbs_10_207_0_v1.0.0.pkl",
    "female": "basicModel_f_lbs_10_207_0_v1.0.0.pkl",
}
GENDER_STR = {"M": "male", "F": "female", "male": "male", "female": "female"}


def save_verts(verts: np.ndarray, fps: float, path: Path) -> None:
    nf, nv, _ = verts.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<II", nf, nv))
        f.write(struct.pack("<f", float(fps)))
        f.write(verts.astype(np.float32).ravel().tobytes())
    print(f"    -> {path}  ({nf} frames, {os.path.getsize(path)/1e6:.1f} MB)")


def save_faces(faces: np.ndarray, path: Path) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(faces)))
        f.write(faces.astype(np.uint32).ravel().tobytes())
    print(f"    -> {path}  ({len(faces)} faces, {os.path.getsize(path)/1e3:.0f} KB)")


def fit_one(label: str, bvh_abs: Path, smpl_pkl: Path, vposer_dir: Path,
            height_cm: float, weight_kg: float, gender: str,
            bvh2smpl_src: Path, device: str, max_frames: int,
            free_arms: bool, z_prior_w: float):
    from soma_viewer_vposer import fit_smpl_vposer
    from soma_viewer import BVHMotionZYX

    print(f"\n[{label}]")
    print(f"  BVH    : {bvh_abs}")
    print(f"  Shape  : {gender} · {height_cm} cm · {weight_kg} kg")
    print(f"  Device : {device}   free_arms={free_arms}   z_prior_w={z_prior_w}")

    bvh_abs = os.path.abspath(str(bvh_abs))
    os.chdir(str(bvh2smpl_src))

    bvh = BVHMotionZYX(bvh_abs)
    print(f"  Frames : {bvh.motion_length} @ 120 fps")

    verts, faces = fit_smpl_vposer(
        bvh,
        scale=100.0,
        smpl_path=str(smpl_pkl),
        vposer_dir=str(vposer_dir),
        device=device,
        height_cm=height_cm,
        weight_kg=weight_kg,
        gender=gender,
        uid=None,
        free_arms=free_arms,
        z_prior_w=z_prior_w,
    )
    verts = np.asarray(verts)
    faces = np.asarray(faces)

    n = len(verts)
    if n > max_frames:
        idx = np.round(np.linspace(0, n - 1, max_frames)).astype(int)
        verts = verts[idx]
        fps = 120.0 * max_frames / n
    else:
        fps = 120.0
    print(f"  Output : {len(verts)} frames @ {fps:.2f} fps")
    return verts, faces, fps


def g1_csv_to_bvh(g1_rel: str, meta) -> str:
    row = meta[meta["move_g1_mujoco_path"] == g1_rel]
    if len(row) == 0:
        raise SystemExit(f"No metadata row for {g1_rel}")
    return row.iloc[0]["move_soma_uniform_path"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bones-seed", required=True, type=Path)
    ap.add_argument("--bvh2smpl-src", required=True, type=Path)
    ap.add_argument("--vposer-dir", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "data" / "smpl")
    ap.add_argument("--no-free-arms", dest="free_arms", action="store_false", default=True)
    ap.add_argument("--z-prior-w", type=float, default=0.001,
                    help="VPoser z-prior weight. Default 0.001 (relaxed).")
    ap.add_argument("--heavy-weight", type=float, default=110.0,
                    help="Override heavy operator's weight for SMPL betas (kg).")
    ap.add_argument("--light-weight", type=float, default=60.0,
                    help="Override light operator's weight for SMPL betas (kg).")
    ap.add_argument("--side", choices=["both", "heavy", "light"], default="both",
                    help="Refit only one side (skip the other to save time).")
    ap.add_argument("--shared-bvh", choices=["heavy", "light"], default=None,
                    help="If set, BOTH sides use the BVH from that side's pair "
                         "entry (so SMPL motions are identical across columns; "
                         "the only difference between meshes is body shape).")
    args = ap.parse_args()

    bs   = args.bones_seed.resolve()
    src  = args.bvh2smpl_src.resolve()
    vp   = args.vposer_dir.resolve()
    out  = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(src))
    smpl_dir = src / "rendering_utils" / "smpl"

    pairs_json = Path(__file__).parent / "data" / "pairs.json"
    if not pairs_json.exists():
        sys.exit("pairs.json missing — run generate_data.py + pick_jumping_pair.py first.")
    pairs = json.loads(pairs_json.read_text())
    if "jumping" not in pairs:
        sys.exit("pairs.json has no 'jumping' entry.")
    pair = pairs["jumping"]

    import pandas as pd
    meta = pd.read_parquet(bs / "metadata" / "seed_metadata_v003.parquet")

    sides = []
    if args.side in ("both", "heavy"):
        sides.append(("HEAVY", pair["heavy"], args.heavy_weight, "heavy_verts.bin"))
    if args.side in ("both", "light"):
        sides.append(("LIGHT", pair["light"], args.light_weight, "light_verts.bin"))

    saved_faces = (out / "faces.bin").exists() and args.side != "both"
    for label, info, override_w, out_name in sides:
        gender_letter = info["gender"]
        smpl_pkl = smpl_dir / GENDER_MODEL[gender_letter]
        if not smpl_pkl.exists():
            sys.exit(f"SMPL model missing: {smpl_pkl}")

        # Map G1 CSV → SOMA-uniform BVH via the parquet so each SMPL plays
        # the same motion as the G1 robot directly below it.
        # --shared-bvh overrides this: both sides use the chosen side's BVH so
        # the two meshes play the SAME motion (only body shape differs).
        src_info = pair[args.shared_bvh] if args.shared_bvh else info
        bvh_rel = g1_csv_to_bvh(src_info["source_csv"], meta)
        bvh_abs = (bs / bvh_rel).resolve()
        if not bvh_abs.exists():
            sys.exit(f"BVH missing: {bvh_abs}")

        print(f"\n  [shape override] real {info['weight']} kg actor "
              f"-> rendered with {override_w} kg for visual balance")

        verts, faces, fps = fit_one(
            label=f"{label} · A{info['uid'][1:]} · {info['gender']} {info['height']:.0f} cm "
                  f"(rendered {override_w} kg)",
            bvh_abs=bvh_abs,
            smpl_pkl=smpl_pkl,
            vposer_dir=vp,
            height_cm=info["height"],
            weight_kg=override_w,
            gender=GENDER_STR[gender_letter],
            bvh2smpl_src=src,
            device=args.device,
            max_frames=args.max_frames,
            free_arms=args.free_arms,
            z_prior_w=args.z_prior_w,
        )
        save_verts(verts, fps, out / out_name)
        if not saved_faces:
            save_faces(faces, out / "faces.bin")
            saved_faces = True

    print("\nDone.")


if __name__ == "__main__":
    main()
