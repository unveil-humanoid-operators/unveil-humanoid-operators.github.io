"""
generate_smpl_turning.py
------------------------
Fits two SMPL meshes for the canonical turning clip used on the project page
(demos_config.json 'turning' → idle_turn_000_R_long_002__A548).

Both meshes share that source BVH; only body-shape params differ:
    older  = 165 cm female · 70 kg  (middle-age build)
    young  = 165 cm female · 55 kg  (lean young-adult build)

Age itself doesn't affect SMPL betas — that's encoded later in twin_view_turning.html
by scaling the OLDER operator's waist angles around their mean (smaller ROM).

Outputs into ./data/smpl_turning/:
    older_verts.bin
    young_verts.bin
    faces.bin

Usage:
  python generate_smpl_turning.py \
      --bones-seed C:/Users/sihat/Downloads/bones-seed \
      --bvh2smpl-src C:/Users/sihat/Downloads/BVH2SMPL/src \
      --vposer-dir  C:/Users/sihat/Downloads/BVH2SMPL/vposer_v1_0 \
      --device cuda
"""

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np


HERO_BVH_REL = "soma_uniform/bvh/240918/idle_turn_000_R_long_002__A548.bvh"

OLDER_SHAPE = dict(height_cm=165.0, weight_kg=70.0, gender="female", label="OLDER")
YOUNG_SHAPE = dict(height_cm=165.0, weight_kg=55.0, gender="female", label="YOUNG")

GENDER_MODEL = {
    "male":   "basicmodel_m_lbs_10_207_0_v1.0.0.pkl",
    "female": "basicModel_f_lbs_10_207_0_v1.0.0.pkl",
}


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


def fit_one(label, bvh_abs, smpl_pkl, vposer_dir, height_cm, weight_kg,
            gender, bvh2smpl_src, device, max_frames, free_arms, z_prior_w,
            max_input_frames):
    from soma_viewer_vposer import fit_smpl_vposer
    from soma_viewer import BVHMotionZYX
    import torch

    print(f"\n[{label}]")
    print(f"  BVH    : {bvh_abs}")
    print(f"  Shape  : {gender} · {height_cm} cm · {weight_kg} kg")
    print(f"  Device : {device}  free_arms={free_arms}  z_prior_w={z_prior_w}")

    os.chdir(str(bvh2smpl_src))
    bvh = BVHMotionZYX(str(bvh_abs))
    print(f"  Frames : {bvh.motion_length} @ 120 fps (source)")
    # The turning BVH is ~70 s / 8420 frames — fitting all of it through
    # VPoser+free-arms takes hours. Truncate to a representative middle slab.
    # motion_length is a @property derived from joint_position.shape[0], so
    # slicing joint_position+joint_rotation is enough; FK will be re-run.
    if bvh.motion_length > max_input_frames:
        start = (bvh.motion_length - max_input_frames) // 2
        end = start + max_input_frames
        bvh.joint_position = bvh.joint_position[start:end]
        bvh.joint_rotation = bvh.joint_rotation[start:end]
        # If FK was pre-run on the full BVH, clear the cached globals so
        # batch_forward_kinematics() recomputes on the truncated slice.
        bvh.joint_translation = None
        bvh.joint_orientation = None
        print(f"  -> truncated to middle {max_input_frames} frames "
              f"({start}..{end}, {max_input_frames/120.0:.1f} s)")

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
    verts = np.asarray(verts); faces = np.asarray(faces)
    n = len(verts)
    if n > max_frames:
        idx = np.round(np.linspace(0, n - 1, max_frames)).astype(int)
        verts = verts[idx]
        fps = 120.0 * max_frames / n
    else:
        fps = 120.0
    print(f"  Output : {len(verts)} frames @ {fps:.2f} fps")
    return verts, faces, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bones-seed", required=True, type=Path)
    ap.add_argument("--bvh2smpl-src", required=True, type=Path)
    ap.add_argument("--vposer-dir", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "data" / "smpl_turning")
    ap.add_argument("--free-arms", action="store_true", default=True)
    ap.add_argument("--z-prior-w", type=float, default=0.001)
    ap.add_argument("--max-input-frames", type=int, default=2400,
                    help="Truncate BVH source to this many frames before fitting. "
                         "The turning BVH is ~8420 frames; full-length fits take hours.")
    args = ap.parse_args()

    bs = args.bones_seed.resolve()
    src = args.bvh2smpl_src.resolve()
    vp = args.vposer_dir.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(src))
    smpl_dir = src / "rendering_utils" / "smpl"

    bvh_abs = bs / HERO_BVH_REL
    if not bvh_abs.exists():
        sys.exit(f"BVH missing: {bvh_abs}")

    saved_faces = False
    for shape, out_name in [(OLDER_SHAPE, "older_verts.bin"),
                            (YOUNG_SHAPE, "young_verts.bin")]:
        smpl_pkl = smpl_dir / GENDER_MODEL[shape["gender"]]
        if not smpl_pkl.exists():
            sys.exit(f"SMPL model missing: {smpl_pkl}")

        verts, faces, fps = fit_one(
            label=f"{shape['label']} · {shape['weight_kg']}kg {shape['gender']} @ {shape['height_cm']}cm",
            bvh_abs=bvh_abs,
            smpl_pkl=smpl_pkl,
            vposer_dir=vp,
            height_cm=shape["height_cm"],
            weight_kg=shape["weight_kg"],
            gender=shape["gender"],
            bvh2smpl_src=src,
            device=args.device,
            max_frames=args.max_frames,
            free_arms=args.free_arms,
            z_prior_w=args.z_prior_w,
            max_input_frames=args.max_input_frames,
        )
        save_verts(verts, fps, out / out_name)
        if not saved_faces:
            save_faces(faces, out / "faces.bin")
            saved_faces = True

    print("\nDone. Both turning SMPL fits used idle_turn_000_R_long_002__A548.bvh.")


if __name__ == "__main__":
    main()
