"""
generate_smpl.py
================
Pre-compute SMPL vertex sequences from a SOMA BVH file.
Reads smpl_engine/config.json, runs fit_smpl for both predicted and ground-truth
body shapes, writes compact binary files to smpl_engine/data/.

Run with a Python env that has torch + smplx + scipy, e.g.:
    python generate_smpl.py

Set BVH2SMPL_SRC below to the absolute path of your BVH2SMPL/src directory.

Output files (in smpl_engine/data/):
    predicted_verts.bin     float32 vertex sequences for InveRT prediction
    groundtruth_verts.bin   float32 vertex sequences for ground truth
    smpl_faces.bin          uint32 triangle indices (shared; same for all SMPL)

Binary format for *_verts.bin:
    bytes 0-3  : uint32  num_frames
    bytes 4-7  : uint32  num_verts   (always 6890 for SMPL)
    bytes 8-11 : float32 fps         (effective fps after downsampling)
    bytes 12+  : float32[num_frames * num_verts * 3]  (X,Y,Z per vertex per frame)
"""

import os
import sys
import json
import struct
import numpy as np

# ── locate this file ──────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── BVH2SMPL source — set this to the absolute path of your BVH2SMPL/src ─
# Example: r"C:\path\to\BVH2SMPL\src"
BVH2SMPL_SRC  = os.environ.get("BVH2SMPL_SRC", r"BVH2SMPL/src")
SMPL_MODEL_DIR = os.path.join(BVH2SMPL_SRC, "rendering_utils", "smpl")
sys.path.insert(0, BVH2SMPL_SRC)

GENDER_MODEL = {
    "male":   "basicmodel_m_lbs_10_207_0_v1.0.0.pkl",
    "female": "basicModel_f_lbs_10_207_0_v1.0.0.pkl",
}


def load_config():
    path = os.path.join(HERE, "config.json")
    with open(path) as f:
        raw = f.read()
    # Strip // comments (not valid JSON but convenient for users)
    import re
    raw = re.sub(r'//.*', '', raw)
    return json.loads(raw)


def save_verts_bin(verts: np.ndarray, fps: float, out_path: str):
    """Write (nframes, nverts, 3) float32 array as binary file."""
    nframes, nverts, _ = verts.shape
    with open(out_path, "wb") as f:
        f.write(struct.pack("<II", nframes, nverts))
        f.write(struct.pack("<f", fps))
        f.write(verts.astype(np.float32).ravel().tobytes())
    mb = os.path.getsize(out_path) / 1e6
    print(f"    -> {out_path}  ({nframes} frames, {mb:.1f} MB)")


def save_faces_bin(faces: np.ndarray, out_path: str):
    """Write (nfaces, 3) uint32 face array as binary file."""
    nfaces = len(faces)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", nfaces))
        f.write(faces.astype(np.uint32).ravel().tobytes())
    kb = os.path.getsize(out_path) / 1e3
    print(f"    -> {out_path}  ({nfaces} faces, {kb:.0f} KB)")


def run_sequence(bvh_path, scale, smpl_path, height, weight, gender,
                 max_frames, label):
    from soma_viewer import BVHMotionZYX, fit_smpl

    print(f"\n[{label}]")
    print(f"  BVH    : {os.path.basename(bvh_path)}")
    print(f"  Shape  : height={height} cm  weight={weight} kg  gender={gender}")

    # Resolve bvh_path to absolute BEFORE chdir (chdir makes relative paths break)
    bvh_path = os.path.abspath(bvh_path)

    # soma_viewer loads SMPL assets with relative paths from BVH2SMPL/src
    os.chdir(BVH2SMPL_SRC)

    bvh = BVHMotionZYX(bvh_path)
    print(f"  Frames : {bvh.motion_length} @ 120 fps")

    verts, faces = fit_smpl(
        bvh,
        scale=scale,
        smpl_path=smpl_path,
        device="cpu",
        height_cm=height,
        weight_kg=weight,
        gender=gender,
    )

    # Downsample to max_frames to keep files manageable
    nframes = len(verts)
    if nframes > max_frames:
        indices = np.round(np.linspace(0, nframes - 1, max_frames)).astype(int)
        verts = verts[indices]
        effective_fps = 120.0 * max_frames / nframes
    else:
        effective_fps = 120.0

    print(f"  Output : {len(verts)} frames @ {effective_fps:.1f} fps")
    return verts, faces, effective_fps


def main():
    cfg        = load_config()
    raw_bvh    = cfg["bvhFile"]
    # Resolve relative paths from smpl_engine/ directory
    bvh_path   = raw_bvh if os.path.isabs(raw_bvh) else os.path.join(HERE, raw_bvh)
    scale      = cfg.get("bvhScale", 100.0)
    max_frames = cfg.get("maxFrames", 300)

    if not os.path.exists(bvh_path):
        sys.exit(f"ERROR: BVH file not found:\n  {bvh_path}")

    faces_saved = False

    for key, label in [("predicted", "InveRT Predicted"),
                        ("groundTruth", "Ground Truth")]:
        seq = cfg.get(key)
        if not seq:
            continue

        gender    = seq["gender"]
        model_pkl = os.path.join(SMPL_MODEL_DIR, GENDER_MODEL[gender])
        if not os.path.exists(model_pkl):
            print(f"  WARNING: SMPL model not found: {model_pkl} — skipping {label}")
            continue

        verts, faces, fps = run_sequence(
            bvh_path=bvh_path,
            scale=scale,
            smpl_path=model_pkl,
            height=seq["height"],
            weight=seq["weight"],
            gender=gender,
            max_frames=max_frames,
            label=label,
        )

        out_name = os.path.basename(seq["vertsFile"])
        save_verts_bin(verts, fps, os.path.join(DATA_DIR, out_name))

        if not faces_saved:
            save_faces_bin(faces, os.path.join(DATA_DIR, "smpl_faces.bin"))
            faces_saved = True

    print("\nDone. Open viewer.html?seq=predicted or ?seq=groundtruth in a browser.")


if __name__ == "__main__":
    main()
