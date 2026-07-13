"""
generate_all_demos.py
=====================
Generates SMPL vertex binaries and copies G1 CSVs for all 15 demo categories.
Run with a Python env that has CUDA torch + smplx installed:
    python generate_all_demos.py

Requires two environment variables pointing to local resources:
    BONES_SEED_DIR  — root of the bones-seed dataset checkout
    BVH2SMPL_SRC    — path to the BVH2SMPL library src/

Output:
  assets/demos/g1_csv/<category>.csv
  assets/demos/smpl/<category>_predicted.bin
  assets/demos/smpl/<category>_gt.bin
  assets/demos/smpl/faces.bin   (shared, written once)
"""

import os, sys, shutil, json, struct
import numpy as np

HERE         = os.path.dirname(os.path.abspath(__file__))
BONES_SEED   = os.environ.get("BONES_SEED_DIR", "<path-to-bones-seed>")
BVH2SMPL_SRC = os.environ.get("BVH2SMPL_SRC",   "<path-to-BVH2SMPL>/src")
SMPL_DIR     = os.path.join(BVH2SMPL_SRC, "rendering_utils", "smpl")
SMPL_MODELS  = {
    "male":   os.path.join(SMPL_DIR, "basicmodel_m_lbs_10_207_0_v1.0.0.pkl"),
    "female": os.path.join(SMPL_DIR, "basicModel_f_lbs_10_207_0_v1.0.0.pkl"),
}

G1_OUT   = os.path.join(HERE, "g1_csv")
SMPL_OUT = os.path.join(HERE, "smpl")
MAX_FRAMES = 150   # ~12MB per file; dancing keeps its existing 300-frame file

# ── Demo definitions ──────────────────────────────────────────────────────
# fmt: (category_id, label, count, g1_csv, bvh, gt_attrs, pred_attrs)
# gt_attrs / pred_attrs: {height, weight, age, gender}

DEMOS = [
    dict(id="dancing",      label="Dancing",      count=12593,
         g1 =os.path.join(BONES_SEED, r"g1\csv\240529\macarena_001__A545_M.csv"),
         bvh=os.path.join(BONES_SEED, r"soma_uniform\bvh\240529\macarena_001__A545_M.bvh"),
         gt  =dict(height=172, weight=68, age=28, gender="female"),
         pred=dict(height=168, weight=68, age=29, gender="female")),
    dict(id="walking",      label="Walking",      count=20222,
         g1 =os.path.join(BONES_SEED, r"g1\csv\240918\grab_walk_ff_180_001__A548.csv"),
         bvh=os.path.join(BONES_SEED, r"soma_uniform\bvh\240918\grab_walk_ff_180_001__A548.bvh"),
         gt  =dict(height=171, weight=62, age=29, gender="female"),
         pred=dict(height=175, weight=57, age=31, gender="female")),
    dict(id="jogging",      label="Jogging",      count=16437,
         g1 =os.path.join(BONES_SEED, r"g1\csv\240918\smoke_jog_ff_360_stop_R_001__A548.csv"),
         bvh=os.path.join(BONES_SEED, r"soma_uniform\bvh\240918\smoke_jog_ff_360_stop_R_001__A548.bvh"),
         gt  =dict(height=171, weight=62, age=29, gender="female"),
         pred=dict(height=168, weight=65, age=27, gender="female")),
    dict(id="gesture",      label="Gesture",      count=15862,
         g1 =os.path.join(BONES_SEED, r"g1\csv\240529\neutral_alone_R_001__A542.csv"),
         bvh=os.path.join(BONES_SEED, r"soma_uniform\bvh\240529\neutral_alone_R_001__A542.bvh"),
         gt  =dict(height=179, weight=77, age=25, gender="male"),
         pred=dict(height=176, weight=72, age=22, gender="male")),
    dict(id="tennis",       label="Playing Tennis", count=820,
         g1 =os.path.join(BONES_SEED, r"g1\csv\240327\play_tennis_R_002__A533_M.csv"),
         bvh=os.path.join(BONES_SEED, r"soma_uniform\bvh\240327\play_tennis_R_002__A533_M.bvh"),
         gt  =dict(height=171, weight=71, age=27, gender="male"),
         pred=dict(height=175, weight=67, age=30, gender="male")),
    dict(id="jumping",      label="Jumping",      count=11475,
         g1 =os.path.join(BONES_SEED, r"g1\csv\240527\neutral_dancecard_jump_002__A534.csv"),
         bvh=os.path.join(BONES_SEED, r"soma_uniform\bvh\240527\neutral_dancecard_jump_002__A534.bvh"),
         gt  =dict(height=179, weight=77, age=25, gender="male"),
         pred=dict(height=183, weight=83, age=28, gender="male")),
    dict(id="sitting",      label="Sitting",      count=5601,
         g1 =os.path.join(BONES_SEED, r"g1\csv\240918\sit_on_heels_loop_009__A548.csv"),
         bvh=os.path.join(BONES_SEED, r"soma_uniform\bvh\240918\sit_on_heels_loop_009__A548.bvh"),
         gt  =dict(height=171, weight=62, age=29, gender="female"),
         pred=dict(height=174, weight=58, age=32, gender="female")),
    dict(id="turning",      label="Turning",      count=3472,
         g1 =os.path.join(BONES_SEED, r"g1\csv\240918\idle_turn_000_R_long_002__A548.csv"),
         bvh=os.path.join(BONES_SEED, r"soma_uniform\bvh\240918\idle_turn_000_R_long_002__A548.bvh"),
         gt  =dict(height=171, weight=62, age=29, gender="female"),
         pred=dict(height=175, weight=58, age=32, gender="female")),
    dict(id="climbing_box", label="Climbing Box", count=3402,
         g1 =os.path.join(BONES_SEED, r"g1\csv\240529\neutral_come_down_50cm_box_R_001__A542.csv"),
         bvh=os.path.join(BONES_SEED, r"soma_uniform\bvh\240529\neutral_come_down_50cm_box_R_001__A542.bvh"),
         gt  =dict(height=179, weight=77, age=25, gender="male"),
         pred=dict(height=175, weight=81, age=29, gender="male")),
    dict(id="kneeling",     label="Kneeling",     count=1731,
         g1 =os.path.join(BONES_SEED, r"g1\csv\230713\knightly_bow_R_001__A429.csv"),
         bvh=os.path.join(BONES_SEED, r"soma_uniform\bvh\230713\knightly_bow_R_001__A429.bvh"),
         gt  =dict(height=178, weight=90, age=25, gender="male"),
         pred=dict(height=173, weight=83, age=28, gender="male")),
    dict(id="hiphop",       label="Hip-Hop Dance", count=540,
         g1 =os.path.join(BONES_SEED, r"g1\csv\230412\dance_hiphop_bart_simpson_R_002__A313_M.csv"),
         bvh=os.path.join(BONES_SEED, r"soma_uniform\bvh\230412\dance_hiphop_bart_simpson_R_002__A313_M.bvh"),
         gt  =dict(height=175, weight=76, age=22, gender="male"),
         pred=dict(height=172, weight=72, age=25, gender="male")),
    dict(id="pulling",      label="Pulling",      count=702,
         g1 =os.path.join(BONES_SEED, r"g1\csv\231019\high_big_crank_ccw_002__A484.csv"),
         bvh=os.path.join(BONES_SEED, r"soma_uniform\bvh\231019\high_big_crank_ccw_002__A484.bvh"),
         gt  =dict(height=167, weight=52, age=29, gender="female"),
         pred=dict(height=163, weight=46, age=33, gender="female")),
    dict(id="throwing",     label="Throwing",     count=316,
         g1 =os.path.join(BONES_SEED, r"g1\csv\230424\throw_ball_R_001__A345.csv"),
         bvh=os.path.join(BONES_SEED, r"soma_uniform\bvh\230424\throw_ball_R_001__A345.bvh"),
         gt  =dict(height=168, weight=61, age=27, gender="female"),
         pred=dict(height=172, weight=68, age=30, gender="female")),
    dict(id="guitar",       label="Guitar",       count=200,
         g1 =os.path.join(BONES_SEED, r"g1\csv\230417\playing_guitar_R_001__A330.csv"),
         bvh=os.path.join(BONES_SEED, r"soma_uniform\bvh\230417\playing_guitar_R_001__A330.bvh"),
         gt  =dict(height=169, weight=61, age=27, gender="female"),
         pred=dict(height=172, weight=58, age=30, gender="female")),
]

# ── Helpers ───────────────────────────────────────────────────────────────

def save_verts(verts, fps, path):
    nf, nv, _ = verts.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<II", nf, nv))
        f.write(struct.pack("<f", float(fps)))
        f.write(verts.astype(np.float32).ravel().tobytes())
    mb = os.path.getsize(path) / 1e6
    print(f"    -> {os.path.basename(path)}  ({nf}fr, {mb:.1f}MB)")

def save_faces(faces, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(faces)))
        f.write(faces.astype(np.uint32).ravel().tobytes())
    print(f"    -> {os.path.basename(path)}")

def run_smpl(bvh_path, height, weight, gender, max_frames, device):
    sys.path.insert(0, BVH2SMPL_SRC)
    os.chdir(BVH2SMPL_SRC)
    from soma_viewer import BVHMotionZYX, fit_smpl
    bvh_path = os.path.abspath(bvh_path)
    bvh = BVHMotionZYX(bvh_path)
    nf  = bvh.motion_length
    print(f"    BVH: {os.path.basename(bvh_path)}  ({nf} frames)")
    verts, faces = fit_smpl(
        bvh, scale=100.0,
        smpl_path=SMPL_MODELS[gender], device=device,
        height_cm=height, weight_kg=weight, gender=gender,
    )
    if nf > max_frames:
        idx = np.round(np.linspace(0, nf-1, max_frames)).astype(int)
        verts = verts[idx]
        eff_fps = 120.0 * max_frames / nf
    else:
        eff_fps = 120.0
    return verts, faces, eff_fps


def main():
    device = "cpu"   # SMPL model has buffers that don't transfer cleanly to CUDA
    print(f"Device: {device}\n")

    faces_saved = False
    config_out  = []

    # Sort by count descending (dancing first so it's the default)
    demos = sorted(DEMOS, key=lambda d: -d["count"])

    for demo in demos:
        did   = demo["id"]
        label = demo["label"]
        print(f"\n{'='*60}")
        print(f"  {label.upper()}  (id={did}, count={demo['count']:,})")
        print(f"{'='*60}")

        # ── Copy G1 CSV ──────────────────────────────────────────
        g1_src  = demo["g1"]
        g1_dst  = os.path.join(G1_OUT, f"{did}.csv")
        if not os.path.exists(g1_src):
            print(f"  [SKIP] G1 CSV not found: {g1_src}")
            continue
        shutil.copy2(g1_src, g1_dst)
        print(f"  G1 CSV -> {g1_dst}")

        # ── SMPL: dancing already has files, skip re-generation ──
        pred_bin = os.path.join(SMPL_OUT, f"{did}_predicted.bin")
        gt_bin   = os.path.join(SMPL_OUT, f"{did}_gt.bin")

        if did == "dancing" and os.path.exists(pred_bin) and os.path.exists(gt_bin):
            print("  [DANCING] Keeping existing SMPL files.")
            # Copy faces if not yet saved
            existing_faces = os.path.join(
                os.path.dirname(HERE), "smpl_engine", "data", "smpl_faces.bin")
            faces_dst = os.path.join(SMPL_OUT, "faces.bin")
            if not faces_saved and os.path.exists(existing_faces):
                shutil.copy2(existing_faces, faces_dst)
                print(f"  Faces -> {faces_dst}")
                faces_saved = True
        else:
            mf = MAX_FRAMES
            try:
                print("  Fitting PREDICTED...")
                p = demo["pred"]
                vp, faces, fps = run_smpl(demo["bvh"], p["height"], p["weight"], p["gender"], mf, device)
                save_verts(vp, fps, pred_bin)
                if not faces_saved:
                    save_faces(faces, os.path.join(SMPL_OUT, "faces.bin"))
                    faces_saved = True

                print("  Fitting GT...")
                g = demo["gt"]
                vg, _, fps2 = run_smpl(demo["bvh"], g["height"], g["weight"], g["gender"], mf, device)
                save_verts(vg, fps2, gt_bin)
            except Exception as e:
                print(f"  [ERROR] {e}")
                continue

        # Count frames
        import struct as st
        with open(pred_bin,"rb") as f:
            pnf = st.unpack("<I",f.read(4))[0]; st.unpack("<I",f.read(4))[0]; pfps=st.unpack("<f",f.read(4))[0]

        # Forward-slash relative paths into the bones-seed dataset so the
        # config documents where each demo came from.
        src_g1  = os.path.relpath(demo["g1"],  BONES_SEED).replace("\\", "/")
        src_bvh = os.path.relpath(demo["bvh"], BONES_SEED).replace("\\", "/")

        config_out.append({
            "id":    did,
            "label": label,
            "count": demo["count"],
            "g1CsvFile": f"./g1_csv/{did}.csv",
            "smplPredFile": f"./smpl/{did}_predicted.bin",
            "smplGtFile":   f"./smpl/{did}_gt.bin",
            "smplFacesFile": "./smpl/faces.bin",
            "sourceG1Csv": src_g1,
            "sourceBvh":   src_bvh,
            "numFrames": pnf,
            "fps": pfps,
            "predicted": demo["pred"],
            "groundTruth": demo["gt"],
        })

    # Write demos_config.json
    cfg = {"defaultDemo": "dancing", "demos": config_out}
    out_path = os.path.join(HERE, "demos_config.json")
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nWrote {out_path}  ({len(config_out)} demos)")


if __name__ == "__main__":
    main()
