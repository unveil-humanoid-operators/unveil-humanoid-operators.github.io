"""
generate_all_demos_vposer.py
============================
Same demo set + same height/weight/gender attributes as generate_all_demos.py,
but the per-frame SMPL pose is fit through VPoser's latent space
(`soma_viewer_vposer.fit_smpl_vposer`). VPoser is a VAE trained on AMASS, so
decoded poses land in the manifold of plausible human poses — eliminates the
inverted-knee / hyperextended-elbow artifacts of the unconstrained
position-only fit.

Run with a Python env that has torch + smplx + human_body_prior installed.

Requires:
    BONES_SEED_DIR  -- root of the bones-seed dataset checkout
    BVH2SMPL_SRC    -- path to the BVH2SMPL library src/
    VPOSER_PATH     -- path to the unpacked VPoser v1.0 (SMPL) experiment dir
                        (must contain snapshots/*.pt and *.ini)

Output:
    assets/demos/g1_csv/<category>.csv
    assets/demos/smpl/<category>_predicted.bin
    assets/demos/smpl/<category>_gt.bin
    assets/demos/smpl/faces.bin     (shared, written once)
    assets/demos/demos_config.json  (rewritten with the regenerated set)

Regenerates every demo, including dancing (no skip).
"""

import os, sys, shutil, json, struct, subprocess
import numpy as np

HERE         = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.normpath(os.path.join(HERE, "..", ".."))
BONES_SEED   = os.environ.get("BONES_SEED_DIR", "<path-to-bones-seed>")
BVH2SMPL_SRC = os.environ.get("BVH2SMPL_SRC",   "<path-to-BVH2SMPL>/src")
VPOSER_PATH  = os.environ.get("VPOSER_PATH",    "")
DEVICE       = os.environ.get("VPOSER_DEVICE",  "cuda")
# Set PUSH_PER_DEMO=0 in the env to keep all artifacts local and skip git.
PUSH_PER_DEMO = os.environ.get("PUSH_PER_DEMO", "1") != "0"
# Comma-separated subset to rebuild (e.g. INCLUDE_IDS=jumping,throwing).
# Empty means rebuild every demo in DEMOS.
INCLUDE_IDS   = {s.strip() for s in os.environ.get("INCLUDE_IDS", "").split(",")
                 if s.strip()}
# Comma-separated ids to skip (applied after INCLUDE_IDS filter).
EXCLUDE_IDS   = {s.strip() for s in os.environ.get("EXCLUDE_IDS", "").split(",")
                 if s.strip()}
# Toggle SMPL arm-joint free-optimisation (bypasses VPoser prior on
# shoulders / elbows / wrists). Fixes "arms stop at the shoulder" and
# overhead-reach restrictions for dance / throw / itching-head motions.
FREE_ARMS     = os.environ.get("FREE_ARMS", "0") == "1"

SMPL_DIR     = os.path.join(BVH2SMPL_SRC, "rendering_utils", "smpl")
SMPL_MODELS  = {
    "male":   os.path.join(SMPL_DIR, "basicmodel_m_lbs_10_207_0_v1.0.0.pkl"),
    "female": os.path.join(SMPL_DIR, "basicModel_f_lbs_10_207_0_v1.0.0.pkl"),
}

G1_OUT     = os.path.join(HERE, "g1_csv")
SMPL_OUT   = os.path.join(HERE, "smpl")
MAX_FRAMES = 150

# Reuse the same demo definitions as the baseline script.
sys.path.insert(0, HERE)
from generate_all_demos import DEMOS, save_verts, save_faces  # noqa: E402


def run_smpl_vposer(bvh_path, height, weight, gender, max_frames, device,
                    vposer_dir):
    sys.path.insert(0, BVH2SMPL_SRC)
    os.chdir(BVH2SMPL_SRC)
    from soma_viewer import BVHMotionZYX
    from soma_viewer_vposer import fit_smpl_vposer

    bvh_path = os.path.abspath(bvh_path)
    bvh = BVHMotionZYX(bvh_path)
    nf  = bvh.motion_length
    print(f"    BVH: {os.path.basename(bvh_path)}  ({nf} frames)")
    verts, faces = fit_smpl_vposer(
        bvh, scale=100.0,
        smpl_path=SMPL_MODELS[gender], vposer_dir=vposer_dir, device=device,
        height_cm=height, weight_kg=weight, gender=gender, uid=None,
        # Boost upper-arm chain weights so the data loss out-pulls VPoser's
        # z prior on extreme reaches (hand-to-head, hand-far-from-body).
        # Wrists are the bottleneck for hand reach — they get the biggest bump.
        elbow_weight=2.5,
        wrist_weight=6.0,
        hand_weight=1.5,
        free_arms=FREE_ARMS,
    )
    if nf > max_frames:
        idx = np.round(np.linspace(0, nf-1, max_frames)).astype(int)
        verts = verts[idx]
        eff_fps = 120.0 * max_frames / nf
    else:
        eff_fps = 120.0
    return verts, faces, eff_fps


def _git(*args):
    """Run a git command in REPO_ROOT, capturing output."""
    return subprocess.run(
        ["git", "-C", REPO_ROOT, *args],
        check=True, capture_output=True, text=True,
    )


def _git_has_staged_changes():
    """Return True iff there's anything staged to commit."""
    r = subprocess.run(
        ["git", "-C", REPO_ROOT, "diff", "--cached", "--quiet"],
    )
    return r.returncode != 0   # quiet: 0 = no diff, 1 = diff


def push_demo(did, idx, total):
    """git add + commit + push the artifacts for one finished demo.
    Silently no-ops if nothing changed or PUSH_PER_DEMO is disabled."""
    if not PUSH_PER_DEMO:
        return
    rel_paths = [
        f"assets/demos/smpl/{did}_predicted.bin",
        f"assets/demos/smpl/{did}_gt.bin",
        f"assets/demos/smpl/faces.bin",
        f"assets/demos/g1_csv/{did}.csv",
        "assets/demos/demos_config.json",
        "smpl_engine/viewer.html",
    ]
    try:
        _git("add", *rel_paths)
        if not _git_has_staged_changes():
            print(f"  [GIT] {did}: no changes to commit")
            return
        _git("commit", "-m",
             f"Rebuild {did} ({idx}/{total}) — VPoser + wrist-weighted fit")
        _git("push")
        print(f"  [GIT] {did}: committed + pushed")
    except subprocess.CalledProcessError as e:
        out = (e.stderr or e.stdout or str(e)).strip()
        print(f"  [GIT ERROR] {did}: {out}")


def _write_config(config_by_id, config_path):
    """Write demos_config.json sorted by count descending (dancing default)."""
    demos_list = sorted(config_by_id.values(), key=lambda d: -d.get("count", 0))
    cfg = {"defaultDemo": "dancing", "demos": demos_list}
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)


def main():
    if not os.path.isdir(VPOSER_PATH):
        sys.exit(f"ERROR: VPOSER_PATH not a valid dir: {VPOSER_PATH!r}")
    print(f"Device         : {DEVICE}")
    print(f"VPoser dir     : {VPOSER_PATH}")
    print(f"Push per demo  : {PUSH_PER_DEMO}\n")

    # Carry forward whatever the live demos_config.json already has, so a
    # partial regen (or one with a new id mid-way) does not drop earlier
    # entries pointing at still-valid bins.
    config_path = os.path.join(HERE, "demos_config.json")
    config_by_id = {}
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                existing = json.load(f)
            for d in existing.get("demos", []):
                config_by_id[d["id"]] = d
        except Exception as e:
            print(f"  [WARN] failed to read existing config: {e}")

    faces_saved = os.path.exists(os.path.join(SMPL_OUT, "faces.bin"))

    # Sort by count descending (dancing first so it's the default)
    demos = sorted(DEMOS, key=lambda d: -d["count"])
    if INCLUDE_IDS:
        demos = [d for d in demos if d["id"] in INCLUDE_IDS]
        print(f"INCLUDE_IDS    : {sorted(INCLUDE_IDS)}")
    if EXCLUDE_IDS:
        demos = [d for d in demos if d["id"] not in EXCLUDE_IDS]
        print(f"EXCLUDE_IDS    : {sorted(EXCLUDE_IDS)}")
    print(f"Free arms      : {FREE_ARMS}")
    print(f"Will rebuild   : {[d['id'] for d in demos]}\n")
    total = len(demos)

    for idx, demo in enumerate(demos, 1):
        did   = demo["id"]
        label = demo["label"]
        print(f"\n{'='*60}")
        print(f"  [{idx}/{total}] {label.upper()}  (id={did}, count={demo['count']:,})")
        print(f"{'='*60}")

        g1_src = demo["g1"]
        g1_dst = os.path.join(G1_OUT, f"{did}.csv")
        if not os.path.exists(g1_src):
            print(f"  [SKIP] G1 CSV not found: {g1_src}")
            continue
        shutil.copy2(g1_src, g1_dst)
        print(f"  G1 CSV -> {g1_dst}")

        pred_bin = os.path.join(SMPL_OUT, f"{did}_predicted.bin")
        gt_bin   = os.path.join(SMPL_OUT, f"{did}_gt.bin")

        try:
            print("  Fitting PREDICTED (VPoser)...")
            p = demo["pred"]
            vp, faces, fps = run_smpl_vposer(
                demo["bvh"], p["height"], p["weight"], p["gender"],
                MAX_FRAMES, DEVICE, VPOSER_PATH,
            )
            save_verts(vp, fps, pred_bin)
            if not faces_saved:
                save_faces(faces, os.path.join(SMPL_OUT, "faces.bin"))
                faces_saved = True

            print("  Fitting GT (VPoser)...")
            g = demo["gt"]
            vg, _, fps2 = run_smpl_vposer(
                demo["bvh"], g["height"], g["weight"], g["gender"],
                MAX_FRAMES, DEVICE, VPOSER_PATH,
            )
            save_verts(vg, fps2, gt_bin)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        with open(pred_bin, "rb") as f:
            pnf = struct.unpack("<I", f.read(4))[0]
            struct.unpack("<I", f.read(4))[0]
            pfps = struct.unpack("<f", f.read(4))[0]

        config_by_id[did] = {
            "id":            did,
            "label":         label,
            "count":         demo["count"],
            "g1CsvFile":     f"./g1_csv/{did}.csv",
            "smplPredFile":  f"./smpl/{did}_predicted.bin",
            "smplGtFile":    f"./smpl/{did}_gt.bin",
            "smplFacesFile": "./smpl/faces.bin",
            "numFrames":     pnf,
            "fps":           pfps,
            "predicted":     demo["pred"],
            "groundTruth":   demo["gt"],
        }

        _write_config(config_by_id, config_path)
        push_demo(did, idx, total)

    print(f"\nDone. {len(config_by_id)} demos in {config_path}.")


if __name__ == "__main__":
    main()
