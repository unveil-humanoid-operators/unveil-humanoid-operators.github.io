"""
generate_demos_basic_push.py
============================
Uses the ORIGINAL soma_viewer.fit_smpl (no VPoser) to regenerate SMPL bins.
Reads the live demos_config.json for the active demo set, skips a configurable
EXCLUDE list, and commits + pushes each completed activity independently so
the page updates incrementally instead of waiting for the full run.

Env:
    BONES_SEED_DIR -- bones-seed root
    BVH2SMPL_SRC   -- BVH2SMPL src/
    PUSH_PER_DEMO  -- "0" disables per-demo commit+push (default on)

Excludes walking and sitting (their current bins are kept as-is).
"""

import os, sys, shutil, json, struct, subprocess
import numpy as np

HERE         = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.normpath(os.path.join(HERE, "..", ".."))
BONES_SEED   = os.environ.get("BONES_SEED_DIR", "<path-to-bones-seed>")
BVH2SMPL_SRC = os.environ.get("BVH2SMPL_SRC",   "<path-to-BVH2SMPL>/src")
PUSH_PER_DEMO = os.environ.get("PUSH_PER_DEMO", "1") != "0"

SMPL_DIR     = os.path.join(BVH2SMPL_SRC, "rendering_utils", "smpl")
SMPL_MODELS  = {
    "male":   os.path.join(SMPL_DIR, "basicmodel_m_lbs_10_207_0_v1.0.0.pkl"),
    "female": os.path.join(SMPL_DIR, "basicModel_f_lbs_10_207_0_v1.0.0.pkl"),
}

G1_OUT      = os.path.join(HERE, "g1_csv")
SMPL_OUT    = os.path.join(HERE, "smpl")
CONFIG_PATH = os.path.join(HERE, "demos_config.json")
MAX_FRAMES  = 150

EXCLUDE = {"walking", "sitting"}

# Reuse demo metadata + helpers from the canonical script.
sys.path.insert(0, HERE)
from generate_all_demos import DEMOS, save_verts, save_faces  # noqa: E402


def run_smpl_basic(bvh_path, height, weight, gender, max_frames):
    sys.path.insert(0, BVH2SMPL_SRC)
    os.chdir(BVH2SMPL_SRC)
    from soma_viewer import BVHMotionZYX, fit_smpl

    bvh_path = os.path.abspath(bvh_path)
    bvh = BVHMotionZYX(bvh_path)
    nf  = bvh.motion_length
    print(f"    BVH: {os.path.basename(bvh_path)}  ({nf} frames)")
    verts, faces = fit_smpl(
        bvh, scale=100.0,
        smpl_path=SMPL_MODELS[gender], device="cpu",
        height_cm=height, weight_kg=weight, gender=gender,
    )
    if nf > max_frames:
        idx = np.round(np.linspace(0, nf-1, max_frames)).astype(int)
        verts = verts[idx]
        eff_fps = 120.0 * max_frames / nf
    else:
        eff_fps = 120.0
    return verts, faces, eff_fps


def _git(*args):
    return subprocess.run(
        ["git", "-C", REPO_ROOT, *args],
        check=True, capture_output=True, text=True,
    )


def _git_has_staged_changes():
    r = subprocess.run(
        ["git", "-C", REPO_ROOT, "diff", "--cached", "--quiet"],
    )
    return r.returncode != 0


def push_demo(did, idx, total):
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
             f"Rebuild {did} ({idx}/{total}) — basic soma_viewer fit")
        _git("push")
        print(f"  [GIT] {did}: committed + pushed")
    except subprocess.CalledProcessError as e:
        out = (e.stderr or e.stdout or str(e)).strip()
        print(f"  [GIT ERROR] {did}: {out}")


def _write_config(config_by_id, config_path):
    demos_list = sorted(config_by_id.values(), key=lambda d: -d.get("count", 0))
    cfg = {"defaultDemo": "dancing", "demos": demos_list}
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)


def main():
    print(f"Push per demo  : {PUSH_PER_DEMO}")
    print(f"Excluded ids   : {sorted(EXCLUDE)}\n")

    # Carry forward the existing config so excluded demos (walking, sitting)
    # keep their entries pointing at their existing bins.
    config_by_id = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                existing = json.load(f)
            for d in existing.get("demos", []):
                config_by_id[d["id"]] = d
        except Exception as e:
            print(f"  [WARN] failed to read existing config: {e}")

    # Use demos_config.json's set of ids as the universe; intersect with DEMOS
    # for BVH paths; then drop the EXCLUDE list.
    config_ids = set(config_by_id.keys())
    demos = [d for d in DEMOS if d["id"] in config_ids and d["id"] not in EXCLUDE]
    demos.sort(key=lambda d: -d["count"])
    total = len(demos)
    print(f"Will rebuild {total} demos: {[d['id'] for d in demos]}\n")

    faces_saved = os.path.exists(os.path.join(SMPL_OUT, "faces.bin"))

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
            print("  Fitting PREDICTED (basic)...")
            p = demo["pred"]
            vp, faces, fps = run_smpl_basic(
                demo["bvh"], p["height"], p["weight"], p["gender"], MAX_FRAMES,
            )
            save_verts(vp, fps, pred_bin)
            if not faces_saved:
                save_faces(faces, os.path.join(SMPL_OUT, "faces.bin"))
                faces_saved = True

            print("  Fitting GT (basic)...")
            g = demo["gt"]
            vg, _, fps2 = run_smpl_basic(
                demo["bvh"], g["height"], g["weight"], g["gender"], MAX_FRAMES,
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

        _write_config(config_by_id, CONFIG_PATH)
        push_demo(did, idx, total)

    print(f"\nDone. {len(config_by_id)} demos in {CONFIG_PATH}.")


if __name__ == "__main__":
    main()
