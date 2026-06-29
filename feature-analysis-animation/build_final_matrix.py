"""
build_final_matrix.py
---------------------
Read data/final_pairs.json (user-curated 8 pairs across 4 attributes), look up
each (actor, task, content_name) clip from the metadata parquet, copy CSVs into
data/final_matrix/, compute per-clip features, and emit
data/final_matrix_config.json — a feature_matrix_v1 config with 8 "rows"
(one per panel), keyed by panel_id, plus per-attribute groups and a smpl_dir
pointer for the SMPL fits.

Usage:
  python build_final_matrix.py --bones-seed C:/Users/sihat/Downloads/bones-seed
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


GENDER_MAP_TO_SMPL = {"M": "male", "F": "female"}


def feature_unit(feat_name: str) -> str:
    if feat_name.startswith("root_translate"):
        return "cm/s" if feat_name.endswith("vel") else "cm"
    return "deg/s" if feat_name.endswith("vel") else "deg"


# Mirrors select_pair_candidates.JOINT_GROUPS so the JS recipe maps to the same
# columns as bones-seed's compute_clip_features.
JOINT_GROUPS = {
    "root_translate": {"kind": "root_translate", "agg": "multi_axis",
        "joints": ["root_translateX", "root_translateY", "root_translateZ"]},
    "root_rotate":    {"kind": "root_rotate",    "agg": "multi_axis",
        "joints": ["root_rotateX", "root_rotateY", "root_rotateZ"]},
    "hip":      {"kind": "joint_group", "agg": "lr_merge",
        "left":  ["left_hip_pitch_joint_dof", "left_hip_roll_joint_dof", "left_hip_yaw_joint_dof"],
        "right": ["right_hip_pitch_joint_dof", "right_hip_roll_joint_dof", "right_hip_yaw_joint_dof"]},
    "knee":     {"kind": "joint_group", "agg": "lr_merge",
        "left":  ["left_knee_joint_dof"], "right": ["right_knee_joint_dof"]},
    "ankle":    {"kind": "joint_group", "agg": "lr_merge",
        "left":  ["left_ankle_pitch_joint_dof", "left_ankle_roll_joint_dof"],
        "right": ["right_ankle_pitch_joint_dof", "right_ankle_roll_joint_dof"]},
    "waist":    {"kind": "joint_group", "agg": "multi_axis",
        "joints": ["waist_yaw_joint_dof", "waist_roll_joint_dof", "waist_pitch_joint_dof"]},
    "shoulder": {"kind": "joint_group", "agg": "lr_merge",
        "left":  ["left_shoulder_pitch_joint_dof", "left_shoulder_roll_joint_dof", "left_shoulder_yaw_joint_dof"],
        "right": ["right_shoulder_pitch_joint_dof", "right_shoulder_roll_joint_dof", "right_shoulder_yaw_joint_dof"]},
    "elbow":    {"kind": "joint_group", "agg": "lr_merge",
        "left":  ["left_elbow_joint_dof"], "right": ["right_elbow_joint_dof"]},
    "wrist":    {"kind": "joint_group", "agg": "lr_merge",
        "left":  ["left_wrist_roll_joint_dof", "left_wrist_pitch_joint_dof", "left_wrist_yaw_joint_dof"],
        "right": ["right_wrist_roll_joint_dof", "right_wrist_pitch_joint_dof", "right_wrist_yaw_joint_dof"]},
}

def split_feature(feat_name: str) -> tuple[str, str]:
    for stat in ("mean_vel", "peak_vel", "rom"):
        if feat_name.endswith("_" + stat):
            return feat_name[: -(len(stat) + 1)], stat
    raise ValueError(f"unrecognized feature name: {feat_name}")

def build_feature_recipe(feat_name: str) -> dict:
    group, stat = split_feature(feat_name)
    spec = JOINT_GROUPS[group]
    rec = {"name": feat_name, "stat": stat, "kind": spec["kind"], "agg": spec["agg"],
           "unit": feature_unit(feat_name)}
    if spec["agg"] == "lr_merge":
        rec["left"]  = list(spec["left"])
        rec["right"] = list(spec["right"])
    else:
        rec["joints"] = list(spec["joints"])
    return rec


def compute_clip_feature_value(csv_path: Path, compute_fn, feat_name: str) -> float | None:
    try:
        df = pd.read_csv(csv_path)
        if "Frame" in df.columns:
            df = df.drop(columns=["Frame"])
        x = df.to_numpy(dtype=np.float32)
        if x.shape[1] < 35: return None
        return float(compute_fn(x, fps=120.0)[feat_name])
    except Exception as e:
        print(f"  ! per-clip compute failed for {csv_path.name}: {e}", file=sys.stderr)
        return None


def attribute_label(actor_row: pd.Series, attribute: str, attribute_unit: str) -> str:
    if attribute == "gender_numeric":
        return "male" if actor_row.get("actor_gender") == "M" else "female"
    cols = {"actor_age_yr": "actor_age_yr",
            "actor_weight_kg": "actor_weight_kg",
            "actor_height_cm": "actor_height_cm"}
    val = float(actor_row[cols[attribute]])
    return f"{val:.0f} {attribute_unit}".strip()


def find_actor_clip(meta: pd.DataFrame, actor_uid: str, task: str,
                    content_name: str | None,
                    min_frames: int = 200, style: str = "neutral") -> pd.Series | None:
    """Find the actor's longest neutral, non-mirror clip of the task. If
    content_name is given, restrict to clips with that exact content_name."""
    sub = meta[(meta["actor_uid"] == actor_uid)
             & (meta["content_type_of_movement"] == task)
             & (~meta["move_g1_mujoco_path"].astype(str).str.endswith("_M.csv"))
             & (meta["content_uniform_style"].astype(str) == style)
             & (pd.to_numeric(meta["move_duration_frames"], errors="coerce").fillna(0) >= min_frames)]
    if content_name is not None:
        sub = sub[sub["content_name"] == content_name]
    if not len(sub): return None
    sub = sub.copy()
    sub["nf"] = pd.to_numeric(sub["move_duration_frames"], errors="coerce")
    sub = sub.sort_values("nf", ascending=False)
    return sub.iloc[0]


def find_best_common_content_name(meta: pd.DataFrame, bs: Path,
                                  high_actor: str, low_actor: str, task: str,
                                  feature_name: str, compute_fn,
                                  min_frames: int = 200, style: str = "neutral") -> str | None:
    """Pick a content_name both actors have neutral clips of, maximizing |F_hi - F_lo|."""
    base = meta[(meta["content_type_of_movement"] == task)
              & (~meta["move_g1_mujoco_path"].astype(str).str.endswith("_M.csv"))
              & (meta["content_uniform_style"].astype(str) == style)
              & (pd.to_numeric(meta["move_duration_frames"], errors="coerce").fillna(0) >= min_frames)]
    his = set(base[base["actor_uid"] == high_actor]["content_name"].unique())
    los = set(base[base["actor_uid"] == low_actor ]["content_name"].unique())
    common = his & los
    if not common: return None
    best_cn, best_gap = None, -1.0
    for cn in common:
        # each actor's longest clip with this cn
        for actor, label in ((high_actor, "hi"), (low_actor, "lo")):
            sub = base[(base["actor_uid"] == actor) & (base["content_name"] == cn)]
            sub = sub.copy()
            sub["nf"] = pd.to_numeric(sub["move_duration_frames"], errors="coerce")
            sub = sub.sort_values("nf", ascending=False)
            csv_rel = str(sub.iloc[0]["move_g1_mujoco_path"]).replace("\\", "/")
            f = compute_clip_feature_value(bs / csv_rel, compute_fn, feature_name)
            if label == "hi": f_hi = f
            else: f_lo = f
        if f_hi is None or f_lo is None: continue
        gap = abs(f_hi - f_lo)
        if gap > best_gap:
            best_gap, best_cn = gap, cn
    return best_cn


def materialize_side(meta: pd.DataFrame, bs: Path, panel_id: str, side: str,
                     actor_uid: str, task: str, content_name: str | None,
                     attribute: str, attribute_unit: str,
                     csv_dir: Path, feature_name: str, compute_fn) -> dict:
    clip = find_actor_clip(meta, actor_uid, task, content_name)
    if clip is None:
        raise SystemExit(
            f"  ! {panel_id} {side}: actor {actor_uid} has no neutral clip of "
            f"task={task!r} content_name={content_name!r}; aborting")
    csv_rel = str(clip["move_g1_mujoco_path"]).replace("\\", "/")
    bvh_rel = str(clip.get("move_soma_uniform_path", "")).replace("\\", "/") or None
    src = bs / csv_rel
    dst = csv_dir / f"{panel_id}_{side}.csv"
    shutil.copyfile(src, dst)
    f_val = compute_clip_feature_value(dst, compute_fn, feature_name)
    shape = {
        "height_cm": int(round(float(clip["actor_height_cm"]))),
        "weight_kg": int(round(float(clip["actor_weight_kg"]))),
        "gender": GENDER_MAP_TO_SMPL.get(str(clip["actor_gender"]), "male"),
    }
    csv_url = "./" + dst.relative_to(Path(__file__).parent).as_posix()
    return {
        "label": attribute_label(clip, attribute, attribute_unit),
        "actor_uid": actor_uid,
        "actor_short": f"{actor_uid} ({clip['actor_gender']}, {int(clip['actor_height_cm'])} cm)",
        "source_bvh": bvh_rel,
        "g1_csv": csv_url,
        "shape": shape,
        "expected_F": f_val,
        "per_actor_F": f_val,
        "n_frames": int(clip["nf" if "nf" in clip else "move_duration_frames"]),
        "content_name": str(clip.get("content_name") or ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bones-seed", required=True, type=Path)
    ap.add_argument("--final-pairs", type=Path,
                    default=Path(__file__).parent / "data" / "final_pairs.json")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "data" / "final_matrix_config.json")
    ap.add_argument("--csv-dir", type=Path,
                    default=Path(__file__).parent / "data" / "final_matrix")
    args = ap.parse_args()

    bs = args.bones_seed.resolve()
    csv_dir = args.csv_dir.resolve(); csv_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(bs / "Correlation_V2"))
    from simple_interpretability import compute_clip_features  # type: ignore

    pairs = json.loads(Path(args.final_pairs).read_text())
    print(f"[load] {args.final_pairs} — {len(pairs['rows'])} attributes")

    print(f"[load] metadata parquet")
    meta = pd.read_parquet(bs / "metadata" / "seed_metadata_v003.parquet")

    out_rows = []
    groups: dict[str, list[str]] = {}
    for attr_row in pairs["rows"]:
        attr = attr_row["attribute"]
        attr_short = attr_row["attribute_short"]
        attr_unit  = attr_row["attribute_unit"]
        groups[attr_short] = []
        for panel in attr_row["panels"]:
            pid = panel["panel_id"]
            print(f"\n[{pid}] {attr_short} · {panel['task']} · {panel['feature']}  "
                  f"({panel['high_actor']} vs {panel['low_actor']})")
            cn = panel.get("content_name")
            if cn is None:
                cn = find_best_common_content_name(meta, bs, panel["high_actor"],
                                                   panel["low_actor"], panel["task"],
                                                   panel["feature"], compute_clip_features)
                print(f"  auto-picked content_name: {cn!r}")
            hi = materialize_side(meta, bs, pid, "high", panel["high_actor"],
                                   panel["task"], cn,
                                   attr, attr_unit, csv_dir, panel["feature"],
                                   compute_clip_features)
            lo = materialize_side(meta, bs, pid, "low",  panel["low_actor"],
                                   panel["task"], cn,
                                   attr, attr_unit, csv_dir, panel["feature"],
                                   compute_clip_features)
            print(f"  high {hi['actor_uid']} {hi['label']} F={hi['expected_F']:.2f}  cn={hi['content_name']}")
            print(f"  low  {lo['actor_uid']} {lo['label']} F={lo['expected_F']:.2f}  cn={lo['content_name']}")
            row = {
                "row_id": pid,           # panel.html keys off this
                "attribute": attr,
                "attribute_short": attr_short,
                "attribute_unit": attr_unit,
                "task": panel["task"],
                "feature": panel["feature"],
                "feature_unit": feature_unit(panel["feature"]),
                "feature_recipe": build_feature_recipe(panel["feature"]),
                "r": float(panel["r"]),
                "p": None,
                "n_actors": int(panel["n_actors"]),
                "predicted_sign": 1 if panel["r"] > 0 else -1,
                "chosen": "manual",
                "content_name": cn,
                "truncate_tail_seconds": float(panel["truncate_tail_seconds"])
                    if panel.get("truncate_tail_seconds") is not None else None,
                "camera_zoom":          float(panel["camera_zoom"])          if panel.get("camera_zoom")          is not None else None,
                "camera_target_y":      float(panel["camera_target_y"])      if panel.get("camera_target_y")      is not None else None,
                "g1_camera_zoom":       float(panel["g1_camera_zoom"])       if panel.get("g1_camera_zoom")       is not None else None,
                "g1_camera_target_y":   float(panel["g1_camera_target_y"])   if panel.get("g1_camera_target_y")   is not None else None,
                "smpl_camera_zoom":     float(panel["smpl_camera_zoom"])     if panel.get("smpl_camera_zoom")     is not None else None,
                "smpl_camera_target_y": float(panel["smpl_camera_target_y"]) if panel.get("smpl_camera_target_y") is not None else None,
                # Per-side trim of the leading N seconds — used to skip a bad
                # SMPL pose at the start of the clip without re-fitting.
                "high_trim_head_seconds": float(panel["high_trim_head_seconds"]) if panel.get("high_trim_head_seconds") is not None else None,
                "low_trim_head_seconds":  float(panel["low_trim_head_seconds"])  if panel.get("low_trim_head_seconds")  is not None else None,
                "high": hi,
                "low":  lo,
            }
            out_rows.append(row)
            groups[attr_short].append(pid)

    smpl_dir = "./" + (csv_dir.parent / ("smpl_" + csv_dir.name))\
                     .relative_to(Path(__file__).parent).as_posix()
    config = {
        "version": 1,
        "schema": "feature_matrix_v1",
        "chrome": "minimal",      # final-matrix.html UX: hide header,
                                  # SMPL chips show only the attribute value,
                                  # plot label shows the attribute name
        "smpl_dir": smpl_dir,
        "rows": out_rows,
        "groups": groups,
    }
    Path(args.out).write_text(json.dumps(config, indent=2))
    print(f"\n[out] wrote {args.out}  ({len(out_rows)} panels)")
    print(f"[out] CSVs under {csv_dir} ({len(out_rows) * 2} files)")
    print(f"[next] SMPL: generate_smpl_matrix.py --config {args.out} --skip-existing")


if __name__ == "__main__":
    main()
