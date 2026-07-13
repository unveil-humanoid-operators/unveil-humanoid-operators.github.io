"""
compute_insights.py
===================
Reads top_features_per_task_attr.json and actor_task_features.csv and produces:
  1. Per-task strongest correlate per attribute (CSV)
  2. Correlation sign distribution per attribute (CSV)
  3. Top 10 positive and negative correlated features per task/attribute (JSON)
  4. Scatter plots (PDF) for each top-10 pos/neg feature saved as:
       feature-attr-plots/{task}/{attribute}/{pos|neg}/{feature}_r_{r}.pdf

Usage:
    python compute_insights.py \
        --input simple_interpretability_results/top_features_per_task_attr.json \
        --actor_features_csv simple_interpretability_results/actor_task_features.csv \
        --min_actors 50 \
        --output_dir ./insight_outputs
"""

import os
import json
import argparse
import csv
from collections import defaultdict

import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────

MOTION_FEATURES = [
    "root_translate_mean_vel", "root_translate_rom", "root_translate_peak_vel",
    "root_rotate_mean_vel", "root_rotate_rom", "root_rotate_peak_vel",
    "hip_mean_vel", "hip_rom", "hip_peak_vel",
    "knee_mean_vel", "knee_rom", "knee_peak_vel",
    "ankle_mean_vel", "ankle_rom", "ankle_peak_vel",
    "waist_mean_vel", "waist_rom", "waist_peak_vel",
    "shoulder_mean_vel", "shoulder_rom", "shoulder_peak_vel",
    "elbow_mean_vel", "elbow_rom", "elbow_peak_vel",
    "wrist_mean_vel", "wrist_rom", "wrist_peak_vel",
]

ATTRS = ["actor_age_yr", "actor_height_cm", "actor_weight_kg", "gender_numeric"]
ATTR_LABELS = {
    "actor_age_yr": "Age",
    "actor_height_cm": "Height",
    "actor_weight_kg": "Weight",
    "gender_numeric": "Gender",
}
ATTR_AXIS_LABELS = {
    "actor_age_yr": "Age (years)",
    "actor_height_cm": "Height (cm)",
    "actor_weight_kg": "Weight (kg)",
    "gender_numeric": "Gender (0=F, 1=M)",
}

# ── Joint channel definitions (for HTML time-series extraction) ───────────────
# Maps feature prefix → ordered list of channel groups to plot.
# Each group: {"key": str, "cols": [int,...], "labels": [str,...]}
# Cols are 0-indexed into the 35-column G1 CSV (Frame column already dropped).
FEATURE_CHANNELS = {
    "root_translate": [
        {"key": "root_translate", "cols": [0, 1, 2],
         "labels": ["Trans X", "Trans Y", "Trans Z"]},
    ],
    "root_rotate": [
        {"key": "root_rotate", "cols": [3, 4, 5],
         "labels": ["Root Roll", "Root Pitch", "Root Yaw"]},
    ],
    "hip": [
        {"key": "left_hip",  "cols": [6,  7,  8],
         "labels": ["L-Hip Pitch", "L-Hip Roll", "L-Hip Yaw"]},
        {"key": "right_hip", "cols": [12, 13, 14],
         "labels": ["R-Hip Pitch", "R-Hip Roll", "R-Hip Yaw"]},
    ],
    "knee": [
        {"key": "left_knee",  "cols": [9],  "labels": ["L-Knee"]},
        {"key": "right_knee", "cols": [15], "labels": ["R-Knee"]},
    ],
    "ankle": [
        {"key": "left_ankle",  "cols": [10, 11],
         "labels": ["L-Ankle Pitch", "L-Ankle Roll"]},
        {"key": "right_ankle", "cols": [16, 17],
         "labels": ["R-Ankle Pitch", "R-Ankle Roll"]},
    ],
    "waist": [
        {"key": "waist", "cols": [18, 19, 20],
         "labels": ["Waist Yaw", "Waist Roll", "Waist Pitch"]},
    ],
    "shoulder": [
        {"key": "left_shoulder",  "cols": [21, 22, 23],
         "labels": ["L-Shoulder Pitch", "L-Shoulder Roll", "L-Shoulder Yaw"]},
        {"key": "right_shoulder", "cols": [28, 29, 30],
         "labels": ["R-Shoulder Pitch", "R-Shoulder Roll", "R-Shoulder Yaw"]},
    ],
    "elbow": [
        {"key": "left_elbow",  "cols": [24], "labels": ["L-Elbow"]},
        {"key": "right_elbow", "cols": [31], "labels": ["R-Elbow"]},
    ],
    "wrist": [
        {"key": "left_wrist",  "cols": [25, 26, 27],
         "labels": ["L-Wrist Yaw", "L-Wrist Roll", "L-Wrist Pitch"]},
        {"key": "right_wrist", "cols": [32, 33, 34],
         "labels": ["R-Wrist Yaw", "R-Wrist Roll", "R-Wrist Pitch"]},
    ],
}

# Flat lookup: group_key → {cols, labels}  (used when loading clip data)
_ALL_CHANNEL_GROUPS = {
    g["key"]: g
    for groups in FEATURE_CHANNELS.values()
    for g in groups
}

JOINT_YLABEL = {
    "root_translate": "Translation",
    "root_rotate":    "Rotation Angle (rad)",
    "hip":            "Hip Angle (rad)",
    "knee":           "Knee Angle (rad)",
    "ankle":          "Ankle Angle (rad)",
    "waist":          "Waist Angle (rad)",
    "shoulder":       "Shoulder Angle (rad)",
    "elbow":          "Elbow Angle (rad)",
    "wrist":          "Wrist Angle (rad)",
}

# feature_channels dict that goes into the HTML (group keys only, no cols)
_FEATURE_CHANNELS_JS = {
    prefix: [g["key"] for g in groups]
    for prefix, groups in FEATURE_CHANNELS.items()
}

# channel label lookup that goes into the HTML
_CHANNEL_LABELS_JS = {
    g["key"]: g["labels"]
    for groups in FEATURE_CHANNELS.values()
    for g in groups
}


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input", type=str,
        default=r".\simple_interpretability_results\top_features_per_task_attr.json",
        help="Path to top_features_per_task_attr.json",
    )
    p.add_argument(
        "--actor_features_csv", type=str,
        default=r".\simple_interpretability_results\actor_task_features.csv",
        help="Path to actor_task_features.csv (needed for top-10 pos/neg and plots)",
    )
    p.add_argument(
        "--task_summary_csv", type=str,
        default=r".\simple_interpretability_results\task_attr_top3_summary.csv",
        help="Optional: path to task_attr_top3_summary.csv for actor counts. "
             "If not provided, all tasks are included.",
    )
    p.add_argument("--min_actors", type=int, default=50,
                   help="Minimum actors per task to include (requires --task_summary_csv)")
    p.add_argument("--output_dir", type=str, default="./insight_outputs")
    p.add_argument("--top_k", type=int, default=10,
                   help="How many top positive and negative features to report per task/attr")
    # ── HTML explorer ──────────────────────────────────────────────────────────
    p.add_argument("--manifests_dir", type=str, default=None,
                   help="Path to splits dir with train/val/test_manifest.csv (needed for HTML time-series)")
    p.add_argument("--data_root", type=str, default="..",
                   help="Data root for resolving G1 clip paths (needed for HTML time-series)")
    p.add_argument("--html_output", type=str, default=None,
                   help="Output path for interactive HTML explorer "
                        "(auto-set to {output_dir}/motion_explorer.html when --manifests_dir is given)")
    p.add_argument("--html_actors_per_task", type=int, default=25,
                   help="Max actors pre-sampled per task in the HTML explorer (default: 25)")
    p.add_argument("--html_downsample", type=int, default=10,
                   help="Downsample factor for time-series (10 → 12 fps from 120 fps, default: 10)")
    return p.parse_args()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_actor_counts(csv_path):
    """Load num_actors per task from the summary CSV."""
    counts = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row["content_type_of_movement"]
            counts[task] = int(row["num_actors"])
    return counts


def load_actor_task_features(csv_path):
    """Load actor_task_features.csv and group rows by task."""
    task_data = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_data[row["content_type_of_movement"]].append(row)
    return task_data


# ── Correlation helpers ────────────────────────────────────────────────────────

def compute_all_correlations(rows, attr, features):
    """Compute Pearson r and p-value for every feature vs attr over the given rows."""
    attr_vals_raw = []
    for row in rows:
        try:
            attr_vals_raw.append(float(row[attr]))
        except (ValueError, KeyError):
            attr_vals_raw.append(None)

    results = []
    for feat in features:
        pairs = []
        for a_val, row in zip(attr_vals_raw, rows):
            if a_val is None:
                continue
            try:
                f_val = float(row[feat])
                pairs.append((a_val, f_val))
            except (ValueError, KeyError):
                pass

        if len(pairs) < 3:
            continue

        a_arr = [p[0] for p in pairs]
        f_arr = [p[1] for p in pairs]
        r, p = stats.pearsonr(a_arr, f_arr)
        results.append({"feature": feat, "r": float(r), "p": float(p)})

    return results


def get_top_k_pos_neg(corr_results, k=10):
    """Return (top_k_positive, top_k_negative) sorted by descending |r|."""
    positives = sorted([e for e in corr_results if e["r"] > 0], key=lambda x: -x["r"])
    negatives = sorted([e for e in corr_results if e["r"] < 0], key=lambda x: x["r"])  # most negative first
    return positives[:k], negatives[:k]


# ── Plotting ───────────────────────────────────────────────────────────────────

def sanitize_dirname(name):
    """Make a string safe for use as a directory/file name component."""
    return name.replace(", ", "_").replace(" ", "_").replace("/", "_")


def plot_feature_vs_attr(attr_vals, feat_vals, attr, feat, r, p, out_path):
    """
    Scatter plot: x = attribute (e.g. Age), y = feature (e.g. waist_rom).
    A regression line shows the correlation direction.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(attr_vals, feat_vals, alpha=0.6, s=20, color="steelblue", edgecolors="none")

    # Regression line
    m, b = np.polyfit(attr_vals, feat_vals, 1)
    x_range = np.linspace(min(attr_vals), max(attr_vals), 200)
    ax.plot(x_range, m * x_range + b, color="crimson", linewidth=1.5)

    ax.set_xlabel(ATTR_AXIS_LABELS[attr], fontsize=10)
    ax.set_ylabel(feat, fontsize=10)
    direction = "positive" if r > 0 else "negative"
    ax.set_title(
        f"{feat}  vs  {ATTR_LABELS[attr]}\n"
        f"r = {r:+.3f}  ({direction})   p = {p:.2e}",
        fontsize=10,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# ── New output: top-10 pos/neg JSON + plots ────────────────────────────────────

def compute_and_save_top10(task_rows_map, actor_counts, args):
    """
    For each task with actor_count >= min_actors:
      - Compute Pearson r for all 27 features vs each attribute
      - Find top_k positive and top_k negative correlated features
      - Save combined result as JSON
      - Save one scatter-plot PDF per feature/attr/sign combination
    """
    k = args.top_k
    plot_root = os.path.join(args.output_dir, "feature-attr-plots")
    top10_json = {}

    tasks_sorted = sorted(task_rows_map.keys())
    for task in tasks_sorted:
        rows = task_rows_map[task]
        n_actors = actor_counts.get(task, 0)
        if actor_counts and n_actors < args.min_actors:
            continue

        task_safe = sanitize_dirname(task)
        top10_json[task] = {"num_actors": n_actors}
        print(f"  Processing task: '{task}'  ({n_actors} actors, {len(rows)} rows)")

        for attr in ATTRS:
            label = ATTR_LABELS[attr]
            corr_results = compute_all_correlations(rows, attr, MOTION_FEATURES)
            top_pos, top_neg = get_top_k_pos_neg(corr_results, k=k)

            top10_json[task][label] = {
                "positive": [
                    {"feature": e["feature"], "r": round(e["r"], 4), "p": e["p"]}
                    for e in top_pos
                ],
                "negative": [
                    {"feature": e["feature"], "r": round(e["r"], 4), "p": e["p"]}
                    for e in top_neg
                ],
            }

            # Build lookup for fast paired-data retrieval
            attr_vals_all = []
            for row in rows:
                try:
                    attr_vals_all.append(float(row[attr]))
                except (ValueError, KeyError):
                    attr_vals_all.append(None)

            for sign_key, entries in [("pos", top_pos), ("neg", top_neg)]:
                for entry in entries:
                    feat = entry["feature"]
                    r_val = entry["r"]
                    p_val = entry["p"]

                    pairs = [
                        (a, float(row[feat]))
                        for a, row in zip(attr_vals_all, rows)
                        if a is not None and row.get(feat, "") != ""
                    ]
                    if len(pairs) < 3:
                        continue

                    a_arr = [p[0] for p in pairs]
                    f_arr = [p[1] for p in pairs]

                    # e.g. waist_rom_r_-0.389.pdf
                    r_str = f"{r_val:.3f}"
                    filename = f"{feat}_r_{r_str}.pdf"
                    out_path = os.path.join(plot_root, task_safe, label, sign_key, filename)
                    plot_feature_vs_attr(a_arr, f_arr, attr, feat, r_val, p_val, out_path)

    # Save JSON
    json_path = os.path.join(args.output_dir, f"top{k}_pos_neg_features_per_task_attr.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(top10_json, f, indent=2)
    print(f"\nSaved: {json_path}")
    print(f"Plots saved under: {plot_root}/")

    return top10_json


# ── HTML Explorer helpers ──────────────────────────────────────────────────────

def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _load_clip_for_html(data_root, rel_path, max_frames=6000):
    """Load a G1 CSV clip; returns ndarray (T, 35) or None on failure."""
    if not _PANDAS_AVAILABLE:
        return None
    try:
        full_path = os.path.join(data_root, rel_path.replace("\\", "/").strip())
        df = pd.read_csv(full_path)
        if "Frame" in df.columns:
            df = df.drop(columns=["Frame"])
        x = df.to_numpy(dtype=np.float32)
        if x.ndim != 2 or x.shape[1] < 35:
            return None
        return np.nan_to_num(x[:max_frames, :35], nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return None


def _load_manifests_for_html(manifests_dir):
    """Load train/val/test manifests; return combined DataFrame or None."""
    if not _PANDAS_AVAILABLE:
        print("  [skip] pandas not installed — cannot generate HTML explorer.")
        return None
    parts = []
    for split in ("train", "val", "test"):
        fp = os.path.join(manifests_dir, f"{split}_manifest.csv")
        if os.path.exists(fp):
            parts.append(pd.read_csv(fp))
    if not parts:
        print(f"  [warn] No manifest CSVs found in {manifests_dir}")
        return None
    return pd.concat(parts, ignore_index=True)


def _build_actor_clip_map(manifest_df):
    """Return {task: {actor_uid: [rel_path, ...]}}."""
    mapping = defaultdict(lambda: defaultdict(list))
    for _, row in manifest_df.iterrows():
        task = row.get("content_type_of_movement")
        actor = row.get("actor_uid")
        path = row.get("move_g1_mujoco_path")
        if pd.isna(task) or pd.isna(actor) or pd.isna(path):
            continue
        mapping[str(task)][str(actor)].append(str(path))
    return mapping


def _build_html_actor_data(top10_json, actor_clip_map, task_rows_map, data_root, args):
    """
    For each qualifying task sample up to html_actors_per_task actors.
    For each actor pick one random demo clip and store raw per-channel time-series
    (downsampled) so the browser can plot any feature's individual channels.
    """
    n_sample = args.html_actors_per_task
    ds = args.html_downsample
    effective_fps = round(120.0 / ds, 3)

    # ── global attribute ranges ───────────────────────────────────────────────
    all_vals = {a: [] for a in ATTRS}
    for rows in task_rows_map.values():
        for row in rows:
            for a in ATTRS:
                v = _safe_float(row.get(a))
                if v is not None:
                    all_vals[a].append(v)

    attr_info = {}
    for attr, label in ATTR_LABELS.items():
        if attr == "gender_numeric":
            attr_info[label] = {
                "key": attr, "type": "binary",
                "values": [0, 1], "labels": ["F", "M"],
            }
        else:
            vals = all_vals[attr]
            attr_info[label] = {
                "key": attr, "type": "continuous",
                "min": float(min(vals)) if vals else 0.0,
                "max": float(max(vals)) if vals else 100.0,
            }

    # ── per-task actor data ───────────────────────────────────────────────────
    actors_data = {}
    has_ts = False

    for task in sorted(top10_json.keys()):
        rows = task_rows_map.get(task, [])
        if not rows:
            continue

        # Build per-actor demographic lookup (one row per actor from aggregated CSV)
        actor_demog = {}
        for row in rows:
            uid = row.get("actor_uid", "")
            if uid and uid not in actor_demog:
                actor_demog[uid] = row

        uids = list(actor_demog.keys())
        random.shuffle(uids)
        sampled_uids = uids[:n_sample]

        task_clip_map = actor_clip_map.get(task, {})
        task_actor_list = []
        print(f"    [{task}]  loading time-series for {len(sampled_uids)} actors...")

        for uid in sampled_uids:
            row = actor_demog[uid]
            entry = {
                "actor_uid":       uid,
                "actor_age_yr":    _safe_float(row.get("actor_age_yr")),
                "actor_height_cm": _safe_float(row.get("actor_height_cm")),
                "actor_weight_kg": _safe_float(row.get("actor_weight_kg")),
                "gender_numeric":  _safe_float(row.get("gender_numeric")),
                "actor_gender":    row.get("actor_gender", ""),
                "channels":        {},   # group_key → {labels, data}
            }

            clip_paths = task_clip_map.get(uid, [])
            if clip_paths and data_root:
                # Pick a random demo clip for this actor
                chosen_path = random.choice(clip_paths)
                x = _load_clip_for_html(data_root, chosen_path)
                if x is not None:
                    has_ts = True
                    for gkey, ginfo in _ALL_CHANNEL_GROUPS.items():
                        cols = ginfo["cols"]
                        labels = ginfo["labels"]
                        # Extract columns, downsample, round to 4 dp
                        raw = x[::ds, cols]  # shape (T//ds, n_cols)
                        entry["channels"][gkey] = {
                            "labels": labels,
                            "data": [
                                [round(float(v), 4) for v in raw[:, ci]]
                                for ci in range(len(cols))
                            ],
                        }

            task_actor_list.append(entry)

        actors_data[task] = task_actor_list

    return {
        "tasks":           sorted(top10_json.keys()),
        "attrs":           attr_info,
        "top10":           top10_json,
        "actors":          actors_data,
        "fps":             effective_fps,
        "joint_ylabels":   JOINT_YLABEL,
        "feature_channels": _FEATURE_CHANNELS_JS,
        "channel_labels":  _CHANNEL_LABELS_JS,
        "has_timeseries":  has_ts,
    }


# ── HTML template ──────────────────────────────────────────────────────────────

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BONES-SEED Motion Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
     background:#f0f2f5;color:#24292e;font-size:14px}
header{background:#1a1a2e;color:#fff;padding:14px 28px}
header h1{font-size:1.25rem;font-weight:600;letter-spacing:.3px}
header p{font-size:.82rem;color:#8b92a5;margin-top:2px}
#app{max-width:1340px;margin:20px auto;padding:0 16px;display:flex;flex-direction:column;gap:14px}
.panel{background:#fff;border-radius:8px;padding:18px 22px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.step-hdr{font-size:.95rem;font-weight:600;margin-bottom:12px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.step-num{background:#0366d6;color:#fff;border-radius:50%;width:22px;height:22px;
          display:inline-flex;align-items:center;justify-content:center;font-size:.78rem;flex-shrink:0}
select#task-select{padding:7px 10px;border:1px solid #d1d5da;border-radius:6px;
                   font-size:.95rem;min-width:320px;max-width:100%}
.chip{display:inline-block;background:#f1f8ff;border:1px solid #c8e1ff;color:#0366d6;
      border-radius:12px;padding:1px 9px;font-size:.78rem;margin-left:8px}
.tab-row{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap}
.tab-btn{padding:5px 16px;border:1px solid #d1d5da;border-radius:20px;background:#fff;
         cursor:pointer;font-size:.88rem;transition:all .12s}
.tab-btn:hover{background:#f6f8fa}
.tab-btn.active{background:#0366d6;color:#fff;border-color:#0366d6}
.range-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.range-row input[type=number]{width:88px;padding:5px 7px;border:1px solid #d1d5da;
                               border-radius:4px;font-size:.88rem}
.range-row label{display:flex;align-items:center;gap:6px;font-size:.88rem;color:#444}
.unit{color:#888;font-size:.8rem}
.feat-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:700px){.feat-grid{grid-template-columns:1fr}}
.feat-label{font-weight:600;padding:4px 0;margin-bottom:6px;font-size:.88rem}
.feat-label.pos{color:#2ea44f}.feat-label.neg{color:#d73a49}
table.feat-tbl{width:100%;border-collapse:collapse;font-size:.83rem}
table.feat-tbl thead tr{background:#f6f8fa}
table.feat-tbl th{text-align:left;padding:5px 8px;border-bottom:2px solid #e1e4e8;
                  font-weight:600;font-size:.78rem;color:#6a737d}
tr.feat-row{cursor:pointer;transition:background .1s}
tr.feat-row:hover{background:#f6f8fa}
tr.feat-row.selected{background:#fff8e1!important;outline:2px solid #f6a800}
tr.feat-row td{padding:5px 8px;border-bottom:1px solid #f0f0f0;vertical-align:middle}
td.fn{font-family:monospace;font-size:.82rem}
td.fr{font-weight:600;white-space:nowrap}
tr.feat-row.pos td.fr{color:#2ea44f}tr.feat-row.neg td.fr{color:#d73a49}
td.fb{width:80px}
.bar-bg{background:#f0f0f0;border-radius:3px;height:7px;width:100%}
.bar-fill{height:7px;border-radius:3px;transition:width .25s}
.bar-fill.pos{background:#2ea44f}.bar-fill.neg{background:#d73a49}
button.action-btn{padding:5px 13px;border:1px solid #d1d5da;border-radius:6px;
                  background:#fff;cursor:pointer;font-size:.88rem;margin-left:8px}
button.action-btn:hover:not(:disabled){background:#f6f8fa}
button.action-btn:disabled{opacity:.45;cursor:not-allowed}
.n-actors-wrap{display:inline-flex;align-items:center;gap:5px;margin-left:12px;font-size:.88rem}
.n-actors-wrap input{width:52px;padding:3px 6px;border:1px solid #d1d5da;border-radius:4px;font-size:.88rem}
.badge{background:#e1e4e8;border-radius:12px;padding:1px 9px;
       font-size:.78rem;color:#6a737d;margin-left:8px}
.hint{font-size:.82rem;color:#6a737d;margin:5px 0}
.warn{font-size:.85rem;color:#b08800;background:#fffbe6;border:1px solid #ffe58f;
      border-radius:4px;padding:8px 12px;margin:6px 0}
.pos-text{color:#2ea44f;font-weight:600}.neg-text{color:#d73a49;font-weight:600}
sup{font-size:.7em}
#no-feat-msg{color:#888;font-size:.88rem;padding:8px 0}
#plot{min-height:380px}
#actor-panel{display:none;margin-top:14px;padding-top:12px;border-top:1px solid #e1e4e8}
#actor-panel-title{font-size:.78rem;font-weight:600;color:#6a737d;
                   letter-spacing:.5px;text-transform:uppercase;margin-bottom:8px}
.actor-list-grid{display:flex;flex-wrap:wrap;gap:8px}
.actor-card{display:flex;align-items:center;gap:7px;padding:5px 9px;
            border-radius:6px;border:1px solid #e1e4e8;background:#fff;font-size:.82rem}
.actor-card.off .actor-attr{opacity:.35}
.actor-card.off .actor-uid{opacity:.35}
.actor-dot{width:12px;height:12px;border-radius:50%;flex-shrink:0}
.actor-attr{font-weight:600}
.actor-uid{color:#999;font-size:.74rem;margin-left:1px}
.toggle-btn{padding:2px 9px;border-radius:4px;font-size:.75rem;cursor:pointer;
            border:1px solid #d1d5da;background:#fff;color:#444;margin-left:4px;
            transition:all .12s}
.toggle-btn:hover{background:#f0f0f0}
.toggle-btn.is-off{background:#f6f8fa;color:#aaa;border-color:#e1e4e8}
</style>
</head>
<body>
<header>
  <h1>BONES-SEED Motion Explorer</h1>
  <p>Feature correlation &amp; raw joint time-series viewer</p>
</header>

<div id="app">

  <!-- Step 1 -->
  <div class="panel">
    <div class="step-hdr"><span class="step-num">1</span> Select Task</div>
    <select id="task-select" onchange="onTaskChange()">
      <option value="">— choose a task —</option>
    </select>
    <span id="task-chip" class="chip" style="display:none"></span>
  </div>

  <!-- Step 2 -->
  <div class="panel">
    <div class="step-hdr"><span class="step-num">2</span> Select Attribute &amp; Range</div>
    <div id="attr-tabs" class="tab-row"></div>
    <div id="range-ctl" class="range-row"></div>
  </div>

  <!-- Step 3 -->
  <div class="panel">
    <div class="step-hdr"><span class="step-num">3</span> Click a Feature to Explore</div>
    <div id="no-feat-msg">Select a task and attribute above.</div>
    <div id="feat-grid" class="feat-grid" style="display:none">
      <div>
        <div class="feat-label pos">&#9650; Top Positive Correlations</div>
        <table class="feat-tbl"><thead><tr>
          <th>Feature</th><th>r</th><th>Strength</th>
        </tr></thead><tbody id="pos-tbody"></tbody></table>
      </div>
      <div>
        <div class="feat-label neg">&#9660; Top Negative Correlations</div>
        <table class="feat-tbl"><thead><tr>
          <th>Feature</th><th>r</th><th>Strength</th>
        </tr></thead><tbody id="neg-tbody"></tbody></table>
      </div>
    </div>
  </div>

  <!-- Step 4 -->
  <div class="panel">
    <div class="step-hdr">
      <span class="step-num">4</span> Joint Time-Series
      <button id="reroll" class="action-btn" onclick="rollActors()" disabled>&#127922; Re-roll</button>
      <span class="n-actors-wrap">
        Actors: <input type="number" id="n-actors" value="10" min="1" max="50">
      </span>
      <span id="actor-badge" class="badge"></span>
    </div>
    <div id="feat-desc" class="hint"></div>
    <div id="plot-msg"></div>
    <div id="plot"></div>
    <div id="actor-panel">
      <div id="actor-panel-title">Actor Legend</div>
      <div id="actor-list" class="actor-list-grid"></div>
    </div>
  </div>

</div>

<script>
const DATA = JSON.parse('/*__DATA__*/');

/* ── 10-colour palette (Plotly's default tab10) ─────────────────── */
const PALETTE = [
  "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
  "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"
];
/* Line-dash styles to distinguish channels within one actor/group */
const DASHES = ["solid","dash","dot","dashdot","longdash","longdashdot"];

const state = {
  task:"", attr:"Age",
  attrMin:-Infinity, attrMax:Infinity,
  genderFilter:new Set([0,1]),
  feature:null, sign:null
};

/* actor toggle state (populated by renderPlot) */
let enabledActors = new Set();   // UIDs currently shown
let traceActorUids = [];         // parallel to Plotly traces: uid for each trace

/* ── init ────────────────────────────────────────────────────────── */
function init(){
  const sel=document.getElementById("task-select");
  DATA.tasks.forEach(t=>{const o=document.createElement("option");o.value=o.textContent=t;sel.appendChild(o);});
  const tabRow=document.getElementById("attr-tabs");
  Object.keys(DATA.attrs).forEach(attr=>{
    const btn=document.createElement("button");
    btn.className="tab-btn"+(attr===state.attr?" active":"");
    btn.textContent=attr; btn.onclick=()=>selectAttr(attr);
    tabRow.appendChild(btn);
  });
  selectAttr("Age");
  if(!DATA.has_timeseries){
    document.getElementById("plot-msg").innerHTML=
      '<div class="warn">&#9888; Time-series data not loaded. Re-run with '+
      '<code>--manifests_dir ../artifacts/splits --data_root ..</code> to enable Step 4.</div>';
  }
}

/* ── task ────────────────────────────────────────────────────────── */
function onTaskChange(){
  state.task=document.getElementById("task-select").value;
  const chip=document.getElementById("task-chip");
  if(state.task&&DATA.actors[state.task]){
    chip.textContent=DATA.actors[state.task].length+" actors";chip.style.display="inline";
  } else {chip.style.display="none";}
  state.feature=null;state.sign=null;
  refreshFeatureTables();clearPlot();
}

/* ── attribute ───────────────────────────────────────────────────── */
function selectAttr(attr){
  state.attr=attr;
  document.querySelectorAll(".tab-btn").forEach(b=>b.classList.toggle("active",b.textContent===attr));
  buildRangeControl();refreshFeatureTables();
  if(state.feature)rollActors();
}

function buildRangeControl(){
  const info=DATA.attrs[state.attr];
  const ctl=document.getElementById("range-ctl");
  if(info.type==="continuous"){
    const lo=info.min,hi=info.max;
    state.attrMin=lo;state.attrMax=hi;
    const unit=info.key.includes("age")?"yr":info.key.includes("height")?"cm":info.key.includes("weight")?"kg":"";
    ctl.innerHTML='<label>From <input type="number" id="rmin" step="0.1" value="'+lo.toFixed(1)+
      '" min="'+lo.toFixed(1)+'" max="'+hi.toFixed(1)+'" onchange="onRangeChange()"> '+
      'to <input type="number" id="rmax" step="0.1" value="'+hi.toFixed(1)+
      '" min="'+lo.toFixed(1)+'" max="'+hi.toFixed(1)+'" onchange="onRangeChange()">'+
      ' <span class="unit">'+unit+'</span></label>';
  } else {
    state.genderFilter=new Set([0,1]);
    ctl.innerHTML='<label><input type="checkbox" checked onchange="onGenderChange(0,this.checked)"> Female</label>'+
      '<label style="margin-left:14px"><input type="checkbox" checked onchange="onGenderChange(1,this.checked)"> Male</label>';
  }
}

function onRangeChange(){
  const a=document.getElementById("rmin"),b=document.getElementById("rmax");
  if(!a||!b)return;
  state.attrMin=parseFloat(a.value);state.attrMax=parseFloat(b.value);
  if(state.feature)rollActors();
}
function onGenderChange(val,checked){
  if(checked)state.genderFilter.add(val);else state.genderFilter.delete(val);
  if(state.feature)rollActors();
}

/* ── feature tables ──────────────────────────────────────────────── */
function refreshFeatureTables(){
  const noMsg=document.getElementById("no-feat-msg");
  const grid=document.getElementById("feat-grid");
  if(!state.task){grid.style.display="none";noMsg.style.display="";return;}
  const block=((DATA.top10[state.task]||{})[state.attr])||{positive:[],negative:[]};
  grid.style.display="";noMsg.style.display="none";
  fillTable("pos-tbody",block.positive||[],"pos");
  fillTable("neg-tbody",block.negative||[],"neg");
}

function fillTable(tbodyId,entries,sign){
  const tbody=document.getElementById(tbodyId);
  if(!entries.length){
    tbody.innerHTML='<tr><td colspan="3" class="hint" style="padding:8px">No data</td></tr>';return;
  }
  const maxR=Math.max(...entries.map(e=>Math.abs(e.r)));
  tbody.innerHTML=entries.map(e=>{
    const pct=Math.round(100*Math.abs(e.r)/(maxR||1));
    const sig=e.p<0.001?"***":e.p<0.01?"**":e.p<0.05?"*":"";
    const rStr=(e.r>0?"+":"")+e.r.toFixed(3);
    return '<tr class="feat-row '+sign+(state.feature===e.feature?" selected":"")+'"'+
           ' onclick="selectFeature(\''+e.feature+'\',\''+sign+'\')">'+
           '<td class="fn">'+e.feature+'</td>'+
           '<td class="fr">'+rStr+'<sup>'+sig+'</sup></td>'+
           '<td class="fb"><div class="bar-bg"><div class="bar-fill '+sign+
           '" style="width:'+pct+'%"></div></div></td></tr>';
  }).join("");
}

/* ── selection & plot ────────────────────────────────────────────── */
function selectFeature(feat,sign){
  state.feature=feat;state.sign=sign;
  document.querySelectorAll(".feat-row").forEach(r=>
    r.classList.toggle("selected",r.querySelector(".fn")?.textContent===feat));
  document.getElementById("reroll").disabled=false;
  rollActors();
}

function getFilteredActors(){
  if(!state.task)return[];
  const actors=DATA.actors[state.task]||[];
  const info=DATA.attrs[state.attr];const key=info.key;
  return actors.filter(a=>{
    const v=a[key];
    if(v===null||v===undefined)return false;
    if(info.type==="continuous")return v>=state.attrMin&&v<=state.attrMax;
    return state.genderFilter.has(Math.round(v));
  });
}

function rollActors(){
  const all=(DATA.actors[state.task]||[]);
  const filtered=getFilteredActors();
  const n=Math.max(1,parseInt(document.getElementById("n-actors").value)||10);
  const selected=[...filtered].sort(()=>Math.random()-.5).slice(0,n);
  const badge=document.getElementById("actor-badge");
  if(filtered.length===0){
    badge.textContent="0 actors in range (pool: "+all.length+")";
    badge.style.color="#d73a49";
  } else if(selected.length<n){
    badge.textContent="Only "+filtered.length+" available in range (pool: "+all.length+") \u2014 showing all";
    badge.style.color="#d73a49";
  } else {
    badge.textContent="Showing "+selected.length+" of "+filtered.length+" in range (pool: "+all.length+")";
    badge.style.color="";
  }
  renderPlot(selected);
}

function renderPlot(actors){
  const plotDiv=document.getElementById("plot");
  const desc=document.getElementById("feat-desc");

  if(!DATA.has_timeseries)return;
  if(!state.feature){clearPlot();return;}
  if(!actors.length){
    desc.textContent="No actors in the selected attribute range.";
    Plotly.purge(plotDiv);return;
  }

  /* feature name → joint prefix → channel group keys */
  let joint=null;
  for(const j of Object.keys(DATA.feature_channels)){
    if(state.feature.startsWith(j+"_")){joint=j;break;}
  }
  if(!joint){desc.textContent="Cannot map '"+state.feature+"' to joint channels.";return;}

  const groupKeys=DATA.feature_channels[joint];
  const info=DATA.attrs[state.attr];
  const key=info.key;
  const unit=info.key.includes("age")?" yr":info.key.includes("height")?" cm":info.key.includes("weight")?" kg":"";

  /* build flat ordered list of (group_key, channel_index, label) */
  const allCh=[];
  groupKeys.forEach(gkey=>{
    (DATA.channel_labels[gkey]||[]).forEach((lbl,ci)=>allCh.push({gkey,ci,lbl}));
  });
  const nCh=allCh.length;
  if(!nCh){desc.textContent="No channels defined for this feature.";return;}

  /* ── layout: stacked subplots sharing one x-axis ────────────────
     yaxis  = top subplot    (index 0)
     yaxis2 = next subplot   (index 1)
     …all anchored to the same "x" axis so zoom/pan is linked.    */
  const gap=0.04;
  const rowH=(1-gap*(nCh-1))/nCh;

  const layout={
    title:{text:state.feature+"  \u00B7  "+state.task+"  \u00B7  by "+state.attr,font:{size:13}},
    height:Math.max(380,210*nCh),
    hovermode:"x unified",
    plot_bgcolor:"#fafafa",paper_bgcolor:"#fff",
    showlegend:true,
    legend:{groupclick:"toggleitem",tracegroupgap:4,font:{size:10},x:1.01,xanchor:"left"},
    margin:{t:48,r:260,b:52,l:90},
    xaxis:{title:"Time (seconds)",gridcolor:"#eee",domain:[0,1]},
  };

  allCh.forEach(({lbl},i)=>{
    const top=1-i*(rowH+gap);
    const bottom=top-rowH;
    const axKey=i===0?"yaxis":"yaxis"+(i+1);
    layout[axKey]={
      title:{text:lbl,font:{size:11}},
      domain:[Math.max(0,parseFloat(bottom.toFixed(4))),Math.min(1,parseFloat(top.toFixed(4)))],
      anchor:"x",
      gridcolor:"#eee",
    };
  });

  /* ── traces ──────────────────────────────────────────────────── */
  const traces=[];
  traceActorUids=[];   // reset global
  let anyData=false;

  actors.forEach((actor,aIdx)=>{
    const color=PALETTE[aIdx%PALETTE.length];
    const av=actor[key];
    const avLabel=info.type==="binary"
      ?(info.labels?.[Math.round(av)]??av)
      :(av!=null?av.toFixed(1):"?");
    const groupTitle=state.attr+": "+avLabel+unit+"  ("+actor.actor_uid.slice(0,8)+")";

    allCh.forEach(({gkey,ci,lbl},plotIdx)=>{
      const ch=actor.channels?.[gkey];
      if(!ch||!ch.data?.[ci])return;
      const chArr=ch.data[ci];
      if(!chArr.length)return;
      anyData=true;
      const xs=Array.from({length:chArr.length},(_,i)=>+(i/DATA.fps).toFixed(3));
      const yRef=plotIdx===0?"y":"y"+(plotIdx+1);
      traces.push({
        x:xs,y:chArr,
        mode:"lines",
        name:groupTitle,
        showlegend:false,         // legend handled by custom actor panel
        line:{color,width:1.5},
        xaxis:"x",yaxis:yRef,
        hovertemplate:"%{x:.2f}s: %{y:.4f}<br><b>"+lbl+"</b><br>"+groupTitle+"<extra></extra>",
      });
      traceActorUids.push(actor.actor_uid);
    });
  });

  if(!anyData){
    desc.innerHTML='<span style="color:#c00">No channel data found. Check --data_root and --manifests_dir.</span>';
    Plotly.purge(plotDiv);return;
  }

  // Use the range the user set in panel 2, not the sampled actors' min/max
  let rangeStr;
  if(info.type==="binary"){
    const gLabels=[...state.genderFilter].map(v=>info.labels?.[v]??v).join(", ");
    rangeStr=state.attr+": "+gLabels;
  } else {
    rangeStr=state.attr+": "+state.attrMin.toFixed(1)+unit+" &ndash; "+state.attrMax.toFixed(1)+unit;
  }
  const signTxt=state.sign==="pos"
    ?'<span class="pos-text">&#8593; positive with '+state.attr+'</span>'
    :'<span class="neg-text">&#8595; negative with '+state.attr+'</span>';
  desc.innerHTML="Showing <strong>"+state.feature+"</strong> &nbsp;("+signTxt+")"+
    " &nbsp;&mdash;&nbsp; <code>"+allCh.map(c=>c.lbl).join(" / ")+"</code>"+
    " &nbsp;| "+rangeStr+
    " &nbsp;| "+actors.length+" actors shown";

  layout.showlegend=false;
  Plotly.react(plotDiv,traces,layout,{responsive:true});

  renderActorPanel(actors);
}

/* ── actor panel ─────────────────────────────────────────────────── */
function renderActorPanel(actors){
  const panel=document.getElementById("actor-panel");
  const list=document.getElementById("actor-list");
  const info=DATA.attrs[state.attr];
  const key=info.key;
  const unit=info.key.includes("age")?" yr"
            :info.key.includes("height")?" cm"
            :info.key.includes("weight")?" kg":"";

  enabledActors=new Set(actors.map(a=>a.actor_uid));

  list.innerHTML=actors.map((actor,i)=>{
    const color=PALETTE[i%PALETTE.length];
    const av=actor[key];
    const avLabel=info.type==="binary"
      ?(info.labels?.[Math.round(av)]??av)
      :(av!=null?av.toFixed(1):"?");
    const uid=actor.actor_uid;
    // Wrap uid so toggleActor can be called safely (no special chars issue)
    const esc=uid.replace(/'/g,"\\'");
    return '<div class="actor-card" id="acard-'+uid+'">'
      +'<span class="actor-dot" style="background:'+color+'"></span>'
      +'<span class="actor-attr">'+state.attr+': '+avLabel+unit+'</span>'
      +'<span class="actor-uid">'+uid.slice(0,10)+'</span>'
      +'<button class="toggle-btn" id="tbtn-'+uid+'" onclick="toggleActor(\''+esc+'\')">Hide</button>'
      +'</div>';
  }).join("");

  panel.style.display="block";
}

function toggleActor(uid){
  const nowEnabled=!enabledActors.has(uid);   // flip
  if(nowEnabled) enabledActors.add(uid);
  else enabledActors.delete(uid);

  const card=document.getElementById("acard-"+uid);
  const btn =document.getElementById("tbtn-"+uid);
  if(card) card.classList.toggle("off",!nowEnabled);
  if(btn){
    btn.textContent=nowEnabled?"Hide":"Show";
    btn.classList.toggle("is-off",!nowEnabled);
  }
  updateTraceVisibility();
}

function updateTraceVisibility(){
  const plotDiv=document.getElementById("plot");
  if(!traceActorUids.length)return;
  const visible=traceActorUids.map(uid=>enabledActors.has(uid));
  Plotly.restyle(plotDiv,{visible:visible});
}

function clearPlot(){
  Plotly.purge(document.getElementById("plot"));
  document.getElementById("feat-desc").textContent="";
  document.getElementById("actor-badge").textContent="";
  document.getElementById("reroll").disabled=true;
  document.getElementById("actor-panel").style.display="none";
  document.getElementById("actor-list").innerHTML="";
  traceActorUids=[];enabledActors=new Set();
}

init();
</script>
</body>
</html>"""


def generate_html_explorer(html_data, out_path):
    """Write the self-contained interactive HTML explorer to out_path."""
    data_json = json.dumps(html_data, separators=(',', ':'))
    # Escape </script> so the inline JSON doesn't break the HTML parser
    data_json = data_json.replace("</", "<\\/")
    # Escape single quotes in JSON for the JS JSON.parse('...') call
    data_json = data_json.replace("'", "\\'")
    html = _HTML_TEMPLATE.replace("'/*__DATA__*/'", "'" + data_json + "'")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    # Load actor counts if available
    actor_counts = {}
    if args.task_summary_csv and os.path.isfile(args.task_summary_csv):
        actor_counts = load_actor_counts(args.task_summary_csv)
        print(f"Loaded actor counts for {len(actor_counts)} tasks from {args.task_summary_csv}")
    else:
        args.min_actors = 0

    # ── 1. Per-task strongest correlate per attribute ──────────────────────────

    strongest_rows = []
    strongest_json = {}

    for task, task_data in sorted(data.items()):
        n_actors = actor_counts.get(task, 0)
        if actor_counts and n_actors < args.min_actors:
            continue

        task_entry = {"num_actors": n_actors}

        for attr in ATTRS:
            label = ATTR_LABELS[attr]
            if attr not in task_data or not task_data[attr]:
                continue

            top = task_data[attr][0]
            feat = top["feature"]
            r = top["r"]
            p = top["p"]

            task_entry[label] = {"feature": feat, "r": round(r, 4), "p": p}
            strongest_rows.append({
                "task": task,
                "num_actors": n_actors,
                "attribute": label,
                "feature": feat,
                "r": round(r, 4),
                "p": p,
                "significant": p < 0.05,
            })

        strongest_json[task] = task_entry

    csv_path = os.path.join(args.output_dir, "per_task_strongest_feature.csv")
    strongest_rows.sort(key=lambda r: (-r["num_actors"], r["task"]))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "task", "num_actors", "attribute", "feature", "r", "p", "significant"
        ])
        writer.writeheader()
        writer.writerows(strongest_rows)
    print(f"Saved: {csv_path}  ({len(strongest_rows)} rows)")

    print(f"\n{'='*80}")
    print(f"  PER-TASK STRONGEST FEATURE (tasks with {args.min_actors}+ actors)")
    print(f"{'='*80}")
    print(f"  {'Task':<30s} {'Actors':>6s}  {'Age':<28s} {'Height':<28s} {'Weight':<28s} {'Gender':<28s}")
    print(f"  {'-'*150}")
    for task in sorted(strongest_json.keys(),
                       key=lambda t: strongest_json[t].get("num_actors", 0), reverse=True):
        entry = strongest_json[task]
        n = entry.get("num_actors", 0)
        parts = [f"  {task:<30s} {n:>6d}"]
        for label in ["Age", "Height", "Weight", "Gender"]:
            if label in entry:
                d = entry[label]
                sig = "**" if d["p"] < 0.01 else "*" if d["p"] < 0.05 else ""
                parts.append(f"{d['feature']} ({d['r']:+.3f}){sig:<3s}")
            else:
                parts.append(f"{'—':<28s}")
        print("  ".join(parts))

    # ── 2. Correlation sign distribution ──────────────────────────────────────

    sign_rows = []
    sign_json = {}
    per_task_signs = defaultdict(lambda: defaultdict(lambda: {"pos": 0, "neg": 0}))

    for task, task_data in sorted(data.items()):
        n_actors = actor_counts.get(task, 0)
        if actor_counts and n_actors < args.min_actors:
            continue
        for attr in ATTRS:
            label = ATTR_LABELS[attr]
            if attr not in task_data:
                continue
            for feat_entry in task_data[attr]:
                r = feat_entry["r"]
                if r > 0:
                    per_task_signs[task][label]["pos"] += 1
                else:
                    per_task_signs[task][label]["neg"] += 1

    for label in ATTR_LABELS.values():
        total_pos = sum(per_task_signs[t][label]["pos"] for t in per_task_signs)
        total_neg = sum(per_task_signs[t][label]["neg"] for t in per_task_signs)
        total = total_pos + total_neg
        pct_neg = round(100 * total_neg / total, 1) if total > 0 else 0
        sign_json[label] = {"positive": total_pos, "negative": total_neg,
                            "total": total, "pct_negative": pct_neg}
        sign_rows.append({"attribute": label, "positive_count": total_pos,
                          "negative_count": total_neg, "total": total, "pct_negative": pct_neg})

    sign_csv = os.path.join(args.output_dir, "sign_distribution.csv")
    with open(sign_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "attribute", "positive_count", "negative_count", "total", "pct_negative"
        ])
        writer.writeheader()
        writer.writerows(sign_rows)
    print(f"\nSaved: {sign_csv}")

    print(f"\n{'='*80}")
    print(f"  CORRELATION SIGN DISTRIBUTION")
    print(f"{'='*80}")
    print(f"  {'Attribute':<10s} {'Positive':>10s} {'Negative':>10s} {'Total':>8s} {'% Negative':>12s}")
    print(f"  {'-'*52}")
    for row in sign_rows:
        bar_len = int(40 * row["pct_negative"] / 100)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  {row['attribute']:<10s} {row['positive_count']:>10d} {row['negative_count']:>10d} "
              f"{row['total']:>8d} {row['pct_negative']:>10.1f}%  {bar}")

    sign_per_task_rows = []
    for task in sorted(per_task_signs.keys()):
        for label in ATTR_LABELS.values():
            s = per_task_signs[task][label]
            n = actor_counts.get(task, 0)
            total = s["pos"] + s["neg"]
            if total == 0:
                continue
            sign_per_task_rows.append({
                "task": task, "num_actors": n, "attribute": label,
                "positive": s["pos"], "negative": s["neg"], "total": total,
                "pct_negative": round(100 * s["neg"] / total, 1),
            })

    sign_per_task_csv = os.path.join(args.output_dir, "sign_distribution_per_task.csv")
    sign_per_task_rows.sort(key=lambda r: (-r["num_actors"], r["task"]))
    with open(sign_per_task_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "task", "num_actors", "attribute", "positive", "negative", "total", "pct_negative"
        ])
        writer.writeheader()
        writer.writerows(sign_per_task_rows)
    print(f"\nSaved: {sign_per_task_csv}  ({len(sign_per_task_rows)} rows)")

    # ── 3 & 4. Top-10 pos/neg JSON + scatter plots ─────────────────────────────

    top10_json = None
    task_rows_map = None

    if args.actor_features_csv and os.path.isfile(args.actor_features_csv):
        print(f"\n{'='*80}")
        print(f"  TOP {args.top_k} POSITIVE / NEGATIVE FEATURES + PLOTS")
        print(f"{'='*80}")
        task_rows_map = load_actor_task_features(args.actor_features_csv)
        top10_json = compute_and_save_top10(task_rows_map, actor_counts, args)
    else:
        print(f"\n[skip] actor_features_csv not found — skipping top-{args.top_k} and plots.")

    # ── 5. Interactive HTML explorer ───────────────────────────────────────────

    if top10_json and task_rows_map:
        print(f"\n{'='*80}")
        print(f"  GENERATING INTERACTIVE HTML EXPLORER")
        print(f"{'='*80}")

        if args.manifests_dir and os.path.isdir(args.manifests_dir):
            manifest_df = _load_manifests_for_html(args.manifests_dir)
            if manifest_df is not None:
                print(f"  Loaded {len(manifest_df):,} manifest rows.")
                actor_clip_map = _build_actor_clip_map(manifest_df)
                html_data = _build_html_actor_data(
                    top10_json, actor_clip_map, task_rows_map, args.data_root, args
                )
            else:
                print("  [warn] Could not load manifests — time-series will be unavailable in HTML.")
                html_data = _build_html_actor_data(
                    top10_json, {}, task_rows_map, None, args
                )
        else:
            if args.manifests_dir:
                print(f"  [warn] --manifests_dir not found: {args.manifests_dir}")
            else:
                print("  [info] --manifests_dir not set — generating HTML without time-series.")
                print("         Add --manifests_dir ../artifacts/splits --data_root .. to enable it.")
            html_data = _build_html_actor_data(
                top10_json, {}, task_rows_map, None, args
            )

        html_out = args.html_output or os.path.join(args.output_dir, "motion_explorer.html")
        generate_html_explorer(html_data, html_out)
        print(f"\n  HTML explorer saved: {html_out}")

    print(f"\nAll outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
