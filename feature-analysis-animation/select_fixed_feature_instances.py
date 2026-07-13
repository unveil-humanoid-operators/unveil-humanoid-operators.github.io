"""
select_fixed_feature_instances.py
---------------------------------
Independent instance picker for the bar plot — replaces the pair-based
selector.

For each (attribute, fixed_feature) in ATTR_FEATURE we:
  1. Score every task by |Pearson r(fixed_feature, attribute)| over actors
     with >= MIN_CLIPS_PER_ACTOR clips.
  2. Filter |r| >= MIN_ABS_R, take the top N_PER_ATTR tasks.
  3. For each kept task, determine sign(r) and split actors into a
     low-attribute decile and a high-attribute decile.
  4. Pick K_PER_SIDE_PER_TASK clips per side, choosing the per-clip F values
     that lie at the predicted extreme:
       sign(r) > 0:  low-side wants the SMALLEST F, high-side the LARGEST
       sign(r) < 0:  low-side wants the LARGEST F,  high-side the SMALLEST
     This naturally maximizes the bar-top gap.

The K instances per side per task are flattened into a single list per
attribute; each entry is a self-contained clip record (no pair linkage,
no row_id pairing).

Outputs:
  data/fixed_feature_instances.json         — flat instances list
  data/fixed_feature_instances_summary.csv  — scan-friendly view
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# Reuse helpers from the pair-based selector for parquet filtering,
# per-clip feature compute, and the canonical attribute→feature map.
sys.path.insert(0, str(Path(__file__).parent))
from select_pair_candidates import (  # type: ignore
    ATTR_SHORT,
    ATTR_UNIT,
    build_feature_recipe,
    compute_per_clip_feature,
    slugify,
)
from select_fixed_feature_pairs import (  # type: ignore
    ATTR_FEATURE,
    score_tasks_for_feature,
)

# Top-3 features per attribute (current default first, then alternates ranked
# by sum of top-5 |r| from discover_top_features). Used by the multi-feature
# selector mode.
ATTR_FEATURE_OPTIONS = {
    "actor_age_yr":    ["waist_rom",          "waist_mean_vel",          "ankle_rom"],
    "actor_weight_kg": ["ankle_peak_vel",     "ankle_rom",               "root_translate_peak_vel"],
    "actor_height_cm": ["root_translate_rom", "root_translate_peak_vel", "ankle_rom"],
    "gender_numeric":  ["ankle_rom",          "shoulder_rom",            "shoulder_mean_vel"],
}


def make_instance(side: str, row, attr: str, feat: str, task: str,
                  r_val: float, p_val: float, sign_r: int,
                  n_actors_scored: int, per_clip_F: float):
    """Build one self-contained instance record."""
    csv_rel = str(row.move_g1_mujoco_path).replace("\\", "/")
    bvh = getattr(row, "move_soma_uniform_path", None)
    bvh = str(bvh).replace("\\", "/") if isinstance(bvh, str) else None
    # Include feature slug so the same actor+task picked under two different
    # features still gets two distinct ids (their per-clip F picks may differ).
    instance_id = f"{ATTR_SHORT[attr]}_{slugify(feat)}_{slugify(task)}_{row.actor_uid}"

    # gender_numeric isn't a parquet column — derive it from actor_gender
    # the same way simple_interpretability.py does (alphabetical: F→0, M→1).
    if attr == "gender_numeric":
        attr_value = 0.0 if str(row.actor_gender) == "F" else 1.0
    else:
        attr_value = float(getattr(row, attr))

    return {
        "instance_id":    instance_id,
        "attribute":      attr,
        "attribute_short": ATTR_SHORT[attr],
        "attribute_unit": ATTR_UNIT[attr],
        "side":           side,                       # 'low' or 'high' (attribute side)
        "task":           task,
        "feature":        feat,
        "feature_recipe": build_feature_recipe(feat),
        "r":              float(r_val),
        "p":              float(p_val),
        "predicted_sign": sign_r,
        "n_actors_scored": int(n_actors_scored),
        "actor_uid":      str(row.actor_uid),
        "attr_value":     attr_value,
        "actor_age_yr":   float(getattr(row, "actor_age_yr")),
        "actor_height_cm": float(getattr(row, "actor_height_cm")),
        "actor_weight_kg": float(getattr(row, "actor_weight_kg")),
        "actor_gender":   str(getattr(row, "actor_gender")),
        "per_clip_F":     float(per_clip_F),
        "g1_csv":         csv_rel,
        "source_bvh":     bvh,
        "n_frames":       int(row.nf),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bones-seed", required=True, type=Path)
    ap.add_argument("--results", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    ap.add_argument("--n-per-attr", type=int, default=5,
                    help="Number of tasks kept per attribute (top-N by |r|).")
    ap.add_argument("--k-per-task-per-side", type=int, default=1,
                    help="Instances to pick per task per side. With 5 tasks "
                         "and K=1, each bar holds 5 dots; K=2 ⇒ 10 dots.")
    ap.add_argument("--min-abs-r", type=float, default=0.1)
    ap.add_argument("--min-task-actors", type=int, default=50)
    ap.add_argument("--min-clips-per-actor", type=int, default=2)
    ap.add_argument("--min-frames", type=int, default=200)
    ap.add_argument("--decile", type=float, default=0.20,
                    help="Decile threshold for splitting actors by attribute. "
                         "0.20 ⇒ bottom-20%% / top-20%% of actors.")
    ap.add_argument("--style", type=str, default="neutral")
    ap.add_argument("--multi", action="store_true",
                    help="Iterate over ATTR_FEATURE_OPTIONS (3 features per attribute) "
                         "instead of ATTR_FEATURE. Output goes to fixed_feature_instances_multi.json.")
    args = ap.parse_args()
    ATTR_FEAT_MAP = ATTR_FEATURE_OPTIONS if args.multi else {k: [v] for k, v in ATTR_FEATURE.items()}

    bs = args.bones_seed.resolve()
    results_dir = (args.results or bs / "Correlation_V2" / "simple_interpretability_results").resolve()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(bs / "Correlation_V2"))
    from simple_interpretability import compute_clip_features  # type: ignore

    print(f"[load] actor_task_features.csv")
    atf = pd.read_csv(results_dir / "actor_task_features.csv")
    print(f"  -> {len(atf):,} actor-task rows, {atf['actor_uid'].nunique()} actors")

    print(f"[load] metadata parquet")
    meta = pd.read_parquet(bs / "metadata" / "seed_metadata_v003.parquet")
    cand = meta[~meta["move_g1_mujoco_path"].astype(str).str.endswith("_M.csv")].copy()
    cand["abs_csv"] = cand["move_g1_mujoco_path"].apply(
        lambda r: bs / r if isinstance(r, str) else None
    )
    cand = cand[cand["abs_csv"].apply(lambda p: p is not None and Path(p).exists())]
    cand["nf"] = pd.to_numeric(cand["move_duration_frames"], errors="coerce")
    cand = cand[cand["nf"].fillna(0) >= args.min_frames]
    if args.style:
        cand = cand[cand["content_uniform_style"].astype(str) == args.style]
    print(f"  -> {len(cand):,} usable clips")

    cache_path = out / "per_clip_features.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    print(f"[cache] per_clip_features.json: {len(cache)} entries (pre-existing)")

    clip_counts = cand.groupby(["actor_uid", "content_type_of_movement"]).size()
    cand_grp = cand.groupby(["actor_uid", "content_type_of_movement"])

    all_instances = []
    summary_rows = []

    attr_feat_pairs = [(a, f) for a, feats in ATTR_FEAT_MAP.items() for f in feats]
    for attr, feat in attr_feat_pairs:
        attr_short = ATTR_SHORT[attr]
        print(f"\n=== {attr_short.upper()} (feature: {feat}) ===")

        scored = score_tasks_for_feature(
            atf, attr, feat,
            clip_counts=clip_counts,
            min_clips_per_actor=args.min_clips_per_actor,
            min_task_actors=args.min_task_actors,
        )
        scored = scored[scored["abs_r"] >= args.min_abs_r].reset_index(drop=True)
        scored = scored.head(args.n_per_attr)
        if scored.empty:
            print(f"  ! no tasks for {attr_short}"); continue
        print(f"  [{len(scored)} tasks]  top:")
        for _, s in scored.iterrows():
            print(f"    {s['task']:30s}  r={s['r']:+.3f}  n_actors={int(s['n_actors'])}")

        for _, s in scored.iterrows():
            task = s["task"]
            r_val, p_val = s["r"], s["p"]
            sign_r = 1 if r_val > 0 else -1

            # All eligible clips for this task: actors who have biometrics in atf
            # AND >= min_clips_per_actor clips of this task, AND a non-mirror CSV.
            actor_pool = atf[atf["content_type_of_movement"] == task].copy()
            actor_pool["n_clips"] = actor_pool["actor_uid"].map(
                lambda uid: int(clip_counts.get((uid, task), 0))
            )
            actor_pool = actor_pool[actor_pool["n_clips"] >= args.min_clips_per_actor]
            actor_pool = actor_pool.dropna(subset=[attr, feat])

            # Decile thresholds (binary attribute → quantile gives a sane split too).
            attr_lo = actor_pool[attr].quantile(args.decile)
            attr_hi = actor_pool[attr].quantile(1 - args.decile)

            low_actors  = set(actor_pool.loc[actor_pool[attr] <= attr_lo, "actor_uid"])
            high_actors = set(actor_pool.loc[actor_pool[attr] >= attr_hi, "actor_uid"])

            task_clips = cand[cand["content_type_of_movement"] == task].copy()
            task_clips = task_clips[task_clips["actor_uid"].isin(low_actors | high_actors)]

            # Per-clip F for every candidate clip.
            rows = []
            for clip_row in task_clips.itertuples():
                csv_rel = str(clip_row.move_g1_mujoco_path).replace("\\", "/")
                per_clip = compute_per_clip_feature(bs / csv_rel, compute_clip_features, cache, csv_rel)
                if per_clip is None or feat not in per_clip:
                    continue
                rows.append((clip_row, float(per_clip[feat])))
            if not rows:
                print(f"  [skip] {task} — no per-clip F computed"); continue

            low_rows  = [(r, f) for r, f in rows if r.actor_uid in low_actors]
            high_rows = [(r, f) for r, f in rows if r.actor_uid in high_actors]
            if not low_rows or not high_rows:
                print(f"  [skip] {task} — empty side  (lo={len(low_rows)}, hi={len(high_rows)})")
                continue

            # Sort each side by F in the predicted direction.
            if sign_r > 0:
                # Low attr → low F; pick SMALLEST F on low side, LARGEST on high.
                low_rows.sort (key=lambda x: x[1])             # ascending
                high_rows.sort(key=lambda x: -x[1])            # descending
            else:
                low_rows.sort (key=lambda x: -x[1])            # descending
                high_rows.sort(key=lambda x: x[1])             # ascending

            # Cap K but also: one clip per actor to keep instance diversity.
            def take_k_distinct(side_rows, k):
                seen_actors = set()
                kept = []
                for r, f in side_rows:
                    if r.actor_uid in seen_actors:
                        continue
                    seen_actors.add(r.actor_uid)
                    kept.append((r, f))
                    if len(kept) == k:
                        break
                return kept

            picked_low  = take_k_distinct(low_rows,  args.k_per_task_per_side)
            picked_high = take_k_distinct(high_rows, args.k_per_task_per_side)

            n_actors_scored = int(s["n_actors"])
            for r, f in picked_low:
                all_instances.append(make_instance(
                    "low", r, attr, feat, task, r_val, p_val, sign_r,
                    n_actors_scored, f,
                ))
            for r, f in picked_high:
                all_instances.append(make_instance(
                    "high", r, attr, feat, task, r_val, p_val, sign_r,
                    n_actors_scored, f,
                ))

            # Per-task gap report.
            lo_max_F = max(f for _, f in picked_low)
            hi_max_F = max(f for _, f in picked_high)
            print(f"  [keep] {task:30s} r={r_val:+.3f}  lo: "
                  + ", ".join(f"{f:.1f}" for _, f in picked_low)
                  + f"  | hi: " + ", ".join(f"{f:.1f}" for _, f in picked_high)
                  + f"  (max lo={lo_max_F:.1f}, max hi={hi_max_F:.1f})")

            for r, f in picked_low + picked_high:
                side = "low" if (r, f) in picked_low else "high"
                attr_val = (0.0 if str(r.actor_gender) == "F" else 1.0) \
                           if attr == "gender_numeric" else float(getattr(r, attr))
                summary_rows.append({
                    "instance_id": f"{attr_short}_{slugify(feat)}_{slugify(task)}_{r.actor_uid}",
                    "attribute":   attr_short,
                    "task":        task,
                    "feature":     feat,
                    "r":           r_val,
                    "side":        side,
                    "actor_uid":   r.actor_uid,
                    "attr_value":  attr_val,
                    "per_clip_F":  f,
                    "n_frames":    int(r.nf),
                    "g1_csv":      str(r.move_g1_mujoco_path).replace("\\", "/"),
                })

        # Cross-task panel-level gap diagnostic.
        panel_low_F = [
            i["per_clip_F"] for i in all_instances
            if i["attribute"] == attr and i["side"] == "low"
        ]
        panel_high_F = [
            i["per_clip_F"] for i in all_instances
            if i["attribute"] == attr and i["side"] == "high"
        ]
        if panel_low_F and panel_high_F:
            lo_max = max(panel_low_F); hi_max = max(panel_high_F)
            print(f"  -> {attr_short} bar tops: low.max={lo_max:.2f}  high.max={hi_max:.2f}  "
                  f"gap={abs(hi_max - lo_max):.2f}")

    # Save outputs.
    cache_path.write_text(json.dumps(cache, indent=1))
    print(f"\n[cache] wrote {cache_path} ({len(cache)} entries)")

    out_json = out / ("fixed_feature_instances_multi.json" if args.multi else "fixed_feature_instances.json")
    out_json.write_text(json.dumps({
        "attribute_feature": ATTR_FEATURE,
        "attribute_feature_options": ATTR_FEAT_MAP,
        "n_per_attr":         args.n_per_attr,
        "k_per_task_per_side": args.k_per_task_per_side,
        "params": {
            "min_abs_r":           args.min_abs_r,
            "min_task_actors":     args.min_task_actors,
            "min_clips_per_actor": args.min_clips_per_actor,
            "min_frames":          args.min_frames,
            "style":               args.style,
            "decile":              args.decile,
        },
        "instances": all_instances,
    }, indent=2))
    print(f"[out] wrote {out_json}  ({len(all_instances)} instances)")

    out_csv = out / "fixed_feature_instances_summary.csv"
    fields = ["instance_id", "attribute", "task", "feature", "r", "side",
              "actor_uid", "attr_value", "per_clip_F", "n_frames", "g1_csv"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for s in summary_rows: w.writerow(s)
    print(f"[out] wrote {out_csv}  ({len(summary_rows)} rows)")

    print("\n=== counts per attribute / side ===")
    from collections import Counter
    cnt = Counter((i["attribute_short"], i["side"]) for i in all_instances)
    for k, v in sorted(cnt.items()):
        print(f"  {k[0]:7s} {k[1]:5s}  {v}")


if __name__ == "__main__":
    main()
