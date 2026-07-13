"""
select_fixed_feature_pairs.py
-----------------------------
Pick N actor pairs per attribute where ONE fixed feature carries the gap.

For each (attribute, fixed feature) in `ATTR_FEATURE`:
  1. Score every task by |Pearson r(fixed_feature, attribute)| across actors
     (using bones-seed's per-(task, actor) feature means).
  2. Walk tasks |r|-desc and keep the first N where an extreme-pair selection
     produces a sign-correct, file-resolvable, n_frames>=min clip pair.

Output: data/fixed_feature_pairs.json + .csv

The fixed features (one per attribute) are chosen to give the bar-plot panel a
single y-axis label that's stable across the 10 task slots.

Usage:
  python select_fixed_feature_pairs.py --bones-seed C:/Users/sihat/Downloads/bones-seed
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# Reuse helpers from the existing candidate selector.
sys.path.insert(0, str(Path(__file__).parent))
from select_pair_candidates import (  # type: ignore
    ATTR_SHORT,
    ATTR_UNIT,
    build_feature_recipe,
    compute_per_clip_feature,
    pick_extreme_pair,
    pick_gap_pair,
    slugify,
)


# One fixed feature per attribute. These are the y-axis the bar plot will
# carry for the full 10-task row. All four are present in
# simple_interpretability.compute_clip_features (and in actor_task_features.csv).
ATTR_FEATURE = {
    "actor_age_yr":    "waist_rom",
    "actor_weight_kg": "ankle_peak_vel",
    "actor_height_cm": "root_translate_rom",
    "gender_numeric":  "ankle_rom",
}


def score_tasks_for_feature(atf: pd.DataFrame, attr: str, feat: str,
                            clip_counts: pd.Series, min_clips_per_actor: int,
                            min_task_actors: int) -> pd.DataFrame:
    """Per-task Pearson r between feat and attr, restricted to actors with
    >= min_clips_per_actor clips of that task. Returns DataFrame sorted by |r|."""
    rows = []
    for task, tdf in atf.groupby("content_type_of_movement"):
        elig = tdf.copy()
        elig["n_clips"] = elig["actor_uid"].map(
            lambda uid: int(clip_counts.get((uid, task), 0))
        )
        elig = elig[elig["n_clips"] >= min_clips_per_actor]
        elig = elig.dropna(subset=[attr, feat])
        n = len(elig)
        if n < min_task_actors:
            continue
        f = elig[feat].to_numpy(dtype=float)
        a = elig[attr].to_numpy(dtype=float)
        if np.std(f) < 1e-9 or np.std(a) < 1e-9:
            continue
        r, p = sp_stats.pearsonr(f, a)
        if not np.isfinite(r):
            continue
        rows.append({"task": task, "r": float(r), "p": float(p), "n_actors": n})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["abs_r"] = out["r"].abs()
    return out.sort_values("abs_r", ascending=False).reset_index(drop=True)


def make_side_block(rec, feat: str, attr: str, csv_rel: str, n_frames: int,
                    source_bvh, per_clip_feats):
    """Per-side JSON block. Identical shape to select_pair_candidates._make_actor_block
    but slimmer (no `per_actor_F` duplication of attribute name)."""
    return {
        "actor_uid": str(rec["actor_uid"]),
        "attr_value": float(rec[attr]),
        "actor_age_yr": float(rec["actor_age_yr"]),
        "actor_height_cm": float(rec["actor_height_cm"]),
        "actor_weight_kg": float(rec["actor_weight_kg"]),
        "actor_gender": str(rec["actor_gender"]),
        "per_actor_F": float(rec[feat]),
        "per_clip_F": (None if per_clip_feats is None else float(per_clip_feats[feat])),
        "g1_csv": csv_rel,
        "source_bvh": source_bvh,
        "n_frames": int(n_frames),
    }


def _materialize_side(side_rec, feat: str, attr: str, task: str,
                      actor_task_clips_grp, cache, bs: Path, compute_fn) -> dict | None:
    """Pick the clip of this (actor, task) whose per-clip F is closest to the
    actor's per-task mean F (most representative). Returns the JSON block or None."""
    uid = side_rec["actor_uid"]
    target_F = float(side_rec[feat])
    try:
        actor_clips = actor_task_clips_grp.get_group((uid, task))
    except KeyError:
        return None
    best = None  # (diff, clip_row, per_clip_dict)
    for clip_row in actor_clips.itertuples():
        csv_rel = str(clip_row.move_g1_mujoco_path).replace("\\", "/")
        per_clip = compute_per_clip_feature(bs / csv_rel, compute_fn, cache, csv_rel)
        if per_clip is None or feat not in per_clip:
            continue
        diff = abs(per_clip[feat] - target_F)
        if best is None or diff < best[0]:
            best = (diff, clip_row, per_clip)
    if best is None:
        return None
    _, clip_row, per_clip = best
    csv_rel = str(clip_row.move_g1_mujoco_path).replace("\\", "/")
    bvh = getattr(clip_row, "move_soma_uniform_path", None)
    bvh = str(bvh).replace("\\", "/") if isinstance(bvh, str) else None
    n_frames = int(clip_row.nf)
    return make_side_block(side_rec, feat, attr, csv_rel, n_frames, bvh, per_clip)


def try_row(pool: pd.DataFrame, attr: str, feat: str, predicted_sign: int,
            task: str, actor_task_clips_grp, cache, bs: Path,
            compute_fn) -> tuple[dict, dict, str] | None:
    """Try extreme selection first (typical-of-extreme-decile actors); if the
    per-clip gap inverts, fall back to gap-pair selection (largest sign-correct
    per-actor F gap). Returns (high_block, low_block, strategy) or None."""
    def materialize_pair(strategy_pair):
        if strategy_pair is None:
            return None
        hi, lo = strategy_pair
        hi_blk = _materialize_side(hi, feat, attr, task, actor_task_clips_grp, cache, bs, compute_fn)
        lo_blk = _materialize_side(lo, feat, attr, task, actor_task_clips_grp, cache, bs, compute_fn)
        if hi_blk is None or lo_blk is None:
            return None
        if hi_blk["per_clip_F"] is None or lo_blk["per_clip_F"] is None:
            return None
        return hi_blk, lo_blk

    for strategy_name, picker in [("extreme", pick_extreme_pair), ("gap", pick_gap_pair)]:
        res = materialize_pair(picker(pool, attr, feat, predicted_sign))
        if res is None:
            continue
        hi_blk, lo_blk = res
        if hi_blk["per_clip_F"] - lo_blk["per_clip_F"] > 0:
            return hi_blk, lo_blk, strategy_name
    return None


def validate_pair(high: dict, low: dict, r: float, attr: str) -> list[str]:
    """Return a list of warnings (empty = ok). Hard failures stop selection upstream."""
    warns = []
    sign_r = 1 if r > 0 else (-1 if r < 0 else 0)
    if high["per_clip_F"] is None or low["per_clip_F"] is None:
        warns.append("per_clip_F missing")
        return warns
    f_gap = high["per_clip_F"] - low["per_clip_F"]
    a_gap = high["attr_value"] - low["attr_value"]
    if f_gap <= 0:
        warns.append(f"per-clip F gap inverted ({f_gap:+.3f})")
    if sign_r != 0 and np.sign(a_gap) != sign_r:
        warns.append(f"attr-gap sign mismatch (a_gap={a_gap:+.3f}, sign(r)={sign_r:+d})")
    return warns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bones-seed", required=True, type=Path)
    ap.add_argument("--results", type=Path, default=None,
                    help="Defaults to <bones-seed>/Correlation_V2/simple_interpretability_results/")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    ap.add_argument("--n-per-attr", type=int, default=10)
    ap.add_argument("--min-abs-r", type=float, default=0.1,
                    help="Drop tasks whose |Pearson r| is below this. Floor on the "
                         "evidence strength shown in the bar plot.")
    ap.add_argument("--min-task-actors", type=int, default=50,
                    help="Skip tasks with fewer eligible actors (kills noise-driven |r|≈1).")
    ap.add_argument("--min-clips-per-actor", type=int, default=2)
    ap.add_argument("--min-frames", type=int, default=200)
    ap.add_argument("--style", type=str, default="neutral",
                    help="Restrict per-clip picks to this content_uniform_style "
                         "(default 'neutral' so we don't show injured/hurry clips). "
                         "Empty string disables.")
    args = ap.parse_args()

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
        before = len(cand)
        cand = cand[cand["content_uniform_style"].astype(str) == args.style]
        print(f"  style filter '{args.style}': {before:,} -> {len(cand):,} clips")
    clip_counts = cand.groupby(["actor_uid", "content_type_of_movement"]).size()
    actor_task_clips_grp = cand.groupby(["actor_uid", "content_type_of_movement"])
    print(f"  -> {len(cand):,} usable clips, {len(clip_counts):,} (actor, task) cells")

    cache_path = out / "per_clip_features.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    print(f"[cache] per_clip_features.json: {len(cache)} entries (pre-existing)")

    all_rows: list[dict] = []
    summary_rows: list[dict] = []

    for attr, feat in ATTR_FEATURE.items():
        attr_short = ATTR_SHORT[attr]
        print(f"\n=== {attr_short.upper()} (feature: {feat}) ===")

        scored = score_tasks_for_feature(
            atf, attr, feat,
            clip_counts=clip_counts,
            min_clips_per_actor=args.min_clips_per_actor,
            min_task_actors=args.min_task_actors,
        )
        if scored.empty:
            print("  ! no tasks passed gating; skipping attribute"); continue
        before_r = len(scored)
        scored = scored[scored["abs_r"] >= args.min_abs_r].reset_index(drop=True)
        if before_r != len(scored):
            print(f"  [|r|>={args.min_abs_r}] {before_r} -> {len(scored)} tasks")
        if scored.empty:
            print(f"  ! no tasks meet |r|>={args.min_abs_r}; skipping attribute"); continue
        print(f"  [{len(scored)} tasks scored]  top 5:")
        for _, s in scored.head(5).iterrows():
            print(f"    {s['task']:30s}  r={s['r']:+.3f}  n_actors={int(s['n_actors'])}")

        accepted = 0
        for _, s in scored.iterrows():
            if accepted >= args.n_per_attr:
                break
            task = s["task"]
            r_val = s["r"]
            p_val = s["p"]
            predicted_sign = 1 if r_val > 0 else -1

            # Eligible actor pool: have entry in atf, enough clips, and a longest-clip row.
            pool = atf[atf["content_type_of_movement"] == task].copy()
            pool["n_clips"] = pool["actor_uid"].map(
                lambda uid: int(clip_counts.get((uid, task), 0))
            )
            pool = pool[pool["n_clips"] >= args.min_clips_per_actor]
            pool = pool.dropna(subset=[attr, feat])
            pool = pool[pool["actor_uid"].apply(
                lambda uid: (uid, task) in clip_counts.index
            )]
            if len(pool) < 4:
                continue

            result = try_row(pool, attr, feat, predicted_sign, task,
                             actor_task_clips_grp, cache, bs, compute_clip_features)
            if result is None:
                continue
            high, low, strategy = result

            warns = validate_pair(high, low, r_val, attr)
            if any("inverted" in w or "mismatch" in w for w in warns):
                # Hard skip — sign violations defeat the whole point of the panel.
                print(f"  [skip] {task:30s} r={r_val:+.3f}  | {'; '.join(warns)}")
                continue
            for w in warns:
                print(f"  [warn] {task:30s} {w}")

            row_id = f"{attr_short}_{slugify(task)}_{slugify(feat)}"
            all_rows.append({
                "row_id": row_id,
                "attribute": attr,
                "attribute_short": attr_short,
                "attribute_unit": ATTR_UNIT[attr],
                "feature": feat,
                "feature_recipe": build_feature_recipe(feat),
                "task": task,
                "r": float(r_val),
                "p": float(p_val),
                "n_actors_scored": int(s["n_actors"]),
                "n_actors_pool": int(len(pool)),
                "predicted_sign": predicted_sign,
                "strategy": strategy,
                "high": high,
                "low": low,
            })
            print(f"  [keep:{strategy:7s}] {task:30s} r={r_val:+.3f}  "
                  f"hi={high['per_clip_F']:.2f} (A={high['attr_value']:.1f})  "
                  f"lo={low['per_clip_F']:.2f} (A={low['attr_value']:.1f})")
            accepted += 1

            for side, blk in (("high", high), ("low", low)):
                summary_rows.append({
                    "row_id": row_id, "attribute": attr_short, "task": task, "feature": feat,
                    "r": r_val, "side": side,
                    "actor_uid": blk["actor_uid"],
                    "attr_value": blk["attr_value"],
                    "per_actor_F": blk["per_actor_F"],
                    "per_clip_F": blk["per_clip_F"],
                    "n_frames": blk["n_frames"],
                    "g1_csv": blk["g1_csv"],
                })

        if accepted < args.n_per_attr:
            print(f"  ! only accepted {accepted}/{args.n_per_attr} for {attr_short}")

    # Save cache
    cache_path.write_text(json.dumps(cache, indent=1))
    print(f"\n[cache] wrote {cache_path} ({len(cache)} entries)")

    out_json = out / "fixed_feature_pairs.json"
    out_json.write_text(json.dumps({
        "attribute_feature": ATTR_FEATURE,
        "n_per_attr": args.n_per_attr,
        "params": {
            "min_abs_r": args.min_abs_r,
            "min_task_actors": args.min_task_actors,
            "min_clips_per_actor": args.min_clips_per_actor,
            "min_frames": args.min_frames,
            "style": args.style,
            "decile": 0.10,
        },
        "rows": all_rows,
    }, indent=2))
    print(f"[out] wrote {out_json} ({len(all_rows)} rows)")

    out_csv = out / "fixed_feature_pairs_summary.csv"
    fields = ["row_id", "attribute", "task", "feature", "r", "side",
              "actor_uid", "attr_value", "per_actor_F", "per_clip_F", "n_frames", "g1_csv"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for s in summary_rows:
            w.writerow(s)
    print(f"[out] wrote {out_csv} ({len(summary_rows)} sides)")

    print("\n=== counts per attribute ===")
    by_attr = pd.DataFrame(all_rows).groupby("attribute_short").size() if all_rows else None
    if by_attr is not None:
        for a, n in by_attr.items():
            print(f"  {a:7s}  {n}/{args.n_per_attr}")


if __name__ == "__main__":
    main()
