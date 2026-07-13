"""
select_pair_candidates.py
-------------------------
Stage 1 of the data-driven feature-visualization pipeline.

Reads bones-seed/Correlation_V2/simple_interpretability_results/ and picks the
top-N (task, feature) correlations per attribute (deduping repeats of the same
feature so the picks are diverse joint groups). For each chosen row it emits
TWO candidate operator pairs:

  - `extreme`: top-10% and bottom-10% of actors by attribute, each side
    represented by the actor whose per-actor-mean of the feature sits closest
    to that decile's median (so neither side is an outlier on the feature).

  - `gap`: actor pair with the largest |F_P - F_Q| that still satisfies
    sign(A_P - A_Q) == sign(F_P - F_Q) == sign(r) — biggest feature gap that
    still matches the predicted direction.

For each chosen actor we use their *longest* (non-mirror) clip of the task as
the representative clip (matches pick_jumping_pair.py convention). We then
compute that clip's per-clip feature value with the exact same function the
bones-seed correlation script used (compute_clip_features in
simple_interpretability.py:164) so the value is reconcilable.

Outputs:
  data/pair_candidates.json         — for user review; user adds "chosen" per row
  data/pair_candidates_summary.csv  — flat scan-friendly view (one row per candidate)
  data/per_clip_features.json       — cache of computed per-clip feature dicts

Usage:
  python select_pair_candidates.py --bones-seed C:/Users/sihat/Downloads/bones-seed
"""

import argparse
import csv
import json
import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ── joint-group → CSV column name mapping ────────────────────────────────
# Mirrors the positional indices in simple_interpretability.py:164-222.
# Used to (a) serialize the feature recipe into pair_candidates.json so
# build_feature_matrix.py + the JS viewer can compute the same per-frame series.
JOINT_GROUPS = {
    "root_translate": {
        "kind": "root_translate", "agg": "multi_axis",
        "joints": ["root_translateX", "root_translateY", "root_translateZ"],
    },
    "root_rotate": {
        "kind": "root_rotate", "agg": "multi_axis",
        "joints": ["root_rotateX", "root_rotateY", "root_rotateZ"],
    },
    "hip": {
        "kind": "joint_group", "agg": "lr_merge",
        "left":  ["left_hip_pitch_joint_dof", "left_hip_roll_joint_dof", "left_hip_yaw_joint_dof"],
        "right": ["right_hip_pitch_joint_dof", "right_hip_roll_joint_dof", "right_hip_yaw_joint_dof"],
    },
    "knee": {
        "kind": "joint_group", "agg": "lr_merge",
        "left":  ["left_knee_joint_dof"],
        "right": ["right_knee_joint_dof"],
    },
    "ankle": {
        "kind": "joint_group", "agg": "lr_merge",
        "left":  ["left_ankle_pitch_joint_dof", "left_ankle_roll_joint_dof"],
        "right": ["right_ankle_pitch_joint_dof", "right_ankle_roll_joint_dof"],
    },
    "waist": {
        "kind": "joint_group", "agg": "multi_axis",
        "joints": ["waist_yaw_joint_dof", "waist_roll_joint_dof", "waist_pitch_joint_dof"],
    },
    "shoulder": {
        "kind": "joint_group", "agg": "lr_merge",
        "left":  ["left_shoulder_pitch_joint_dof", "left_shoulder_roll_joint_dof", "left_shoulder_yaw_joint_dof"],
        "right": ["right_shoulder_pitch_joint_dof", "right_shoulder_roll_joint_dof", "right_shoulder_yaw_joint_dof"],
    },
    "elbow": {
        "kind": "joint_group", "agg": "lr_merge",
        "left":  ["left_elbow_joint_dof"],
        "right": ["right_elbow_joint_dof"],
    },
    "wrist": {
        "kind": "joint_group", "agg": "lr_merge",
        "left":  ["left_wrist_roll_joint_dof", "left_wrist_pitch_joint_dof", "left_wrist_yaw_joint_dof"],
        "right": ["right_wrist_roll_joint_dof", "right_wrist_pitch_joint_dof", "right_wrist_yaw_joint_dof"],
    },
}

# stat suffix → display unit (matches simple_interpretability.py's raw units:
# degrees for joints, cm for root translate, both un-converted from CSV)
def feature_unit(feat_name: str) -> str:
    if feat_name.startswith("root_translate"):
        return "cm/s" if feat_name.endswith("vel") else "cm"
    return "deg/s" if feat_name.endswith("vel") else "deg"


def split_feature(feat_name: str) -> tuple[str, str]:
    """waist_rom → ('waist', 'rom'); root_translate_mean_vel → ('root_translate', 'mean_vel')."""
    for stat in ("mean_vel", "peak_vel", "rom"):
        if feat_name.endswith("_" + stat):
            return feat_name[: -(len(stat) + 1)], stat
    raise ValueError(f"unrecognized feature name: {feat_name}")


def build_feature_recipe(feat_name: str) -> dict:
    group, stat = split_feature(feat_name)
    spec = JOINT_GROUPS[group]
    recipe = {
        "name": feat_name, "stat": stat,
        "kind": spec["kind"], "agg": spec["agg"],
        "unit": feature_unit(feat_name),
    }
    if spec["agg"] == "lr_merge":
        recipe["left"]  = list(spec["left"])
        recipe["right"] = list(spec["right"])
    else:
        recipe["joints"] = list(spec["joints"])
    return recipe


ATTR_SHORT = {
    "actor_age_yr":   "age",
    "actor_weight_kg":"weight",
    "actor_height_cm":"height",
    "gender_numeric": "gender",
}

ATTR_UNIT = {
    "actor_age_yr":   "yr",
    "actor_weight_kg":"kg",
    "actor_height_cm":"cm",
    "gender_numeric": "",
}


def slugify(s: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]+", "_", s)).strip("_").lower()


# ── pick top-N (task, feature) pairs per attribute, deduping feature names ──
def pick_rows_per_attribute(top_json: dict, task_n_actors: dict[str, int],
                            top_n: int = 3, min_task_actors: int = 50,
                            unique_tasks: bool = True,
                            unique_features: bool = True,
                            exclude_row_ids: set | None = None,
                            exclude_tasks: set | None = None) -> list[dict]:
    """Returns a list of {attribute, task, feature, r, p, n_actors, predicted_sign}.

    Filters out tasks with fewer than `min_task_actors` actors (spurious r from
    tiny samples) BEFORE ranking. By default the top-N per attribute are
    deduped on BOTH task and feature, so the 3 rows show different activities
    AND different joint groups — maximally diverse evidence for the leak.
    """
    exclude_row_ids = exclude_row_ids or set()
    exclude_tasks   = exclude_tasks   or set()
    out: list[dict] = []
    for attr in ATTR_SHORT:
        attr_short = ATTR_SHORT[attr]
        flat = []
        for task, blocks in top_json.items():
            n_actors = task_n_actors.get(task, 0)
            if n_actors < min_task_actors:
                continue
            if (attr_short, task) in exclude_tasks:
                continue
            if attr not in blocks: continue
            for entry in blocks[attr]:
                r = entry.get("r")
                if r is None or not np.isfinite(r):
                    continue
                row_id = f"{attr_short}_{slugify(task)}_{slugify(entry['feature'])}"
                if row_id in exclude_row_ids:
                    continue
                flat.append({
                    "task": task,
                    "feature": entry["feature"],
                    "r": float(r),
                    "p": float(entry.get("p", float("nan"))),
                    "n_actors": int(n_actors),
                })
        flat.sort(key=lambda d: abs(d["r"]), reverse=True)
        kept, seen_tasks, seen_features = [], set(), set()
        for d in flat:
            if unique_tasks    and d["task"]    in seen_tasks:    continue
            if unique_features and d["feature"] in seen_features: continue
            seen_tasks.add(d["task"]); seen_features.add(d["feature"])
            kept.append(d)
            if len(kept) == top_n:
                break
        for d in kept:
            out.append({
                "attribute": attr,
                "task": d["task"],
                "feature": d["feature"],
                "r": d["r"],
                "p": d["p"],
                "n_actors": d["n_actors"],
                "predicted_sign": 1 if d["r"] > 0 else -1,
            })
    return out


# ── per-clip feature compute (cached on disk) ────────────────────────────
def compute_per_clip_feature(csv_path: Path, compute_fn, cache: dict, key: str) -> dict | None:
    if key in cache:
        return cache[key]
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if "Frame" in df.columns:
            df = df.drop(columns=["Frame"])
        x = df.to_numpy(dtype=np.float32)
        if x.shape[1] < 35:
            return None
        feats = compute_fn(x, fps=120.0)
        feats = {k: float(v) for k, v in feats.items()}
        cache[key] = feats
        return feats
    except Exception as e:
        print(f"  ! per-clip compute failed for {csv_path.name}: {e}", file=sys.stderr)
        return None


# ── candidate pair selection ─────────────────────────────────────────────
def _make_actor_block(rec: dict, feat_name: str, attr: str, csv_rel: str,
                      n_frames: int, source_bvh: str | None,
                      per_clip_feats: dict | None) -> dict:
    return {
        "actor_uid": rec["actor_uid"],
        "attr_value": float(rec[attr]),
        "actor_height_cm": float(rec["actor_height_cm"]),
        "actor_weight_kg": float(rec["actor_weight_kg"]),
        "actor_age_yr": float(rec["actor_age_yr"]),
        "actor_gender": str(rec["actor_gender"]),
        "per_actor_F": float(rec[feat_name]),
        "per_clip_F": (None if per_clip_feats is None else float(per_clip_feats[feat_name])),
        "g1_csv": csv_rel,
        "source_bvh": source_bvh,
        "n_frames": int(n_frames),
    }


def pick_extreme_pair(actors: pd.DataFrame, attr: str, feat: str, predicted_sign: int,
                      decile: float = 0.10):
    """Top-decile and bottom-decile by attr, each represented by the actor whose
    per-actor F is closest to that decile's median F."""
    n = len(actors)
    if n < 4:
        return None
    sorted_by_attr = actors.sort_values(attr).reset_index(drop=True)
    k = max(1, int(round(n * decile)))
    bottom = sorted_by_attr.iloc[:k]
    top    = sorted_by_attr.iloc[-k:]
    # high side = predicted-larger-F side
    high_pool, low_pool = (top, bottom) if predicted_sign > 0 else (bottom, top)
    def representative(pool):
        med = pool[feat].median()
        return pool.iloc[(pool[feat] - med).abs().argsort().iloc[0]]
    return representative(high_pool), representative(low_pool)


def pick_pair_match_content_name(task_clips_df: pd.DataFrame, atf_pool: pd.DataFrame,
                                 attr: str, feat: str, predicted_sign: int,
                                 compute_fn, cache: dict, bs: Path,
                                 min_actors_per_cn: int = 4,
                                 exclude_cn: set | None = None):
    """Pick extreme/gap pairs but restrict to actors that share the same content_name.

    For each content_name within `task_clips_df` (already filtered to this task,
    neutral, valid CSV), find actors who have at least one clip with that
    content_name. If ≥ min_actors_per_cn, compute per-clip F on each actor's
    LONGEST clip of that content_name, then run extreme/gap selection using
    that per-clip F as the actor's F value. Score candidate pairs by absolute
    per-clip F gap and return the best (extreme, gap) candidates with the
    biggest gap.
    """
    # Dedupe the column list — for age/height/weight rows, `attr` is already
    # one of the four biometrics and pandas would otherwise return a 2-col
    # DataFrame on `biom[attr]` instead of a scalar.
    cols = list(dict.fromkeys([attr, "actor_age_yr",
        "actor_height_cm", "actor_weight_kg", "actor_gender"]))
    actor_biom = (atf_pool.drop_duplicates("actor_uid")
                  .set_index("actor_uid")[cols])
    best = {"extreme": None, "gap": None, "extreme_cn": None, "gap_cn": None}
    exclude_cn = exclude_cn or set()

    # group: per content_name -> list of (actor_uid, clip_row)
    for cn, grp in task_clips_df.groupby("content_name"):
        if cn in exclude_cn: continue
        # one clip per actor (longest of this content_name)
        grp = grp.sort_values("nf", ascending=False).drop_duplicates("actor_uid", keep="first")
        # keep only actors with biometrics in atf_pool
        grp = grp[grp["actor_uid"].isin(actor_biom.index)]
        if len(grp) < min_actors_per_cn: continue

        # compute per-clip F for each actor's chosen clip
        rows = []
        for r in grp.itertuples():
            csv_rel = str(r.move_g1_mujoco_path).replace("\\", "/")
            f = compute_per_clip_feature(bs / csv_rel, compute_fn, cache, csv_rel)
            if f is None or feat not in f: continue
            biom = actor_biom.loc[r.actor_uid]
            rows.append({
                "actor_uid": r.actor_uid,
                attr: float(biom[attr]),
                feat: float(f[feat]),                # per-CLIP F at this content_name
                "actor_age_yr":    float(biom["actor_age_yr"]),
                "actor_height_cm": float(biom["actor_height_cm"]),
                "actor_weight_kg": float(biom["actor_weight_kg"]),
                "actor_gender":    str(biom["actor_gender"]),
                "g1_csv": csv_rel,
                "source_bvh": str(r.move_soma_uniform_path).replace("\\", "/") if isinstance(r.move_soma_uniform_path, str) else None,
                "n_frames": int(r.nf),
                "content_name": cn,
            })
        if len(rows) < min_actors_per_cn: continue
        sub = pd.DataFrame(rows)

        def _sign_ok(hi, lo):
            # Require the attribute gap to follow sign(r) AND the high side to
            # actually have larger F (since "high" is defined as predicted-larger).
            if predicted_sign == 0: return True
            if np.sign(hi[attr] - lo[attr]) != predicted_sign: return False
            if hi[feat] - lo[feat] <= 0: return False
            return True

        # extreme
        ext = pick_extreme_pair(sub, attr, feat, predicted_sign)
        if ext is not None:
            hi, lo = ext
            if _sign_ok(hi, lo):
                gap = abs(hi[feat] - lo[feat])
                if best["extreme"] is None or gap > abs(best["extreme"][0][feat] - best["extreme"][1][feat]):
                    best["extreme"]    = (hi, lo)
                    best["extreme_cn"] = cn

        # gap (pick_gap_pair already enforces signs, but recheck defensively)
        g = pick_gap_pair(sub, attr, feat, predicted_sign)
        if g is not None:
            hi, lo = g
            if _sign_ok(hi, lo):
                gap = abs(hi[feat] - lo[feat])
                if best["gap"] is None or gap > abs(best["gap"][0][feat] - best["gap"][1][feat]):
                    best["gap"]    = (hi, lo)
                    best["gap_cn"] = cn

    return best


def pick_gap_pair(actors: pd.DataFrame, attr: str, feat: str, predicted_sign: int):
    """Pair (P, Q) with largest |F_P - F_Q| s.t. sign(A_P-A_Q) == sign(F_P-F_Q) == sign(r)."""
    if len(actors) < 2:
        return None
    a = actors[attr].to_numpy()
    f = actors[feat].to_numpy()
    n = len(actors)
    best = (-1.0, None, None)  # score, hi_idx, lo_idx
    # vectorize pairwise sign check
    for i in range(n):
        ai, fi = a[i], f[i]
        da = a - ai            # A_Q - A_P
        df = f - fi            # F_Q - F_P
        # we want pairs where sign matches and i is the "low F" side; pick j as high
        valid = (np.sign(da) == predicted_sign) & (np.sign(df) > 0)
        if not valid.any():
            continue
        scores = np.abs(df) * valid
        j = int(scores.argmax())
        if scores[j] > best[0]:
            best = (float(scores[j]), j, i)
    if best[1] is None:
        return None
    hi = actors.iloc[best[1]]
    lo = actors.iloc[best[2]]
    return hi, lo


# ── main ─────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bones-seed", required=True, type=Path)
    ap.add_argument("--results", type=Path, default=None,
                    help="Defaults to <bones-seed>/Correlation_V2/simple_interpretability_results/")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    ap.add_argument("--top-n", type=int, default=3,
                    help="Top-N (task, feature) per attribute (deduped by feature).")
    ap.add_argument("--min-task-actors", type=int, default=50,
                    help="Skip tasks with fewer than this many actors (filters noise-driven |r|≈1).")
    ap.add_argument("--allow-repeat-tasks", dest="unique_tasks", action="store_false", default=True,
                    help="Let the same task appear in multiple top-N rows per attribute.")
    ap.add_argument("--allow-repeat-features", dest="unique_features", action="store_false", default=True,
                    help="Let the same feature appear in multiple top-N rows per attribute.")
    ap.add_argument("--exclude-row-id", nargs="*", default=[],
                    help="Drop these row_ids (auto-named '<attr>_<task>_<feature>') and "
                         "fall through to the next-best alternative.")
    ap.add_argument("--exclude-task", nargs="*", default=[],
                    help="Drop entire (attribute, task) combos. Format: 'attr:task' "
                         "(e.g. 'age:gesture, dancing'). All features of that task are skipped.")
    ap.add_argument("--exclude-content-name", nargs="*", default=[],
                    help="Skip these content_names within specific rows. "
                         "Format: '<row_id>:<content_name>' (e.g. "
                         "'height_jogging_ankle_rom:jog_ff_stop_225_R'). "
                         "Forces the selector to fall through to the next-best "
                         "content_name for that row. Repeatable.")
    ap.add_argument("--match-content-name", action="store_true", default=False,
                    help="Restrict each pair so both clips share the same content_name "
                         "(fine-grained motion subtype). Pairs become visually comparable "
                         "but pool sizes shrink — may fail on small tasks.")
    ap.add_argument("--min-actors-per-cn", type=int, default=4,
                    help="With --match-content-name: skip content_names with fewer actors "
                         "than this. 4 lets extreme/gap selection work; bigger = more reliable.")
    ap.add_argument("--min-clips-per-actor", type=int, default=2,
                    help="Drop actors with fewer than this many clips of the task.")
    ap.add_argument("--min-frames", type=int, default=200,
                    help="Skip clips shorter than this many frames.")
    ap.add_argument("--style", type=str, default="neutral",
                    help="Restrict per-clip picks to this content_uniform_style "
                         "(default 'neutral' so we never display injured/hurry/etc. clips). "
                         "Pass empty string to disable.")
    args = ap.parse_args()

    bs = args.bones_seed.resolve()
    results_dir = (args.results or bs / "Correlation_V2" / "simple_interpretability_results").resolve()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)

    # import compute_clip_features from the bones-seed correlation script
    si_dir = bs / "Correlation_V2"
    sys.path.insert(0, str(si_dir))
    from simple_interpretability import compute_clip_features  # type: ignore

    # 1. top-N (task, feature) per attribute (filtered by task sample size)
    print(f"[load] top_features_per_task_attr.json + task_attr_top3_summary.csv")
    top_json = json.loads((results_dir / "top_features_per_task_attr.json").read_text())
    summary = pd.read_csv(results_dir / "task_attr_top3_summary.csv")
    task_n_actors = dict(zip(summary["content_type_of_movement"], summary["num_actors"]))
    excl_tasks = set()
    for spec in args.exclude_task:
        if ":" not in spec:
            print(f"  ! bad --exclude-task '{spec}', expected 'attr:task'"); continue
        a, t = spec.split(":", 1)
        excl_tasks.add((a.strip(), t.strip()))
    excl_row_ids = set(args.exclude_row_id)
    excl_cn_by_row: dict[str, set] = {}
    for spec in args.exclude_content_name:
        if ":" not in spec:
            print(f"  ! bad --exclude-content-name '{spec}', expected 'row_id:cn'"); continue
        rid, cn = spec.split(":", 1)
        excl_cn_by_row.setdefault(rid.strip(), set()).add(cn.strip())
    if excl_tasks:    print(f"  exclude tasks: {sorted(excl_tasks)}")
    if excl_row_ids:  print(f"  exclude row_ids: {sorted(excl_row_ids)}")
    if excl_cn_by_row:print(f"  exclude content_names per row: {dict((k, sorted(v)) for k,v in excl_cn_by_row.items())}")
    rows = pick_rows_per_attribute(top_json, task_n_actors,
                                   top_n=args.top_n, min_task_actors=args.min_task_actors,
                                   unique_tasks=args.unique_tasks,
                                   unique_features=args.unique_features,
                                   exclude_row_ids=excl_row_ids,
                                   exclude_tasks=excl_tasks)
    print(f"  [{len(rows)} rows] {args.top_n} per attribute x {len(ATTR_SHORT)} attributes, "
          f"task n_actors>={args.min_task_actors}")
    for r in rows:
        print(f"    {ATTR_SHORT[r['attribute']]:7s}  {r['task']:30s}  {r['feature']:25s}  "
              f"r={r['r']:+.3f}  n_actors={r['n_actors']}")

    # 2. actor_task_features.csv → per-actor mean F + biometrics
    print(f"[load] actor_task_features.csv")
    atf = pd.read_csv(results_dir / "actor_task_features.csv")
    print(f"  → {len(atf)} actor-task rows, {atf['actor_uid'].nunique()} actors")

    # 3. metadata parquet for clip paths
    print(f"[load] metadata parquet")
    meta = pd.read_parquet(bs / "metadata" / "seed_metadata_v003.parquet")
    cand = meta[~meta["move_g1_mujoco_path"].astype(str).str.endswith("_M.csv")].copy()
    cand["abs_csv"] = cand["move_g1_mujoco_path"].apply(lambda r: bs / r if isinstance(r, str) else None)
    cand = cand[cand["abs_csv"].apply(lambda p: p is not None and Path(p).exists())]
    cand["nf"] = pd.to_numeric(cand["move_duration_frames"], errors="coerce")
    cand = cand[cand["nf"].fillna(0) >= args.min_frames]
    if args.style:
        before = len(cand)
        cand = cand[cand["content_uniform_style"].astype(str) == args.style]
        print(f"  style filter '{args.style}': {before} -> {len(cand)} clips "
              f"(drops injured/hurry/old/etc.)")
    # group: per (actor, task) — count clips, take longest
    cand_sorted = cand.sort_values(["actor_uid", "content_type_of_movement", "nf"],
                                   ascending=[True, True, False])
    longest_per_actor_task = cand_sorted.drop_duplicates(
        subset=["actor_uid", "content_type_of_movement"], keep="first"
    ).set_index(["actor_uid", "content_type_of_movement"])
    clip_counts = cand.groupby(["actor_uid", "content_type_of_movement"]).size()
    print(f"  → {len(cand)} usable clips, {len(longest_per_actor_task)} (actor, task) cells")

    # 4. per-clip feature cache (loaded if exists, written at end)
    cache_path = out / "per_clip_features.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    print(f"[cache] per_clip_features.json: {len(cache)} entries (pre-existing)")

    # 5. build candidates per row
    output_rows = []
    summary_rows = []
    for r in rows:
        attr, task, feat, r_val, p_val, sign = (
            r["attribute"], r["task"], r["feature"], r["r"], r["p"], r["predicted_sign"]
        )
        row_id = f"{ATTR_SHORT[attr]}_{slugify(task)}_{slugify(feat)}"
        print(f"\n[row] {row_id}  (r={r_val:+.3f})")

        # actor pool: actor_task_features for this task, with enough clips
        pool = atf[atf["content_type_of_movement"] == task].copy()
        # join clip counts (default 0 if missing)
        pool["n_clips"] = pool["actor_uid"].map(
            lambda uid: int(clip_counts.get((uid, task), 0))
        )
        pool = pool[pool["n_clips"] >= args.min_clips_per_actor]
        pool = pool.dropna(subset=[attr, feat])
        # restrict to actors who actually have a longest_per_actor_task entry
        pool = pool[pool["actor_uid"].apply(lambda uid: (uid, task) in longest_per_actor_task.index)]
        if len(pool) < 4:
            print(f"  ! only {len(pool)} eligible actors; skipping row")
            continue
        print(f"  pool: {len(pool)} actors")

        candidates = {}
        cn_used = {}
        if args.match_content_name:
            # Restrict each pair to actors sharing the same content_name (the
            # fine-grained motion subtype) so the two clips show the same kind
            # of motion, not just the same coarse task category.
            task_clips = cand[cand["content_type_of_movement"] == task]
            best_cn = pick_pair_match_content_name(
                task_clips, pool, attr, feat, sign,
                compute_clip_features, cache, bs,
                min_actors_per_cn=args.min_actors_per_cn,
                exclude_cn=excl_cn_by_row.get(row_id, set()),
            )
            if best_cn["extreme"] is None and best_cn["gap"] is None:
                print(f"  ! no content_name had ≥{args.min_actors_per_cn} actors "
                      f"with sign-correct pair; skipping row")
                continue

            def materialize_cn(side_rec):
                csv_rel = side_rec["g1_csv"]
                per_clip = {feat: side_rec[feat]}  # already computed
                return _make_actor_block(side_rec, feat, attr, csv_rel,
                                          side_rec["n_frames"],
                                          side_rec["source_bvh"], per_clip)
            if best_cn["extreme"]:
                hi, lo = best_cn["extreme"]
                candidates["extreme"] = {"high": materialize_cn(hi), "low": materialize_cn(lo)}
                cn_used["extreme"] = best_cn["extreme_cn"]
                print(f"    extreme cn={best_cn['extreme_cn']!r}: high A={hi[attr]:.1f} F={hi[feat]:.2f}  | "
                      f"low A={lo[attr]:.1f} F={lo[feat]:.2f}")
            if best_cn["gap"]:
                hi, lo = best_cn["gap"]
                candidates["gap"] = {"high": materialize_cn(hi), "low": materialize_cn(lo)}
                cn_used["gap"] = best_cn["gap_cn"]
                print(f"    gap     cn={best_cn['gap_cn']!r}: high A={hi[attr]:.1f} F={hi[feat]:.2f}  | "
                      f"low A={lo[attr]:.1f} F={lo[feat]:.2f}")
        else:
            extreme = pick_extreme_pair(pool, attr, feat, sign)
            gap     = pick_gap_pair(pool, attr, feat, sign)

            if extreme is None and gap is None:
                print(f"  ! no candidates found; skipping row")
                continue

            def materialize(side_rec):
                uid = side_rec["actor_uid"]
                clip = longest_per_actor_task.loc[(uid, task)]
                csv_rel = str(clip["move_g1_mujoco_path"]).replace("\\", "/")
                bvh_rel = clip.get("move_soma_uniform_path")
                bvh_rel = str(bvh_rel).replace("\\", "/") if isinstance(bvh_rel, str) else None
                n_frames = int(clip["nf"])
                per_clip = compute_per_clip_feature(
                    bs / csv_rel, compute_clip_features, cache, csv_rel,
                )
                return _make_actor_block(side_rec, feat, attr, csv_rel, n_frames, bvh_rel, per_clip)

            if extreme is not None:
                hi, lo = extreme
                candidates["extreme"] = {"high": materialize(hi), "low": materialize(lo)}
                print(f"    extreme: high A={hi[attr]:.1f} F={hi[feat]:.2f}  | "
                      f"low A={lo[attr]:.1f} F={lo[feat]:.2f}")
            if gap is not None:
                hi, lo = gap
                candidates["gap"] = {"high": materialize(hi), "low": materialize(lo)}
                print(f"    gap:     high A={hi[attr]:.1f} F={hi[feat]:.2f}  | "
                      f"low A={lo[attr]:.1f} F={lo[feat]:.2f}")

        # sanity at the per-clip level (warn, don't drop — user can override)
        # "high" is defined as the side predicted to have LARGER F (regardless
        # of sign(r)); the per-clip F gap should therefore be positive, and the
        # attribute gap should follow sign(r).
        for kind, pair in candidates.items():
            hi_b, lo_b = pair["high"], pair["low"]
            if hi_b["per_clip_F"] is not None and lo_b["per_clip_F"] is not None:
                if hi_b["per_clip_F"] - lo_b["per_clip_F"] <= 0:
                    print(f"  ! {kind} per-clip F gap inverted at clip level: "
                          f"hi={hi_b['per_clip_F']:.2f} lo={lo_b['per_clip_F']:.2f} "
                          f"(per-actor was hi={hi_b['per_actor_F']:.2f} > lo={lo_b['per_actor_F']:.2f}); "
                          "the actor's longest clip is atypical")
            if np.sign(hi_b["attr_value"] - lo_b["attr_value"]) != sign and sign != 0:
                print(f"  ! {kind} attr direction wrong vs sign(r): "
                      f"hi A={hi_b['attr_value']:.2f} lo A={lo_b['attr_value']:.2f} sign(r)={sign}")

        output_rows.append({
            "row_id": row_id,
            "attribute": attr,
            "attribute_short": ATTR_SHORT[attr],
            "attribute_unit": ATTR_UNIT[attr],
            "task": task,
            "feature": feat,
            "feature_recipe": build_feature_recipe(feat),
            "r": r_val, "p": p_val,
            "predicted_sign": sign,
            "n_actors_pool": int(len(pool)),
            "content_name_per_candidate": cn_used,   # populated when match_content_name=True
            "candidates": candidates,
            "chosen": None,  # user fills this in
        })

        # flat summary rows
        for kind, pair in candidates.items():
            for side, blk in pair.items():
                summary_rows.append({
                    "row_id": row_id, "attribute": ATTR_SHORT[attr], "task": task, "feature": feat,
                    "r": r_val, "candidate": kind, "side": side,
                    "actor_uid": blk["actor_uid"],
                    "attr_value": blk["attr_value"],
                    "per_actor_F": blk["per_actor_F"],
                    "per_clip_F": blk["per_clip_F"],
                    "n_frames": blk["n_frames"],
                    "g1_csv": blk["g1_csv"],
                })

    # 6. write outputs
    cache_path.write_text(json.dumps(cache, indent=1))
    print(f"\n[cache] wrote {cache_path} ({len(cache)} entries)")

    out_json = out / "pair_candidates.json"
    out_json.write_text(json.dumps({"rows": output_rows}, indent=2))
    print(f"[out] wrote {out_json} ({len(output_rows)} rows)")

    out_csv = out / "pair_candidates_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()) if summary_rows else
                           ["row_id","attribute","task","feature","r","candidate","side",
                            "actor_uid","attr_value","per_actor_F","per_clip_F","n_frames","g1_csv"])
        w.writeheader()
        for s in summary_rows: w.writerow(s)
    print(f"[out] wrote {out_csv} ({len(summary_rows)} candidates)")

    print(f"\nNext: open {out_json}, add \"chosen\": \"extreme\" or \"gap\" per row, then run build_feature_matrix.py")


if __name__ == "__main__":
    main()
