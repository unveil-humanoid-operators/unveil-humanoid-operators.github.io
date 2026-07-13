"""
generate_data.py
----------------
Builds the JSON + trajectory inputs consumed by viewer.html.

For each "content type" subcategory under
  C:/Users/sihat/Downloads/bones-seed/Correlation/cross-joint-task-results-content-type/
this script re-computes the partial correlation
  partial_r(actor_weight_kg, root_avg_speed | actor_age_yr, actor_height_cm, actor_gender)
from `actor_features_v2.csv` (per-actor aggregates) and writes the result, plus
the per-actor scatter data and the heavy/light G1 CSV pair, into ./data/.

Outputs (relative to this script):
  data/partials.json                  per-category partial r + n + sign + pearson
  data/actors.json                    per-category list of {weight, root_avg_speed, ...} for the scatter
  data/pairs.json                     per-category {heavy: {...}, light: {...}}
  data/trajectories/<category>_<heavy|light>.csv     the actual G1 motion files

Usage:
  python generate_data.py --bones-seed C:/Users/sihat/Downloads/bones-seed
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ----- Categories to include ----------------------------------------------------
# Foreground (jumping subcategories) and background (coarse 8 categories).
JUMPING_FOLDER_HINTS = ("jump",)        # any content-type folder name containing this substring
COARSE_8 = ("locomotion", "dances", "everyday", "gaming",
            "interactions", "communication", "sport", "other")

# Minimum actor counts. Coarse 8 categories need a higher bar (they have it);
# jumping subcategories are inherently smaller, so we let any subcategory with
# >= 5 actors through and only filter the bar caption count by significance.
MIN_ACTORS_FINE = 5
MIN_ACTORS_COARSE = 30


# ----- Partial correlation helper ----------------------------------------------
def partial_corr(df: pd.DataFrame,
                 x: str,
                 y: str,
                 controls: list[str]) -> tuple[float, float, float, int]:
    """Pearson partial correlation of x and y after regressing out controls.

    Returns (partial_r, pearson_r, p_partial_two_sided, n_used).
    OLS-residualizes x and y on [1, *controls], then returns Pearson r of residuals.
    """
    sub = df[[x, y, *controls]].dropna()
    n = len(sub)
    if n < len(controls) + 4:
        return float("nan"), float("nan"), float("nan"), n

    A = np.column_stack([np.ones(n), *[sub[c].to_numpy(float) for c in controls]])

    def resid(v):
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ coef

    rx = resid(sub[x].to_numpy(float))
    ry = resid(sub[y].to_numpy(float))
    partial = float(np.corrcoef(rx, ry)[0, 1])
    pearson = float(sub[[x, y]].corr().iloc[0, 1])

    # p-value via t with (n - k - 2) df, k = number of controls
    k = len(controls)
    dof = n - k - 2
    if dof <= 0 or not np.isfinite(partial) or abs(partial) >= 1:
        p = float("nan")
    else:
        t = partial * np.sqrt(dof / max(1e-12, 1 - partial ** 2))
        # two-sided survival of |t| under t-distribution; use scipy if present,
        # otherwise a reasonable normal approximation (dof here is always >= 30)
        try:
            from scipy import stats
            p = float(2 * (1 - stats.t.cdf(abs(t), dof)))
        except Exception:
            from math import erf, sqrt
            p = float(2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))))
    return partial, pearson, p, n


def gender_to_float(s: pd.Series) -> pd.Series:
    return s.map({"M": 1.0, "Male": 1.0, "m": 1.0,
                  "F": 0.0, "Female": 0.0, "f": 0.0}).astype(float)


# ----- Folder → content_type_of_movement string -------------------------------
def folder_to_content_type(folder_name: str, known_content_types: list[str]) -> str | None:
    """Match folder name like 'standing_jumping' or 'climbing_ladder_jumping' back to
    the original 'standing, jumping' / 'climbing ladder, jumping' value in the parquet.
    Normalisation rule the analysis used: ', ' -> '_'  and  ' ' -> '_' .
    """
    for ct in known_content_types:
        norm = ct.replace(", ", "_").replace(" ", "_")
        if norm == folder_name:
            return ct
    return None


# ----- Main --------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bones-seed", required=True, type=Path,
                    help="path to the bones-seed directory (read-only).")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "data",
                    help="output directory (default ./data).")
    args = ap.parse_args()

    bs = args.bones_seed.resolve()
    out = args.out.resolve()
    (out / "trajectories").mkdir(parents=True, exist_ok=True)

    fine_root = bs / "Correlation" / "cross-joint-task-results-content-type"
    coarse_root = bs / "Correlation" / "cross-joint-task-results"
    meta_path = bs / "metadata" / "seed_metadata_v003.parquet"

    print(f"[load] metadata parquet: {meta_path}")
    meta = pd.read_parquet(meta_path)
    print(f"       {len(meta):,} rows × {len(meta.columns)} cols")
    known_ct = sorted(meta["content_type_of_movement"].dropna().astype(str).unique().tolist())

    partials_out: dict[str, dict] = {}
    actors_out: dict[str, list] = {}
    pairs_out: dict[str, dict] = {}

    # ---- 1. Fine content-type subcategories (all of them; flag jumping-related) ----
    print("[scan] content-type subcategories…")
    for folder in sorted(p for p in fine_root.iterdir() if p.is_dir()):
        feats_path = folder / "actor_features_v2.csv"
        if not feats_path.exists():
            continue
        try:
            df = pd.read_csv(feats_path)
        except Exception as e:
            print(f"  skip {folder.name}: read error {e}")
            continue
        needed = {"actor_uid", "actor_weight_kg", "actor_age_yr",
                  "actor_height_cm", "actor_gender", "root_avg_speed"}
        if not needed.issubset(df.columns):
            print(f"  skip {folder.name}: missing columns")
            continue
        df = df.copy()
        df["gender_bin"] = gender_to_float(df["actor_gender"])
        df = df.dropna(subset=["actor_weight_kg", "actor_age_yr",
                               "actor_height_cm", "gender_bin", "root_avg_speed"])
        n_actors = len(df)
        if n_actors < MIN_ACTORS_FINE:
            continue

        partial, pearson, p, n = partial_corr(
            df, x="actor_weight_kg", y="root_avg_speed",
            controls=["actor_age_yr", "actor_height_cm", "gender_bin"],
        )
        is_jump = any(h in folder.name.lower() for h in JUMPING_FOLDER_HINTS)
        partials_out[folder.name] = {
            "category": folder.name,
            "kind": "jumping" if is_jump else "fine",
            "n": int(n_actors),
            "pearson": pearson,
            "partial": partial,
            "p_partial": p,
            "sign_partial": int(np.sign(partial)) if np.isfinite(partial) else 0,
            "content_type": folder_to_content_type(folder.name, known_ct),
        }
        actors_out[folder.name] = [
            {
                "uid": str(row.actor_uid),
                "weight": float(row.actor_weight_kg),
                "root_avg_speed": float(row.root_avg_speed),
                "age": float(row.actor_age_yr),
                "height": float(row.actor_height_cm),
                "gender": str(row.actor_gender),
            }
            for row in df.itertuples()
        ]

    # ---- 2. Coarse 8 categories (background bar group) ----
    print("[scan] coarse 8 categories…")
    if coarse_root.exists():
        for folder in sorted(p for p in coarse_root.iterdir() if p.is_dir()):
            if folder.name not in COARSE_8:
                continue
            feats_path = folder / "actor_features_v2.csv"
            if not feats_path.exists():
                continue
            df = pd.read_csv(feats_path)
            needed = {"actor_weight_kg", "actor_age_yr",
                      "actor_height_cm", "actor_gender", "root_avg_speed"}
            if not needed.issubset(df.columns):
                continue
            df = df.copy()
            df["gender_bin"] = gender_to_float(df["actor_gender"])
            df = df.dropna(subset=["actor_weight_kg", "actor_age_yr",
                                   "actor_height_cm", "gender_bin", "root_avg_speed"])
            n_actors = len(df)
            if n_actors < MIN_ACTORS_COARSE:
                continue
            partial, pearson, p, n = partial_corr(
                df, x="actor_weight_kg", y="root_avg_speed",
                controls=["actor_age_yr", "actor_height_cm", "gender_bin"],
            )
            key = f"coarse:{folder.name}"
            partials_out[key] = {
                "category": folder.name,
                "kind": "coarse",
                "n": int(n_actors),
                "pearson": pearson,
                "partial": partial,
                "p_partial": p,
                "sign_partial": int(np.sign(partial)) if np.isfinite(partial) else 0,
                "content_type": None,
            }

    # ---- 3. Heavy/light pair selection per fine category --------------------
    # We need a G1 CSV per chosen actor in that category. Strategy: query the
    # parquet for rows where content_type_of_movement matches AND actor_uid
    # matches AND the move_g1_mujoco_path file exists on disk. Pick that actor's
    # longest such clip (longest = most cinematic, more motion to watch).
    print("[pair] selecting heavy/light operators per category…")

    # Pre-index parquet by content type for speed
    meta_ct = {ct: meta[meta["content_type_of_movement"] == ct]
               for ct in known_ct}

    def g1_csv_path(rel: str) -> Path | None:
        if not isinstance(rel, str):
            return None
        p = bs / rel
        if p.exists():
            return p
        return None

    def pick_clip_for_actor(ct: str, uid: str) -> tuple[Path, int] | None:
        """Return (path, n_frames) for the longest available G1 CSV in this content
        type performed by this actor. Excludes mirror (_M.csv) takes to avoid
        playing essentially-identical motion for both members of the pair.
        """
        rows = meta_ct.get(ct)
        if rows is None or len(rows) == 0:
            return None
        sub = rows[rows["actor_uid"] == uid].copy()
        if len(sub) == 0:
            return None
        best = None
        for r in sub.itertuples():
            rel = getattr(r, "move_g1_mujoco_path", None)
            if not isinstance(rel, str):
                continue
            if rel.endswith("_M.csv"):
                continue
            p = g1_csv_path(rel)
            if p is None:
                continue
            # use parquet's duration_frames if present, else line-count of csv
            n_frames = getattr(r, "move_duration_frames", None)
            try:
                n_frames = int(n_frames) if n_frames is not None else None
            except Exception:
                n_frames = None
            if n_frames is None:
                try:
                    with open(p, "rb") as f:
                        n_frames = sum(1 for _ in f) - 1
                except Exception:
                    continue
            if n_frames < 60:
                continue
            if best is None or n_frames > best[1]:
                best = (p, n_frames)
        return best

    for cat_key, info in list(partials_out.items()):
        if info.get("kind") not in ("jumping", "fine"):
            continue
        cat = info["category"]
        ct = info.get("content_type")
        if ct is None:
            continue
        actors = actors_out.get(cat, [])
        if len(actors) < 10:
            continue

        # Pick heavy (top decile) and light (bottom decile) by weight,
        # then within each decile pick the actor closest to the decile-mean
        # root_avg_speed (avoids cherry-picking an outlier).
        weights = np.array([a["weight"] for a in actors])
        speeds  = np.array([a["root_avg_speed"] for a in actors])
        hi = np.quantile(weights, 0.90)
        lo = np.quantile(weights, 0.10)
        hi_idx = np.where(weights >= hi)[0]
        lo_idx = np.where(weights <= lo)[0]
        if len(hi_idx) == 0 or len(lo_idx) == 0:
            continue
        hi_mean_spd = float(np.mean(speeds[hi_idx]))
        lo_mean_spd = float(np.mean(speeds[lo_idx]))

        def best_in(idx_arr, target_spd):
            idx_arr = sorted(idx_arr, key=lambda i: abs(speeds[i] - target_spd))
            for i in idx_arr:
                clip = pick_clip_for_actor(ct, actors[i]["uid"])
                if clip is not None:
                    return i, clip
            return None

        h = best_in(hi_idx, hi_mean_spd)
        l = best_in(lo_idx, lo_mean_spd)
        if h is None or l is None:
            print(f"  pair {cat}: could not find clip files for one side; skipping")
            continue

        h_i, (h_path, h_nf) = h
        l_i, (l_path, l_nf) = l

        # Copy CSVs to trajectories/<cat>_<heavy|light>.csv
        dest_h = out / "trajectories" / f"{cat}_heavy.csv"
        dest_l = out / "trajectories" / f"{cat}_light.csv"
        shutil.copyfile(h_path, dest_h)
        shutil.copyfile(l_path, dest_l)

        pairs_out[cat] = {
            "heavy": {
                "uid": actors[h_i]["uid"],
                "weight": actors[h_i]["weight"],
                "age": actors[h_i]["age"],
                "height": actors[h_i]["height"],
                "gender": actors[h_i]["gender"],
                "root_avg_speed": actors[h_i]["root_avg_speed"],
                "csv": f"data/trajectories/{cat}_heavy.csv",
                "source_csv": str(h_path.relative_to(bs)).replace("\\", "/"),
                "n_frames": int(h_nf),
            },
            "light": {
                "uid": actors[l_i]["uid"],
                "weight": actors[l_i]["weight"],
                "age": actors[l_i]["age"],
                "height": actors[l_i]["height"],
                "gender": actors[l_i]["gender"],
                "root_avg_speed": actors[l_i]["root_avg_speed"],
                "csv": f"data/trajectories/{cat}_light.csv",
                "source_csv": str(l_path.relative_to(bs)).replace("\\", "/"),
                "n_frames": int(l_nf),
            },
        }

    # ---- 4. Write outputs ----------------------------------------------------
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj

    (out / "partials.json").write_text(
        json.dumps(_sanitize(partials_out), indent=2, allow_nan=False))
    (out / "actors.json").write_text(
        json.dumps(_sanitize(actors_out), indent=2, allow_nan=False))
    (out / "pairs.json").write_text(
        json.dumps(_sanitize(pairs_out), indent=2, allow_nan=False))

    # ---- 5. Print summary ---------------------------------------------------
    fine = {k: v for k, v in partials_out.items() if v["kind"] == "jumping"}
    coarse = {k: v for k, v in partials_out.items() if v["kind"] == "coarse"}
    other_fine = {k: v for k, v in partials_out.items() if v["kind"] == "fine"}

    def sign_count(d):
        finite = [v for v in d.values() if np.isfinite(v["partial"])]
        neg = sum(1 for v in finite if v["partial"] < 0)
        return neg, len(finite)

    n_jp_neg, n_jp_tot = sign_count(fine)
    n_co_neg, n_co_tot = sign_count(coarse)
    print()
    print(f"=== summary ===")
    print(f"jumping subcategories with stats: {n_jp_tot}")
    print(f"  negative-partial sign retention: {n_jp_neg}/{n_jp_tot}")
    for k, v in sorted(fine.items(), key=lambda kv: kv[1]["partial"]):
        print(f"    {k:35s}  n={v['n']:4d}  partial={v['partial']:+.3f}  pearson={v['pearson']:+.3f}")
    print(f"coarse categories with stats: {n_co_tot}")
    print(f"  negative-partial sign retention: {n_co_neg}/{n_co_tot}")
    for k, v in sorted(coarse.items(), key=lambda kv: kv[1]["partial"]):
        print(f"    {k:35s}  n={v['n']:4d}  partial={v['partial']:+.3f}  pearson={v['pearson']:+.3f}")
    print(f"other fine categories computed (not displayed by default): {len(other_fine)}")
    print(f"pairs picked: {len(pairs_out)}")
    print(f"trajectory CSVs copied to: {out / 'trajectories'}")


if __name__ == "__main__":
    main()
