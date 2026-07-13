"""
build_feature_matrix.py
-----------------------
Stage 2 of the data-driven feature-visualization pipeline.

Reads data/pair_candidates.json (produced by select_pair_candidates.py and
optionally annotated with "chosen": "extreme"|"gap" per row), copies the two
chosen G1 CSVs into data/feature_matrix/, re-runs the bones-seed correlation
script's compute_clip_features() on each copy to record the final
expected_F_high / expected_F_low scalars, and writes
data/feature_matrix_config.json — a flat row-list consumed by panel.html and
generate_smpl_matrix.py.

If a row has no "chosen" annotation, the script defaults to --default-pick
(default "extreme"). Use --force-pick to override every row.

Usage:
  python build_feature_matrix.py --bones-seed C:/Users/sihat/Downloads/bones-seed
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


def compute_clip_feature_value(csv_path: Path, compute_fn, feat_name: str) -> float | None:
    try:
        df = pd.read_csv(csv_path)
        if "Frame" in df.columns:
            df = df.drop(columns=["Frame"])
        x = df.to_numpy(dtype=np.float32)
        if x.shape[1] < 35:
            return None
        feats = compute_fn(x, fps=120.0)
        return float(feats[feat_name])
    except Exception as e:
        print(f"  ! per-clip compute failed for {csv_path.name}: {e}", file=sys.stderr)
        return None


def actor_short(blk: dict) -> str:
    g = blk.get("actor_gender", "?")
    h = blk.get("actor_height_cm", float("nan"))
    return f"{blk['actor_uid']} ({g}, {h:.0f} cm)"


def attribute_label(blk: dict, attribute: str, attribute_unit: str) -> str:
    """Chip text for the right-side label, e.g. '24 yr' or 'male'."""
    if attribute == "gender_numeric":
        return "male" if blk["actor_gender"] == "M" else "female"
    val = blk["attr_value"]
    return f"{val:.0f} {attribute_unit}".strip()


def build_row(row: dict, side: str, pair: dict, csv_url: str, expected_F: float | None) -> dict:
    blk = pair[side]
    shape = {
        "height_cm": int(round(blk["actor_height_cm"])),
        "weight_kg": int(round(blk["actor_weight_kg"])),
        "gender": GENDER_MAP_TO_SMPL.get(blk["actor_gender"], "male"),
    }
    return {
        "label": attribute_label(blk, row["attribute"], row["attribute_unit"]),
        "actor_uid": blk["actor_uid"],
        "actor_short": actor_short(blk),
        "source_bvh": blk["source_bvh"],
        "g1_csv": csv_url,                                  # URL relative to panel.html
        "shape": shape,
        "expected_F": expected_F if expected_F is not None else blk.get("per_clip_F"),
        "per_actor_F": blk["per_actor_F"],
        "n_frames": blk["n_frames"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bones-seed", required=True, type=Path)
    ap.add_argument("--candidates", type=Path,
                    default=Path(__file__).parent / "data" / "pair_candidates.json")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "data" / "feature_matrix_config.json")
    ap.add_argument("--csv-dir", type=Path,
                    default=Path(__file__).parent / "data" / "feature_matrix")
    ap.add_argument("--default-pick", choices=["extreme", "gap", "auto"], default="auto",
                    help="Used when a row has no 'chosen' annotation. "
                         "'auto' picks whichever candidate has the larger per-clip |F_high - F_low|.")
    ap.add_argument("--force-pick", choices=["extreme", "gap", "auto"], default=None,
                    help="If set, overrides every row's 'chosen' field.")
    args = ap.parse_args()

    bs = args.bones_seed.resolve()
    csv_dir = args.csv_dir.resolve(); csv_dir.mkdir(parents=True, exist_ok=True)

    # import compute_clip_features
    sys.path.insert(0, str(bs / "Correlation_V2"))
    from simple_interpretability import compute_clip_features  # type: ignore

    data = json.loads(Path(args.candidates).read_text())
    rows_in = data["rows"]
    print(f"[load] {args.candidates}  ({len(rows_in)} rows)")

    out_rows = []
    skipped = 0
    for row in rows_in:
        rid = row["row_id"]
        pick = args.force_pick or row.get("chosen") or args.default_pick
        if pick == "auto":
            # Compare per-clip F gap and choose the candidate with the bigger
            # |F_high - F_low|. Matches the user's preference: show the largest
            # observable difference between operators for that (task, feature).
            def _gap(c):
                hi, lo = c["high"].get("per_clip_F"), c["low"].get("per_clip_F")
                if hi is None or lo is None: return float("-inf")
                return abs(hi - lo)
            cand_kinds = list(row["candidates"].keys())
            pick = max(cand_kinds, key=lambda k: _gap(row["candidates"][k]))
        if pick not in row["candidates"]:
            other = next(iter(row["candidates"].keys()), None)
            if other is None:
                print(f"  ! {rid}: no candidates at all, skipping")
                skipped += 1
                continue
            print(f"  ! {rid}: '{pick}' not found, falling back to '{other}'")
            pick = other
        pair = row["candidates"][pick]

        # copy CSVs locally so panel.html can fetch them (and so the file
        # layout is self-contained — no cross-tree fetches at runtime).
        local_paths, csv_urls = {}, {}
        for side in ("high", "low"):
            src = bs / pair[side]["g1_csv"]
            dst = csv_dir / f"{rid}_{side}.csv"
            shutil.copyfile(src, dst)
            local_paths[side] = dst
            # URL relative to panel.html (which sits one level above data/).
            csv_urls[side] = "./" + dst.relative_to(Path(__file__).parent).as_posix()

        # recompute F from the COPIED CSV (the value JS must reproduce)
        f_hi = compute_clip_feature_value(local_paths["high"], compute_clip_features, row["feature"])
        f_lo = compute_clip_feature_value(local_paths["low"],  compute_clip_features, row["feature"])

        out = {
            "row_id":          rid,
            "attribute":       row["attribute"],
            "attribute_short": row["attribute_short"],
            "task":            row["task"],
            "feature":         row["feature"],
            "feature_unit":    feature_unit(row["feature"]),
            "feature_recipe":  row["feature_recipe"],
            "r":               row["r"],
            "p":               row["p"],
            "n_actors":        row["n_actors_pool"],
            "predicted_sign":  row["predicted_sign"],
            "chosen":          pick,
            "high": build_row(row, "high", pair, csv_urls["high"], f_hi),
            "low":  build_row(row, "low",  pair, csv_urls["low"],  f_lo),
        }
        out_rows.append(out)
        f_hi_s = "n/a" if f_hi is None else f"{f_hi:.2f}"
        f_lo_s = "n/a" if f_lo is None else f"{f_lo:.2f}"
        print(f"  [{rid}] pick={pick}  high={out['high']['label']} F={f_hi_s}  "
              f"low={out['low']['label']} F={f_lo_s}")

    # Derive the SMPL bin directory from csv_dir's name so a parallel "extreme"
    # build (--csv-dir data/feature_matrix_extreme) lives in its own SMPL dir
    # without colliding with the default "gap" matrix.
    smpl_dir = "./" + (csv_dir.parent / ("smpl_" + csv_dir.name))\
                     .relative_to(Path(__file__).parent).as_posix()
    config = {
        "version": 1,
        "schema": "feature_matrix_v1",
        "chrome": "minimal",      # same panel UX as final-matrix.html: G1 on top,
                                  # SMPL below the plot, activity-name label,
                                  # attribute-value chips, scrubber + play/pause
        "smpl_dir": smpl_dir,             # consumed by panel.html for SMPL bin loads
        "rows": out_rows,
        # group row_ids by attribute for the 4-tab matrix.html
        "groups": {
            short: [r["row_id"] for r in out_rows if r["attribute_short"] == short]
            for short in ("age", "weight", "height", "gender")
        },
    }
    Path(args.out).write_text(json.dumps(config, indent=2))
    print(f"\n[out] wrote {args.out}  ({len(out_rows)} rows; {skipped} skipped)")
    print(f"[out] CSVs under {csv_dir} ({len(out_rows) * 2} files)")
    print(f"\nNext: run generate_smpl_matrix.py --config {args.out} to fit SMPL meshes "
          "for the {len(out_rows)*2} chosen clips.")


if __name__ == "__main__":
    main()
