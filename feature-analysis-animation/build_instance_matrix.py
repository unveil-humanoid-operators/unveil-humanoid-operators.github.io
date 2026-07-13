"""
build_instance_matrix.py
------------------------
Adapter: fixed_feature_instances.json -> feature_matrix_config_instances.json

Each instance becomes one row in a feature_matrix_v1 config. The row's "high"
block carries the instance's clip + shape; the "low" block is left empty so
generate_smpl_matrix.py skips it (one fit per instance, not two). Output:

  data/feature_matrix_instances/<instance_id>.csv      — copied G1 CSV
  data/feature_matrix_config_instances.json            — config rows
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bones-seed", required=True, type=Path)
    ap.add_argument("--instances", type=Path,
                    default=Path(__file__).parent / "data" / "fixed_feature_instances.json")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "data" / "feature_matrix_config_instances.json")
    ap.add_argument("--csv-dir", type=Path,
                    default=Path(__file__).parent / "data" / "feature_matrix_instances")
    args = ap.parse_args()

    bs = args.bones_seed.resolve()
    csv_dir = args.csv_dir.resolve(); csv_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(bs / "Correlation_V2"))
    from simple_interpretability import compute_clip_features  # type: ignore

    data = json.loads(args.instances.read_text())
    instances = data["instances"]
    print(f"[load] {args.instances}  ({len(instances)} instances)")

    out_rows = []
    for inst in instances:
        iid = inst["instance_id"]

        # Copy the G1 CSV into the local matrix dir so the viewer can fetch it
        # via a relative URL without crossing into bones-seed.
        src = bs / inst["g1_csv"]
        dst = csv_dir / f"{iid}.csv"
        shutil.copyfile(src, dst)
        csv_url = "./" + dst.relative_to(Path(__file__).parent).as_posix()

        # Recompute per-clip F from the COPIED CSV — sanity check value.
        try:
            df = pd.read_csv(dst)
            if "Frame" in df.columns: df = df.drop(columns=["Frame"])
            x = df.to_numpy(dtype=np.float32)
            feats = compute_clip_features(x, fps=120.0)
            f_check = float(feats[inst["feature"]])
        except Exception as e:
            print(f"  ! {iid}: per-clip F check failed: {e}")
            f_check = inst["per_clip_F"]

        shape = {
            "height_cm": int(round(inst["actor_height_cm"])),
            "weight_kg": int(round(inst["actor_weight_kg"])),
            "gender":    GENDER_MAP_TO_SMPL.get(inst["actor_gender"], "male"),
        }
        block = {
            "label":       _label_for(inst),
            "actor_uid":   inst["actor_uid"],
            "actor_short": f"{inst['actor_uid']} ({inst['actor_gender']}, {int(round(inst['actor_height_cm']))} cm)",
            "source_bvh":  inst["source_bvh"],
            "g1_csv":      csv_url,
            "shape":       shape,
            "expected_F":  f_check,
            "per_clip_F":  inst["per_clip_F"],
            "n_frames":    inst["n_frames"],
        }

        out_rows.append({
            "row_id":          iid,
            "attribute":       inst["attribute"],
            "attribute_short": inst["attribute_short"],
            "side":            inst["side"],
            "task":            inst["task"],
            "feature":         inst["feature"],
            "feature_unit":    feature_unit(inst["feature"]),
            "feature_recipe":  inst["feature_recipe"],
            "r":               inst["r"],
            "p":               inst["p"],
            "predicted_sign":  inst["predicted_sign"],
            # We only need one fit per instance; the SMPL generator skips a
            # side when its source_bvh / shape is missing, so "low" stays {}.
            "high": block,
            "low":  {},
        })
        print(f"  [{iid}] side={inst['side']}  F_csv={inst['per_clip_F']:.2f}  F_check={f_check:.2f}")

    smpl_dir = "./" + (csv_dir.parent / ("smpl_" + csv_dir.name)) \
                     .relative_to(Path(__file__).parent).as_posix()
    config = {
        "version": 1,
        "schema":  "feature_matrix_v1",
        "chrome":  "minimal",
        "smpl_dir": smpl_dir,
        "rows":    out_rows,
    }
    Path(args.out).write_text(json.dumps(config, indent=2))
    print(f"\n[out] wrote {args.out}  ({len(out_rows)} rows)")
    print(f"[out] CSVs under {csv_dir}  ({len(out_rows)} files)")
    print(f"\nNext: python generate_smpl_matrix.py --config {args.out}")


def _label_for(inst):
    attr = inst["attribute"]
    if attr == "gender_numeric":
        return "male" if inst["actor_gender"] == "M" else "female"
    val = inst["attr_value"]
    return f"{int(round(val))} {inst['attribute_unit']}".strip()


if __name__ == "__main__":
    main()
