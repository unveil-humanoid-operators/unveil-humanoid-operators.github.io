#!/usr/bin/env python3
"""
non_adaptive_eval.py — Non-adaptive privacy attack evaluation.

Uses the PRE-TRAINED ProtoGCN checkpoints (trained on original G1 data)
to infer attributes from SANITIZED G1 data.

The attacker does NOT know about the defense and does NOT re-train.
This is the realistic threat model for a deployed defense system.

Datasets evaluated:
  - Original G1            (no defense, reads existing result JSONs)
  - g1_sanitized_100       (GRL single-encoder defense, 100-actor subset)
  - g1_sanitized_pmr       (PMR random pairing,         100-actor subset)
  - g1_sanitized_pmr_contrast (PMR contrast pairing,    100-actor subset)

Results saved to: Defense/non_adaptive_eval_results.json

Usage:
  cd <bones-seed-root>
  python Defense/non_adaptive_eval.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))   # pyskl stub lives under src/

# pyskl / ProtoGCN stubs must be loaded before protogcn_bones_seed
import pyskl             # noqa: F401
import pyskl.utils       # noqa: F401
import pyskl.models      # noqa: F401
import pyskl.models.gcns           # noqa: F401
import pyskl.models.gcns.utils     # noqa: F401

from protogcn_bones_seed import (
    BonesSeedDataset,
    ProtoGCNBonesSeed,
    collect_embeddings,
    compute_global_norm,
    collate_padded,
    load_g1_cache,
    deconfound_embeddings,
    get_format_path_col,
)
from linkage_eval import evaluate_linkage

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FMT    = "g1"
ALL_TASKS        = ["reid", "linkage", "gender", "age", "height", "weight"]
REGRESSION_TASKS = {"age", "height", "weight"}

PROTOGCN_CKPT_BASE = str(_PROJECT_ROOT / "artifacts" / "models" / "protogcn" / "actor_holdout_split_g1")
ORIG_SPLITS_DIR    = str(_PROJECT_ROOT / "artifacts" / "splits")
ORIG_DATA_ROOT     = str(_PROJECT_ROOT)

DATASETS = {
    "Original G1 (no defense)": {
        "data_root":  ORIG_DATA_ROOT,
        "splits_dir": ORIG_SPLITS_DIR,
        "result_json_dir": str(_PROJECT_ROOT / "artifacts" / "models" / "protogcn" / "actor_holdout_split_g1"),
        "precomputed": True,   # read from existing JSON, skip inference
    },
    "GRL defense": {
        "data_root":  str(_PROJECT_ROOT / "g1_sanitized_100"),
        "splits_dir": str(_PROJECT_ROOT / "g1_sanitized_100" / "artifacts" / "splits"),
        "precomputed": False,
    },
    "PMR (random pairs)": {
        "data_root":  str(_PROJECT_ROOT / "g1_sanitized_pmr"),
        "splits_dir": str(_PROJECT_ROOT / "g1_sanitized_pmr" / "artifacts" / "splits"),
        "precomputed": False,
    },
    "PMR (contrast pairs)": {
        "data_root":  str(_PROJECT_ROOT / "g1_sanitized_pmr_contrast"),
        "splits_dir": str(_PROJECT_ROOT / "g1_sanitized_pmr_contrast" / "artifacts" / "splits"),
        "precomputed": False,
    },
}


# ===========================================================================
# Model loading
# ===========================================================================

def load_protogcn(task: str) -> Tuple[ProtoGCNBonesSeed, dict]:
    ckpt_path = os.path.join(PROTOGCN_CKPT_BASE, task, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg  = ckpt["config"]
    model = ProtoGCNBonesSeed(
        fmt=FMT,
        num_classes=cfg["num_classes"],
        emb_dim=cfg.get("emb_dim", 256),
        base_channels=cfg.get("base_channels", 96),
        num_prototype=cfg.get("num_prototype", 100),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


# ===========================================================================
# Shared preprocessing
# ===========================================================================

_norm_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

def get_original_norm_stats(g1_cache_info: Optional[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (or return cached) global norm stats from ORIGINAL training data.
    The non-adaptive attacker always uses their original training stats."""
    global _norm_cache
    if _norm_cache is not None:
        return _norm_cache
    print("  Computing global norm stats from original G1 training data...")
    train_df = pd.read_csv(os.path.join(ORIG_SPLITS_DIR, "train_manifest.csv"))
    val_df   = pd.read_csv(os.path.join(ORIG_SPLITS_DIR, "val_manifest.csv"))
    combined = pd.concat([train_df, val_df], ignore_index=True)
    mean, std = compute_global_norm(
        combined, ORIG_DATA_ROOT, FMT,
        channel_indices=None, downsample_factor=4,
        max_samples=5000, seed=42, g1_cache_info=g1_cache_info,
    )
    _norm_cache = (mean, std)
    return mean, std


def make_dataset(
    df: pd.DataFrame,
    data_root: str,
    label_col: str,
    label_map: Optional[dict],
    is_regression: bool,
    task_map: dict,
    global_mean: np.ndarray,
    global_std: np.ndarray,
    g1_cache_info: Optional[dict],
) -> BonesSeedDataset:
    return BonesSeedDataset(
        df=df,
        data_root=data_root,
        fmt=FMT,
        label_col=label_col,
        label_map=label_map,
        is_regression=is_regression,
        task_col="package",
        task_map=task_map,
        channel_indices=None,
        downsample_factor=4,
        max_seq_len=256,
        min_seq_len=16,
        global_mean=global_mean,
        global_std=global_std,
        train=False,
        seed=42,
        g1_cache_info=None,  # sanitized data has no cache; always read CSVs
    )


# ===========================================================================
# Metric helpers
# ===========================================================================

def reid_rank1(Z: np.ndarray, y: np.ndarray) -> float:
    """Closed-set centroid Rank@1."""
    classes = np.unique(y)
    centroids = np.stack([Z[y == c].mean(0) for c in classes])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    sims = Z @ centroids.T
    pred = classes[np.argmax(sims, axis=1)]
    return float(np.mean(pred == y))


@torch.no_grad()
def collect_regression_preds(
    model: ProtoGCNBonesSeed,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    preds, trues = [], []
    for xb, lengths, yb, _ in loader:
        xb = xb.to(DEVICE)
        logits, _, _ = model(xb)   # ProtoGCN returns (logits, z, reconstructed_graph)
        pred = logits.squeeze(1)
        mask = torch.isfinite(pred) & torch.isfinite(yb.to(DEVICE))
        if mask.any():
            preds.append(pred[mask].cpu().numpy())
            trues.append(yb.to(DEVICE)[mask].cpu().numpy())
    if not preds:
        return np.array([]), np.array([])
    return np.concatenate(preds), np.concatenate(trues)


@torch.no_grad()
def collect_gender_preds(
    model: ProtoGCNBonesSeed,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    preds, trues = [], []
    for xb, lengths, yb, _ in loader:
        xb = xb.to(DEVICE)
        logits, _, _ = model(xb)   # ProtoGCN returns (logits, z, reconstructed_graph)
        preds.extend(logits.argmax(1).cpu().tolist())
        trues.extend(yb.tolist())
    return np.array(preds), np.array(trues)


# ===========================================================================
# Evaluation per dataset × task
# ===========================================================================

def eval_dataset(name: str, cfg: dict, active_tasks: List[str]) -> dict:
    """Run inference with pre-trained ProtoGCN on one sanitized dataset."""
    data_root  = cfg["data_root"]
    splits_dir = cfg["splits_dir"]

    if not os.path.isdir(data_root):
        return {"error": f"data_root not found: {data_root}"}
    if not os.path.isdir(splits_dir):
        return {"error": f"splits_dir not found: {splits_dir}"}

    # Load manifests
    train_df = pd.read_csv(os.path.join(splits_dir, "train_manifest.csv"))
    val_df   = pd.read_csv(os.path.join(splits_dir, "val_manifest.csv"))
    try:
        test_df = pd.read_csv(os.path.join(splits_dir, "test_manifest.csv"))
    except FileNotFoundError:
        test_df = pd.DataFrame()

    combined = pd.concat([train_df, val_df], ignore_index=True)
    task_vals = sorted(combined["package"].dropna().unique().tolist())
    task_map  = {t: i for i, t in enumerate(task_vals)}
    num_tasks = len(task_map)

    # Original norm stats (attacker doesn't re-compute)
    global_mean, global_std = get_original_norm_stats(g1_cache_info=None)

    results = {}

    for task in active_tasks:
        print(f"    [{task}]", end=" ", flush=True)

        if task == "linkage":
            # ── Linkage reuses the reid checkpoint + val embeddings ──────────
            # Pairwise same/different-actor similarity on val (seen-unseen-demos)
            label_col = "actor_uid"
            all_actors = sorted(combined[label_col].dropna().unique().tolist())
            label_map  = {a: i for i, a in enumerate(all_actors)}
            val_valid  = val_df[val_df[label_col].isin(label_map)].reset_index(drop=True)
            if len(val_valid) == 0:
                results[task] = {"error": "no val data"}
                print("skip"); continue
            reid_model, _ = load_protogcn("reid")
            ds_val  = make_dataset(val_valid, data_root, label_col, label_map,
                                   False, task_map, global_mean, global_std, None)
            ldr_val = DataLoader(ds_val, batch_size=32, shuffle=False,
                                 collate_fn=collate_padded, num_workers=0)
            Z_l, y_l, t_l = collect_embeddings(reid_model, ldr_val)
            Z_dc_l, _ = deconfound_embeddings(Z_l, Z_l, t_l, t_l, num_tasks, "residual")
            lnk = evaluate_linkage(Z_dc_l, y_l, max_pos_per_actor=50,
                                   max_total_pairs=100_000, seed=42)
            print(f"acc={lnk.get('accuracy', float('nan')):.4f}  "
                  f"auc={lnk.get('auc', lnk.get('roc_auc', float('nan'))):.4f}  "
                  f"eer={lnk.get('eer', float('nan')):.4f}")
            results[task] = lnk
            continue

        model, model_cfg = load_protogcn(task)
        num_classes = model_cfg["num_classes"]

        if task == "reid":
            # ── Closed-set: val actors (seen actors, unseen demos) ───────────
            label_col = "actor_uid"
            all_actors = sorted(combined[label_col].dropna().unique().tolist())
            label_map  = {a: i for i, a in enumerate(all_actors)}

            val_valid = val_df[val_df[label_col].isin(label_map)].reset_index(drop=True)
            if len(val_valid) == 0:
                results[task] = {"error": "no val data"}
                print("skip")
                continue

            ds_val = make_dataset(val_valid, data_root, label_col, label_map,
                                  False, task_map, global_mean, global_std, None)
            ldr_val = DataLoader(ds_val, batch_size=32, shuffle=False,
                                 collate_fn=collate_padded, num_workers=0)
            Z, y, t = collect_embeddings(model, ldr_val)
            Z_dc, _ = deconfound_embeddings(Z, Z, t, t, num_tasks, "residual")

            r1_seen_unseen = reid_rank1(Z_dc, y)

            # ── Closed-set reid within unseen test actors ────────────────────
            # Correct evaluation: test actors are unknown identities.
            # We do closed-set matching WITHIN the test set (gallery=probe=test),
            # checking whether two clips from the same unseen actor are matched.
            r1_unseen = float("nan")
            if len(test_df) > 0:
                test_valid = test_df[test_df[label_col].notna()].reset_index(drop=True)
                test_actors = sorted(test_valid[label_col].dropna().unique().tolist())
                # Need ≥2 actors with ≥2 clips each for meaningful reid
                if len(test_actors) >= 2:
                    lm_test = {a: i for i, a in enumerate(test_actors)}
                    ds_test = make_dataset(test_valid, data_root, label_col, lm_test,
                                          False, task_map, global_mean, global_std, None)
                    ldr_test = DataLoader(ds_test, batch_size=32, shuffle=False,
                                         collate_fn=collate_padded, num_workers=0)
                    Z_tst, y_tst, t_tst = collect_embeddings(model, ldr_test)
                    if len(Z_tst) > 0:
                        Z_tst_dc, _ = deconfound_embeddings(
                            Z_tst, Z_tst, t_tst, t_tst, num_tasks, "residual")
                        r1_unseen = reid_rank1(Z_tst_dc, y_tst)

            print(f"seen-unseen R@1={r1_seen_unseen:.4f}  unseen R@1={r1_unseen:.4f}")
            results[task] = {
                "seen_actors_unseen_demos_rank1": r1_seen_unseen,
                "unseen_actors_rank1": r1_unseen,
                "n_seen_unseen": len(val_valid),
                "n_unseen": len(test_df),
            }

        elif task == "gender":
            label_col = "actor_gender"
            all_labels = sorted(combined[label_col].dropna().unique().tolist())
            label_map  = {g: i for i, g in enumerate(all_labels)}

            # Seen actors, unseen demos (val)
            val_valid = val_df[val_df[label_col].isin(label_map)].reset_index(drop=True)
            acc_seen_unseen = float("nan")
            if len(val_valid) > 0:
                ds = make_dataset(val_valid, data_root, label_col, label_map,
                                  False, task_map, global_mean, global_std, None)
                ldr = DataLoader(ds, batch_size=32, shuffle=False,
                                 collate_fn=collate_padded, num_workers=0)
                preds, trues = collect_gender_preds(model, ldr)
                acc_seen_unseen = float(np.mean(preds == trues)) if len(preds) > 0 else float("nan")

            # Unseen actors (test)
            acc_unseen = float("nan")
            if len(test_df) > 0:
                test_valid = test_df[test_df[label_col].isin(label_map)].reset_index(drop=True)
                if len(test_valid) > 0:
                    ds = make_dataset(test_valid, data_root, label_col, label_map,
                                      False, task_map, global_mean, global_std, None)
                    ldr = DataLoader(ds, batch_size=32, shuffle=False,
                                     collate_fn=collate_padded, num_workers=0)
                    preds, trues = collect_gender_preds(model, ldr)
                    acc_unseen = float(np.mean(preds == trues)) if len(preds) > 0 else float("nan")

            print(f"seen-unseen acc={acc_seen_unseen:.4f}  unseen acc={acc_unseen:.4f}")
            results[task] = {
                "seen_actors_unseen_demos_accuracy": acc_seen_unseen,
                "unseen_actors_accuracy": acc_unseen,
            }

        else:  # age, height, weight
            label_col_map = {"age": "actor_age_yr", "height": "actor_height_cm",
                             "weight": "actor_weight_kg"}
            label_col = label_col_map[task]

            mae_seen_unseen = mae_unseen = float("nan")

            val_valid = val_df[val_df[label_col].notna()].reset_index(drop=True)
            if len(val_valid) > 0:
                ds = make_dataset(val_valid, data_root, label_col, None,
                                  True, task_map, global_mean, global_std, None)
                ldr = DataLoader(ds, batch_size=32, shuffle=False,
                                 collate_fn=collate_padded, num_workers=0)
                preds, trues = collect_regression_preds(model, ldr)
                mae_seen_unseen = float(np.mean(np.abs(preds - trues))) if len(preds) > 0 else float("nan")

            if len(test_df) > 0:
                test_valid = test_df[test_df[label_col].notna()].reset_index(drop=True)
                if len(test_valid) > 0:
                    ds = make_dataset(test_valid, data_root, label_col, None,
                                      True, task_map, global_mean, global_std, None)
                    ldr = DataLoader(ds, batch_size=32, shuffle=False,
                                     collate_fn=collate_padded, num_workers=0)
                    preds, trues = collect_regression_preds(model, ldr)
                    mae_unseen = float(np.mean(np.abs(preds - trues))) if len(preds) > 0 else float("nan")

            print(f"seen-unseen MAE={mae_seen_unseen:.3f}  unseen MAE={mae_unseen:.3f}")
            results[task] = {
                "seen_actors_unseen_demos_mae": mae_seen_unseen,
                "unseen_actors_mae": mae_unseen,
            }

    return results


# ===========================================================================
# Read baseline from existing JSONs
# ===========================================================================

def read_baseline_results() -> dict:
    base = PROTOGCN_CKPT_BASE
    r = {}
    for task in ALL_TASKS:
        path = os.path.join(base, task, f"final_metrics_g1_{task}.json")
        if os.path.exists(path):
            with open(path) as f:
                r[task] = json.load(f)
        else:
            r[task] = {}
    return r


# ===========================================================================
# Print comparison table
# ===========================================================================

def fmt(v, digits=4):
    if v is None or (isinstance(v, float) and (v != v)):  # NaN check
        return " N/A  "
    return f"{float(v):.{digits}f}"


def print_table(all_results: dict):
    datasets = list(all_results.keys())
    W = 28

    def header(title):
        print(f"\n{'─'*80}")
        print(f"  {title}")
        print(f"{'─'*80}")
        print(f"  {'Dataset':<{W}}  {'seen-unseen':>12}  {'unseen-actors':>13}")
        print(f"  {'─'*W}  {'─'*12}  {'─'*13}")

    shown = set(all_results[next(iter(all_results))].keys()) if all_results else set(ALL_TASKS)

    if "reid" in shown:
        header("RE-ID  Rank@1  (lower = better defense)")
        for ds, results in all_results.items():
            r = results.get("reid", {})
            su = fmt(r.get("seen_actors_unseen_demos_rank1", r.get("seen_actors_unseen_top1")))
            ua = fmt(r.get("unseen_actors_rank1"))
            print(f"  {ds:<{W}}  {su:>12}  {ua:>13}")

    if "linkage" in shown:
        header("LINKAGE  Accuracy / AUC / EER  (lower acc/AUC, higher EER = better defense)")
        print(f"  {'Dataset':<{W}}  {'accuracy':>10}  {'auc':>7}  {'eer':>7}")
        print(f"  {'─'*W}  {'─'*10}  {'─'*7}  {'─'*7}")
        for ds, results in all_results.items():
            r = results.get("linkage", {})
            acc = fmt(r.get("accuracy"))
            auc = fmt(r.get("auc", r.get("roc_auc")))
            eer = fmt(r.get("eer"))
            print(f"  {ds:<{W}}  {acc:>10}  {auc:>7}  {eer:>7}")

    if "gender" in shown:
        header("GENDER  Accuracy  (chance = 0.50, lower = better defense)")
        for ds, results in all_results.items():
            r = results.get("gender", {})
            su = fmt(r.get("seen_actors_unseen_demos_accuracy", r.get("seen_actors_unseen_accuracy")))
            ua = fmt(r.get("unseen_actors_accuracy"))
            print(f"  {ds:<{W}}  {su:>12}  {ua:>13}")

    for task, unit in [("age", "yr"), ("height", "cm"), ("weight", "kg")]:
        if task in shown:
            header(f"{task.upper()}  MAE ({unit})  (higher = better defense)")
            for ds, results in all_results.items():
                r = results.get(task, {})
                su = fmt(r.get("seen_actors_unseen_demos_mae", r.get("seen_actors_unseen_mae")), 2)
                ua = fmt(r.get("unseen_actors_mae"), 2)
                print(f"  {ds:<{W}}  {su:>12}  {ua:>13}")

    print(f"\n{'─'*80}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description="Non-adaptive ProtoGCN evaluation on sanitized G1 data")
    p.add_argument("--task", nargs="+", default=ALL_TASKS,
                   choices=ALL_TASKS,
                   help="Tasks to evaluate (default: all). "
                        "Example: --task linkage   or   --task reid linkage gender")
    args = p.parse_args()
    active_tasks = args.task

    print("=" * 70)
    print("NON-ADAPTIVE ATTACK EVALUATION")
    print(f"Tasks:  {active_tasks}")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # Load existing results JSON to merge with, so --task linkage doesn't
    # wipe out previously computed reid/gender/etc. results.
    out_path = str(_HERE / "non_adaptive_eval_results.json")
    all_results: dict = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)
        print(f"  Loaded existing results from {out_path} (will merge new tasks)")

    for ds_name, ds_cfg in DATASETS.items():
        print(f"\n[{ds_name}]")

        if ds_cfg.get("precomputed"):
            if ds_name not in all_results:
                print("  Reading from existing result JSONs...")
                all_results[ds_name] = read_baseline_results()
            else:
                # Merge any missing tasks from baseline JSONs
                baseline = read_baseline_results()
                for t in active_tasks:
                    if t not in all_results[ds_name] and t in baseline:
                        all_results[ds_name][t] = baseline[t]
                        print(f"  Merged baseline task: {t}")
        else:
            if not os.path.isdir(ds_cfg["data_root"]):
                print(f"  SKIP — data_root not found: {ds_cfg['data_root']}")
                all_results.setdefault(ds_name, {"error": "data_root not found"})
                continue
            if not os.path.isdir(ds_cfg["splits_dir"]):
                print(f"  SKIP — splits_dir not found: {ds_cfg['splits_dir']}")
                all_results.setdefault(ds_name, {"error": "splits_dir not found"})
                continue
            new = eval_dataset(ds_name, ds_cfg, active_tasks)
            # Merge into existing results (preserve previously computed tasks)
            existing = all_results.get(ds_name, {})
            existing.update(new)
            all_results[ds_name] = existing

    print_table(all_results)

    out_path = str(_HERE / "non_adaptive_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
