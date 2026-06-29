#!/usr/bin/env python3
"""
apply_g1_pmr.py — Anonymize G1 clips using trained PMR defense.

For each clip:
  z_m = E_M(target_clip)            ← motion content from target
  z_p = E_P(dummy_clip)             ← style from a randomly chosen dummy actor
  sanitized = D(z_m, z_p)           ← target's action in dummy's movement signature

Dummy actors are sampled from a configurable pool. Using multiple dummies
adds privacy-by-obscurity on top of the learned disentanglement.

Output CSVs mirror the original g1/csv/ layout so eval_dsgcn.py can consume them.

Also exposes G1PMRSanitizerWrapper — compatible with evaluate_defense.py's
sanitize_clip() interface so MSE can be computed on-the-fly.

Usage:
  cd Defense/g1_pmr_defense
  python apply_g1_pmr.py --checkpoint g1_pmr_ckpts/final.pt --data-root ../..
  python apply_g1_pmr.py --checkpoint g1_pmr_ckpts/final.pt --data-root ../.. --num-dummies 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_HERE.parent / "src"))   # pyskl stub lives under src/
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "Privacy-Retargeting"))

from models.pmr import PMRModel
from dataset_g1_paired import _load_clip, _normalise, G1_CHANNELS, T_WINDOW
from dsgcn_bones_seed import load_g1_cache, g1_cache_fetch, read_g1_motion

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STRIDE = 128    # overlap for stitching long clips


# ===========================================================================
# Model loader
# ===========================================================================

def load_pmr_model(ckpt_path: str) -> tuple:
    """Returns (model, cfg, global_mean, global_std)."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    cfg  = ckpt["config"]
    model = PMRModel(
        T=cfg["T"], encoded_channels=tuple(cfg["encoded_channels"]),
        use_2d=False, out_channels=cfg["out_channels"],
        num_joints=cfg["out_channels"], num_coords=1,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    ckpt_dir    = str(Path(ckpt_path).parent)
    global_mean = np.load(os.path.join(ckpt_dir, "global_mean.npy")).astype(np.float32)
    global_std  = np.load(os.path.join(ckpt_dir, "global_std.npy")).astype(np.float32)
    return model, cfg, global_mean, global_std


# ===========================================================================
# Dummy embedding pool
# ===========================================================================

@torch.no_grad()
def build_dummy_pool(
    model: PMRModel,
    dummy_rel_paths: List[str],
    data_root: str,
    global_mean: np.ndarray,
    global_std: np.ndarray,
    g1_cache_info: Optional[dict],
) -> List[torch.Tensor]:
    """
    Precompute E_P embeddings for dummy actors.
    Each dummy actor contributes one averaged embedding over all their clips.
    Returns list of (1, enc_ch[0], enc_ch[1]) tensors.
    """
    embeddings = []
    for rel in tqdm(dummy_rel_paths, desc="dummy embeddings", leave=False):
        x_raw = _load_clip(rel, data_root, g1_cache_info)
        if x_raw is None:
            continue
        # Use multiple windows, average the embedding
        clip_embs = []
        T = x_raw.shape[0]
        if T < T_WINDOW:
            x_pad = np.concatenate(
                [x_raw, np.zeros((T_WINDOW - T, G1_CHANNELS), np.float32)], axis=0
            )
            windows = [x_pad]
        else:
            starts = list(range(0, T - T_WINDOW + 1, STRIDE))
            if not starts or starts[-1] + T_WINDOW < T:
                starts.append(T - T_WINDOW)
            windows = [x_raw[s: s + T_WINDOW] for s in starts]

        for w in windows:
            w_n = _normalise(w, global_mean, global_std)
            inp  = torch.from_numpy(w_n).unsqueeze(0).to(DEVICE)
            emb  = model.privacy_encoder(inp)   # (1, enc_ch[0], enc_ch[1])
            clip_embs.append(emb)

        if clip_embs:
            # clip_embs[i]: (1, C, S).  Stack→(N,1,C,S), mean(0)→(1,C,S) ✓
            mean_emb = torch.stack(clip_embs, dim=0).mean(dim=0)  # (1, enc_ch[0], enc_ch[1])
            embeddings.append(mean_emb)

    print(f"  Dummy pool: {len(embeddings)} embeddings built")
    return embeddings


# ===========================================================================
# Sanitizer wrapper (compatible with evaluate_defense.py)
# ===========================================================================

class G1PMRSanitizerWrapper:
    """
    Wraps PMR model + precomputed dummy embeddings.
    Implements sanitize(x) → compatible with evaluate_defense.py's sanitize_clip().
    """

    def __init__(self, model: PMRModel, dummy_pool: List[torch.Tensor]):
        self.model      = model
        self.dummy_pool = dummy_pool
        self._rng       = np.random.default_rng(42)

    @torch.no_grad()
    def sanitize(self, x: torch.Tensor) -> torch.Tensor:
        """x: (1, T, 35) → sanitized (1, T, 35)."""
        z_m = self.model.motion_encoder(x)
        z_p = self.dummy_pool[int(self._rng.integers(len(self.dummy_pool)))]
        return self.model.decoder(z_m, z_p)


def sanitize_clip_pmr(
    sanitizer: G1PMRSanitizerWrapper,
    x: np.ndarray,          # (T, 35) raw unnormalised
    global_mean: np.ndarray,
    global_std: np.ndarray,
) -> np.ndarray:
    """Drop-in replacement for apply_g1_defense.sanitize_clip()."""
    T = x.shape[0]
    x_n = _normalise(x.astype(np.float32), global_mean, global_std)

    if T <= T_WINDOW:
        pad = np.zeros((T_WINDOW - T, G1_CHANNELS), np.float32)
        x_pad = np.concatenate([x_n, pad], axis=0)
        inp   = torch.from_numpy(x_pad).unsqueeze(0).to(DEVICE)
        out_n = sanitizer.sanitize(inp).squeeze(0).cpu().numpy()[:T]
    else:
        out_sum = np.zeros_like(x_n)
        count   = np.zeros(T, np.float32)
        starts  = list(range(0, T - T_WINDOW + 1, STRIDE))
        if not starts or starts[-1] + T_WINDOW < T:
            starts.append(T - T_WINDOW)
        for s in starts:
            inp = torch.from_numpy(x_n[s: s + T_WINDOW]).unsqueeze(0).to(DEVICE)
            out = sanitizer.sanitize(inp).squeeze(0).cpu().numpy()
            out_sum[s: s + T_WINDOW] += out
            count[s: s + T_WINDOW]   += 1.0
        out_n = out_sum / np.maximum(count[:, None], 1.0)

    return (out_n * (global_std + 1e-6) + global_mean).astype(np.float32)


# ===========================================================================
# Main
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply G1 PMR defense to all clips")
    p.add_argument("--checkpoint",    type=str, required=True)
    p.add_argument("--data-root",     type=str, default="../..")
    p.add_argument("--splits-dir",    type=str, default=None)
    p.add_argument("--g1-cache-dir",  type=str, default=None)
    p.add_argument("--no-g1-cache",   action="store_true")
    p.add_argument("--output-root",   type=str, default=None,
                   help="Where to write sanitized CSVs (default: <data-root>/g1_sanitized_pmr)")
    p.add_argument("--splits",        nargs="+", default=["train", "val", "test"])
    p.add_argument("--num-dummies",   type=int, default=3,
                   help="Number of dummy actors to use in the pool")
    p.add_argument("--overwrite",     action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    output_root = args.output_root or os.path.join(args.data_root, "g1_sanitized_pmr")
    splits_dir  = args.splits_dir  or os.path.join(args.data_root, "artifacts", "splits")

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading: {args.checkpoint}")
    model, cfg, global_mean, global_std = load_pmr_model(args.checkpoint)
    print(f"  epoch={cfg.get('epoch','?')}  enc_ch={cfg['encoded_channels']}")

    fast_actors  = cfg.get("fast_actors")
    fast_splits  = cfg.get("fast_splits_dir")
    if fast_actors is not None:
        fast_actors = set(fast_actors)
        print(f"  [fast] filtering to {len(fast_actors)} actors")

    # ── G1 cache ──────────────────────────────────────────────────────────
    g1_cache_info = None
    if not args.no_g1_cache:
        cache_dir = args.g1_cache_dir or os.path.join(
            args.data_root, "artifacts", "cache", "g1_motions")
        g1_cache_info = load_g1_cache(cache_dir)

    # ── Build dummy pool ──────────────────────────────────────────────────
    # Pick dummy actors from the training set (in config actor_map)
    actor_map = cfg.get("actor_map", {})
    all_train_actors = list(actor_map.keys())
    rng = np.random.default_rng(42)
    dummy_actor_ids = list(rng.choice(
        all_train_actors,
        min(args.num_dummies, len(all_train_actors)),
        replace=False
    ))
    print(f"Dummy actors: {dummy_actor_ids}")

    # Collect clip paths for dummy actors
    train_df = pd.read_csv(os.path.join(splits_dir, "train_manifest.csv"))
    dummy_paths = []
    for actor in dummy_actor_ids:
        rows = train_df[train_df["actor_uid"].astype(str) == str(actor)]
        for _, row in rows.iterrows():
            rel = str(row.get("move_g1_mujoco_path", ""))
            if rel:
                dummy_paths.append(rel)
    rng.shuffle(dummy_paths)
    dummy_paths = list(dummy_paths[:50])   # cap to 50 clips per pool build

    dummy_pool = build_dummy_pool(
        model, dummy_paths, args.data_root, global_mean, global_std, g1_cache_info
    )
    if not dummy_pool:
        print("[warn] No dummy embeddings — using zero embedding")
        enc_ch = tuple(cfg["encoded_channels"])
        dummy_pool = [torch.zeros(1, enc_ch[0], enc_ch[1], device=DEVICE)]

    sanitizer = G1PMRSanitizerWrapper(model, dummy_pool)

    # ── Process splits ────────────────────────────────────────────────────
    split_files = {"train": "train_manifest.csv",
                   "val":   "val_manifest.csv",
                   "test":  "test_manifest.csv"}
    done = skip = err = 0
    _first_err_printed = False

    for split in args.splits:
        csv_path = os.path.join(splits_dir, split_files[split])
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        df = df[df["move_g1_mujoco_path"].notna()].reset_index(drop=True)
        if fast_actors is not None:
            df = df[df["actor_uid"].isin(fast_actors)].reset_index(drop=True)
        print(f"\n[{split}] {len(df):,} clips")

        # Keep a persistent memmap handle — returning a VIEW from g1_cache_fetch
        # and discarding the memmap handle causes it to be GC'd, invalidating the view.
        _g1_memmap = None

        for _, row in tqdm(df.iterrows(), total=len(df), desc=split):
            rel      = str(row["move_g1_mujoco_path"])
            out_path = os.path.join(output_root, rel)

            if not args.overwrite and os.path.exists(out_path):
                skip += 1
                continue

            x_raw = None
            if g1_cache_info is not None:
                try:
                    x_slice, _g1_memmap = g1_cache_fetch(g1_cache_info, rel, _g1_memmap)
                    if x_slice is not None:
                        x_raw = x_slice.copy()   # copy OUT of memmap view
                except Exception:
                    x_raw = None
            if x_raw is None:
                try:
                    x_raw = read_g1_motion(os.path.join(args.data_root, rel))
                except Exception:
                    err += 1
                    continue

            if not np.isfinite(x_raw).all():
                x_raw = np.nan_to_num(x_raw, 0.0)
            if x_raw.shape[0] < 2:
                err += 1
                continue

            try:
                x_san = sanitize_clip_pmr(sanitizer, x_raw, global_mean, global_std)
            except Exception as e:
                if not _first_err_printed:
                    import traceback
                    print(f"\n[ERROR in sanitize_clip_pmr] {type(e).__name__}: {e}")
                    traceback.print_exc()
                    _first_err_printed = True
                err += 1
                continue

            # Write CSV with original column names
            try:
                orig_df   = pd.read_csv(os.path.join(args.data_root, rel))
                data_cols = [c for c in orig_df.columns if c != "Frame"]
                san_df    = pd.DataFrame(x_san, columns=data_cols)
                if "Frame" in orig_df.columns:
                    san_df.insert(0, "Frame", range(len(san_df)))
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                san_df.to_csv(out_path, index=False)
            except Exception:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                np.savetxt(out_path, x_san, delimiter=",")
            done += 1

    print(f"\nDone: {done:,}  Skipped: {skip:,}  Errors: {err:,}")
    print(f"Sanitized CSVs → {output_root}")

    # Write manifests (same relative paths, different data-root)
    san_splits_dir = os.path.join(output_root, "artifacts", "splits")
    os.makedirs(san_splits_dir, exist_ok=True)
    for split in args.splits:
        csv_path = os.path.join(splits_dir, split_files[split])
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if fast_actors is not None:
            df = df[df["actor_uid"].isin(fast_actors)].reset_index(drop=True)
        df.to_csv(os.path.join(san_splits_dir, split_files[split]), index=False)
    print(f"Manifests → {san_splits_dir}")
    print("To evaluate: python ../g1_sanitizer/evaluate_defense.py "
          f"--sanitizer-ckpt {args.checkpoint} --sanitized-root {output_root} --data-root {args.data_root}")


if __name__ == "__main__":
    main()
