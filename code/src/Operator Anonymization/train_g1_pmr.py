#!/usr/bin/env python3
"""
train_g1_pmr.py — PMR defense on G1 joint-angle trajectories.

Both E_M and E_P receive G1 clips (T=256, C=35). No proportional/uniform distinction.
Disentanglement comes from the 4-tuple cross-reconstruction + triplet + latent consistency.

Four training stages:
  1. ae_warmup     : Self-reconstruction only. Both encoders see the same clip.
  2. clf_pretrain  : Freeze encoders, train action/actor classifiers on frozen embeddings.
  3. unpaired_adv  : Full unpaired loss — action cooperative on E_M, actor adversarial on E_M,
                     actor cooperative on E_P.
  4. paired_cross  : Add cross-reconstruction, triplet, and latent consistency losses.

Usage:
  cd Defense/g1_pmr_defense
  python train_g1_pmr.py --data-root ../.. 
  python train_g1_pmr.py --data-root ../.. --resume g1_pmr_ckpts/latest.pt
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_HERE.parent / "src"))   # pyskl stub lives under src/
sys.path.insert(0, str(_HERE))
# PMR models
_PMR_ROOT = _HERE.parent / "Privacy-Retargeting"
sys.path.insert(0, str(_PMR_ROOT))

from models.pmr import PMRModel
from models.classifiers import MotionClassifier, PrivacyClassifier
from dataset_g1_paired import ActorActionIndex, G1UnpairedDataset, G1PairedDataset
from dsgcn_bones_seed import read_g1_motion, load_g1_cache, g1_cache_fetch

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G1_CHANNELS = 35
T_WINDOW    = 256


# ===========================================================================
# Norm stats
# ===========================================================================

def compute_norm_stats(
    df: pd.DataFrame, data_root: str, g1_cache_info: Optional[dict],
    max_samples: int = 5000, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    df_s = df.sample(min(max_samples, len(df)), random_state=seed)
    frames: List[np.ndarray] = []
    for _, row in tqdm(df_s.iterrows(), total=len(df_s), desc="norm stats", leave=False):
        rel = str(row.get("move_g1_mujoco_path", ""))
        if not rel:
            continue
        x = None
        if g1_cache_info is not None:
            try:
                x, _ = g1_cache_fetch(g1_cache_info, rel, None)
            except Exception:
                x = None
        if x is None:
            try:
                x = read_g1_motion(os.path.join(data_root, rel))
            except Exception:
                continue
        x = x.astype(np.float32)
        if not np.isfinite(x).all():
            continue
        x = x - x.mean(axis=0, keepdims=True)
        frames.append(x)
    if not frames:
        return np.zeros(G1_CHANNELS, np.float32), np.ones(G1_CHANNELS, np.float32)
    stacked = np.concatenate(frames, axis=0)
    stacked = stacked[np.isfinite(stacked).all(axis=1)]
    mean = stacked.mean(axis=0).astype(np.float32)
    std  = np.clip(stacked.std(axis=0), 1e-6, None).astype(np.float32)
    return mean, std


# ===========================================================================
# PMR model factory
# ===========================================================================

def make_pmr(enc_ch: Tuple[int, int]) -> PMRModel:
    return PMRModel(
        T=T_WINDOW, encoded_channels=enc_ch, use_2d=False,
        out_channels=G1_CHANNELS, num_joints=G1_CHANNELS, num_coords=1,
    ).to(DEVICE)


# ===========================================================================
# Argument parsing
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PMR defense on G1 trajectories")
    p.add_argument("--data-root",     type=str, default="../..")
    p.add_argument("--splits-dir",    type=str, default=None)
    p.add_argument("--g1-cache-dir",  type=str, default=None)
    p.add_argument("--no-g1-cache",   action="store_true")
    p.add_argument("--output-dir",    type=str, default=None,
                   help="Checkpoint directory. Defaults to g1_pmr_ckpts or "
                        "g1_pmr_ckpts_contrast depending on --pairing-strategy.")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--batch-size",    type=int, default=32)
    p.add_argument("--num-workers",   type=int, default=0)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--adv-lr",        type=float, default=1e-4)
    # Stage epochs
    p.add_argument("--epochs-ae",       type=int, default=10)
    p.add_argument("--epochs-clf",      type=int, default=10)
    p.add_argument("--epochs-unpaired", type=int, default=20)
    p.add_argument("--epochs-paired",   type=int, default=30)
    # Loss weights
    p.add_argument("--alpha-rec",     type=float, default=2.0)
    p.add_argument("--alpha-smooth",  type=float, default=1.0)
    p.add_argument("--alpha-coop",    type=float, default=1.0)
    p.add_argument("--alpha-adv",     type=float, default=0.5)
    p.add_argument("--alpha-cross",   type=float, default=0.5)
    p.add_argument("--alpha-trip",    type=float, default=1.0)
    p.add_argument("--alpha-latent",  type=float, default=5.0)
    # Model
    p.add_argument("--encoded-channels", nargs=2, type=int, default=[256, 32])
    p.add_argument("--norm-samples",     type=int, default=5000)
    p.add_argument("--max-combos",          type=int,   default=200_000,
                   help="Cap on pre-computed valid 4-tuples")
    p.add_argument("--pairing-strategy",   type=str,   default="random",
                   choices=["random", "attribute_contrast", "max_attribute_distance"],
                   help="'random': all actor pairs.  "
                        "'attribute_contrast': bottom/top percentile extreme groups.  "
                        "'max_attribute_distance': each actor paired with their "
                        "top-K most attribute-distant partners (strictest).")
    p.add_argument("--contrast-percentile", type=float, default=0.33,
                   help="Percentile threshold for attribute_contrast (default 0.33).")
    p.add_argument("--max-dist-topk",       type=int,   default=5,
                   help="Number of most-distant partners per actor for "
                        "max_attribute_distance strategy (default 5).")
    # Subset / split
    p.add_argument("--subset-actors",    type=int,   default=None)
    p.add_argument("--test-ratio",       type=float, default=0.20)
    p.add_argument("--val-actor-ratio",  type=float, default=0.10)
    p.add_argument("--val-demo-holdout", type=float, default=0.20)
    p.add_argument("--resume",           type=str,   default=None)
    return p.parse_args()


# ===========================================================================
# Training
# ===========================================================================

def train(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Resolve default output dir based on pairing strategy
    if args.output_dir is None:
        args.output_dir = {
            "attribute_contrast":    "g1_pmr_ckpts_contrast",
            "max_attribute_distance": "g1_pmr_ckpts_maxdist",
        }.get(args.pairing_strategy, "g1_pmr_ckpts")
    print(f"Output dir: {args.output_dir}  (pairing: {args.pairing_strategy})")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Manifests ─────────────────────────────────────────────────────────
    splits_dir = args.splits_dir or os.path.join(args.data_root, "artifacts", "splits")
    train_df    = pd.read_csv(os.path.join(splits_dir, "train_manifest.csv"))
    val_df      = pd.read_csv(os.path.join(splits_dir, "val_manifest.csv"))
    test_df_all = pd.read_csv(os.path.join(splits_dir, "test_manifest.csv"))
    df_seen = pd.concat([train_df, val_df], ignore_index=True)
    df_seen = df_seen[df_seen["move_g1_mujoco_path"].notna()].reset_index(drop=True)

    # ── Subset split ──────────────────────────────────────────────────────
    fast_splits_dir         = None
    fast_actors_for_config  = None

    if args.subset_actors is not None:
        n     = args.subset_actors
        n_test      = max(1, round(n * args.test_ratio))
        n_val_act   = max(1, round(n * args.val_actor_ratio))
        n_train_only = n - n_test - n_val_act

        rng = np.random.default_rng(args.seed)
        seen_pool = sorted(df_seen["actor_uid"].dropna().unique().tolist())
        n_seen    = n_train_only + n_val_act
        seen_sel  = list(rng.choice(seen_pool, min(n_seen, len(seen_pool)), replace=False).tolist())
        rng.shuffle(seen_sel)
        fast_train = sorted(seen_sel[:n_train_only])
        fast_val   = sorted(seen_sel[n_train_only:n_train_only + n_val_act])

        test_pool   = sorted(test_df_all["actor_uid"].dropna().unique().tolist())
        fast_unseen = sorted(
            rng.choice(test_pool, min(n_test, len(test_pool)), replace=False).tolist()
        )
        fast_actors_for_config = fast_train + fast_val + fast_unseen
        fast_all_seen = set(fast_train + fast_val)

        # val actors: 80% demos → training, 20% held out
        val_act_clips = pd.concat([
            train_df[train_df["actor_uid"].isin(set(fast_val))],
            val_df[val_df["actor_uid"].isin(set(fast_val))],
        ], ignore_index=True)
        val_act_clips = val_act_clips[val_act_clips["move_g1_mujoco_path"].notna()].sample(
            frac=1, random_state=args.seed)
        n_vt = int(len(val_act_clips) * (1 - args.val_demo_holdout))
        val_act_train = val_act_clips.iloc[:n_vt]
        val_act_hold  = val_act_clips.iloc[n_vt:]

        train_only_clips = pd.concat([
            train_df[train_df["actor_uid"].isin(set(fast_train))],
            val_df[val_df["actor_uid"].isin(set(fast_train))],
        ], ignore_index=True)
        train_only_clips = train_only_clips[train_only_clips["move_g1_mujoco_path"].notna()]

        df_train_pmr = pd.concat([train_only_clips, val_act_train], ignore_index=True)
        test_clips   = test_df_all[
            test_df_all["actor_uid"].isin(set(fast_unseen)) &
            test_df_all["move_g1_mujoco_path"].notna()
        ]

        fast_splits_dir = os.path.join(args.output_dir, "splits")
        os.makedirs(fast_splits_dir, exist_ok=True)
        df_train_pmr.to_csv(os.path.join(fast_splits_dir, "train_manifest.csv"), index=False)
        val_act_hold.to_csv( os.path.join(fast_splits_dir, "val_manifest.csv"),   index=False)
        test_clips.to_csv(   os.path.join(fast_splits_dir, "test_manifest.csv"),  index=False)

        df_seen = df_train_pmr
        print(f"[subset] {n_train_only} train-only + {n_val_act} val + {n_test} unseen = {n} actors")
        print(f"         training clips: {len(df_seen):,}")
    else:
        df_seen = df_seen  # all seen actors

    print(f"Training clips: {len(df_seen):,}")

    # ── G1 cache ──────────────────────────────────────────────────────────
    g1_cache_info = None
    if not args.no_g1_cache:
        cache_dir = args.g1_cache_dir or os.path.join(
            args.data_root, "artifacts", "cache", "g1_motions")
        g1_cache_info = load_g1_cache(cache_dir)
        if g1_cache_info:
            print(f"G1 cache: {len(g1_cache_info['index_map']):,} clips")

    # ── Norm stats ────────────────────────────────────────────────────────
    print("Computing norm stats...")
    global_mean, global_std = compute_norm_stats(
        df_seen, args.data_root, g1_cache_info,
        max_samples=args.norm_samples, seed=args.seed)
    np.save(os.path.join(args.output_dir, "global_mean.npy"), global_mean)
    np.save(os.path.join(args.output_dir, "global_std.npy"),  global_std)
    print(f"  std range: [{global_std.min():.4f}, {global_std.max():.4f}]")

    # ── Actor-action index ────────────────────────────────────────────────
    index = ActorActionIndex(
        df_seen,
        max_combos=args.max_combos,
        seed=args.seed,
        pairing_strategy=args.pairing_strategy,
        contrast_percentile=args.contrast_percentile,
        max_dist_topk=args.max_dist_topk,
    )
    num_actors  = len(index.actors)
    num_actions = len(index.actions)
    print(f"Actors: {num_actors}  Actions: {num_actions}  "
          f"4-tuples: {len(index.valid_combos):,}")

    # ── Datasets & loaders ────────────────────────────────────────────────
    unpaired_ds = G1UnpairedDataset(index, args.data_root, global_mean, global_std, g1_cache_info)
    paired_ds   = G1PairedDataset(  index, args.data_root, global_mean, global_std, g1_cache_info)

    unp_loader = DataLoader(unpaired_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, drop_last=True,
                            pin_memory=(DEVICE.type == "cuda"))
    pair_loader = DataLoader(paired_ds, batch_size=min(args.batch_size, 16), shuffle=True,
                             num_workers=args.num_workers, drop_last=True,
                             pin_memory=(DEVICE.type == "cuda"))
    print(f"Unpaired loader: {len(unp_loader)} batches/epoch")
    print(f"Paired loader:   {len(pair_loader)} batches/epoch")

    # ── Models ────────────────────────────────────────────────────────────
    enc_ch  = tuple(args.encoded_channels)
    model   = make_pmr(enc_ch)
    motion_clf  = MotionClassifier(num_actions, enc_ch).to(DEVICE)
    privacy_clf = PrivacyClassifier(num_actors,  enc_ch).to(DEVICE)

    opt_ae     = torch.optim.AdamW(model.parameters(), lr=args.lr)
    opt_m_clf  = torch.optim.AdamW(motion_clf.parameters(),  lr=args.adv_lr)
    opt_p_clf  = torch.optim.AdamW(privacy_clf.parameters(), lr=args.adv_lr)

    triplet_loss = nn.TripletMarginLoss(margin=1.0)
    ce_loss      = nn.CrossEntropyLoss()

    # ── Resume ────────────────────────────────────────────────────────────
    resume_stage: Optional[str] = None
    resume_ep_in_stage: int = -1
    epoch_global: int = 0

    # Auto-detect latest.pt if --resume not given
    auto_latest = os.path.join(args.output_dir, "latest.pt")
    resume_path = args.resume or (auto_latest if os.path.exists(auto_latest) else None)

    if resume_path:
        print(f"Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        if "motion_clf_state" in ckpt:
            motion_clf.load_state_dict(ckpt["motion_clf_state"])
        if "privacy_clf_state" in ckpt:
            privacy_clf.load_state_dict(ckpt["privacy_clf_state"])
        for opt, key in [(opt_ae, "opt_ae"), (opt_m_clf, "opt_m_clf"),
                         (opt_p_clf, "opt_p_clf")]:
            if f"{key}_state" in ckpt:
                opt.load_state_dict(ckpt[f"{key}_state"])
        resume_stage       = ckpt.get("stage")
        resume_ep_in_stage = ckpt.get("epoch_in_stage", -1)
        epoch_global       = ckpt.get("epoch", 0)
        print(f"  stage={resume_stage}  ep_in_stage={resume_ep_in_stage}")

    # ── Checkpoint helper ─────────────────────────────────────────────────
    def save_ckpt(path: str, stage: str, ep_in_stage: int):
        torch.save({
            "epoch": epoch_global, "stage": stage, "epoch_in_stage": ep_in_stage,
            "model_state":       model.state_dict(),
            "motion_clf_state":  motion_clf.state_dict(),
            "privacy_clf_state": privacy_clf.state_dict(),
            "opt_ae_state":      opt_ae.state_dict(),
            "opt_m_clf_state":   opt_m_clf.state_dict(),
            "opt_p_clf_state":   opt_p_clf.state_dict(),
            "config": {
                "T": T_WINDOW, "out_channels": G1_CHANNELS,
                "encoded_channels": list(enc_ch),
                "num_actors": num_actors, "num_actions": num_actions,
                "actor_map":  index.actor_map, "action_map": index.action_map,
                "fast_actors":     fast_actors_for_config,
                "fast_splits_dir": fast_splits_dir,
                "sanitizer_type":    "pmr",
                "pairing_strategy":  args.pairing_strategy,
            },
        }, path)

    # ── Loss helpers ──────────────────────────────────────────────────────
    def _rec(pred, target):
        return F.mse_loss(pred, target)

    def _smooth(x):
        return ((x[:, 1:] - x[:, :-1]) ** 2).mean()

    def _flatten(emb):
        return emb.view(emb.size(0), -1)

    def _adv_kl(clf, emb, n_classes):
        """KL(softmax(clf(emb)) ‖ uniform) — confuse classifier in emb."""
        probs   = torch.softmax(clf(emb), dim=1).clamp(1e-8)
        uniform = torch.full_like(probs, 1.0 / n_classes)
        return F.kl_div(probs.log(), uniform, reduction="batchmean")

    # ── Training stages ───────────────────────────────────────────────────
    stage_schedule = [
        ("ae_warmup",     args.epochs_ae,       unp_loader),
        ("clf_pretrain",  args.epochs_clf,       unp_loader),
        ("unpaired_adv",  args.epochs_unpaired,  unp_loader),
        ("paired_cross",  args.epochs_paired,    pair_loader),
    ]
    total_epochs = sum(n for _, n, _ in stage_schedule)

    completed: set = set()
    if resume_stage is not None:
        for sname, sn, _ in stage_schedule:
            if sname == resume_stage:
                if resume_ep_in_stage >= sn - 1:
                    completed.add(sname)
                break
            completed.add(sname)

    for stage_name, n_epochs, loader in stage_schedule:
        if stage_name in completed:
            print(f"\n[skip] {stage_name}")
            continue

        ep_start = 0
        if stage_name == resume_stage:
            ep_start = resume_ep_in_stage + 1
            if ep_start >= n_epochs:
                print(f"\n[skip] {stage_name}")
                continue

        print(f"\n{'='*60}\nStage: {stage_name}  ({n_epochs} epochs)\n{'='*60}")

        for ep in range(ep_start, n_epochs):
            epoch_global += 1
            model.train(); motion_clf.train(); privacy_clf.train()

            totals: Dict[str, float] = {}
            n_bat = 0

            # ─────────────────────────────────────────────────────────────
            if stage_name in ("ae_warmup", "clf_pretrain", "unpaired_adv"):
                # Unpaired: batch = (x, actor_idx, action_idx)
                for x, actor_idx, action_idx in loader:
                    x          = x.to(DEVICE)
                    actor_idx  = actor_idx.to(DEVICE)
                    action_idx = action_idx.to(DEVICE)

                    z_m  = model.motion_encoder(x)
                    z_p  = model.privacy_encoder(x)
                    x_hat = model.decoder(z_m, z_p)

                    l_rec    = _rec(x_hat, x)
                    l_smooth = _smooth(x_hat)
                    loss_ae  = args.alpha_rec * l_rec + args.alpha_smooth * l_smooth

                    l_coop_m = l_coop_p = l_adv = torch.zeros(1, device=DEVICE)[0]

                    if stage_name in ("clf_pretrain", "unpaired_adv"):
                        # Cooperative clf losses on detached z (prepared before AE backward)
                        l_coop_m = ce_loss(motion_clf(z_m.detach()), action_idx)
                        l_coop_p = ce_loss(privacy_clf(z_p.detach()), actor_idx)

                    if stage_name == "unpaired_adv":
                        # Freeze clf params so their grad doesn't enter loss_ae.
                        # Grad from l_adv/l_util flows through z_m → encoder only.
                        for p in privacy_clf.parameters(): p.requires_grad_(False)
                        for p in motion_clf.parameters():  p.requires_grad_(False)
                        l_adv    = _adv_kl(privacy_clf, z_m, num_actors)
                        l_util_m = ce_loss(motion_clf(z_m), action_idx)
                        for p in privacy_clf.parameters(): p.requires_grad_(True)
                        for p in motion_clf.parameters():  p.requires_grad_(True)
                        loss_ae  = (loss_ae
                                    + args.alpha_coop * l_util_m
                                    + args.alpha_adv  * l_adv)

                    # Step AE first (before any clf optimizer step)
                    opt_ae.zero_grad()
                    loss_ae.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt_ae.step()

                    # Then step classifiers on their independently computed losses
                    if stage_name in ("clf_pretrain", "unpaired_adv"):
                        opt_m_clf.zero_grad(); l_coop_m.backward(); opt_m_clf.step()
                        opt_p_clf.zero_grad(); l_coop_p.backward(); opt_p_clf.step()

                    for k, v in [("rec", l_rec), ("smooth", l_smooth),
                                 ("coop_m", l_coop_m), ("coop_p", l_coop_p),
                                 ("adv", l_adv), ("total", loss_ae)]:
                        totals[k] = totals.get(k, 0.0) + float(v.item() if hasattr(v, "item") else v)
                    n_bat += 1

            # ─────────────────────────────────────────────────────────────
            else:  # paired_cross
                for x1, x2, y1, y2, a1_idx, a2_idx, act1_idx, act2_idx in loader:
                    x1, x2, y1, y2 = x1.to(DEVICE), x2.to(DEVICE), y1.to(DEVICE), y2.to(DEVICE)
                    a1_idx  = a1_idx.to(DEVICE);  a2_idx  = a2_idx.to(DEVICE)
                    act1_idx = act1_idx.to(DEVICE); act2_idx = act2_idx.to(DEVICE)

                    # Encode all four clips
                    zm1 = model.motion_encoder(x1);  zp1 = model.privacy_encoder(x1)
                    zm2 = model.motion_encoder(x2);  zp2 = model.privacy_encoder(x2)
                    zm_y1 = model.motion_encoder(y1); zp_y1 = model.privacy_encoder(y1)
                    zm_y2 = model.motion_encoder(y2); zp_y2 = model.privacy_encoder(y2)

                    # Self-reconstructions
                    x1_hat = model.decoder(zm1,  zp1)
                    x2_hat = model.decoder(zm2,  zp2)
                    y1_hat = model.decoder(zm_y1, zp_y1)
                    y2_hat = model.decoder(zm_y2, zp_y2)

                    # Cross-reconstructions
                    # D(E_M(x1), E_P(x2)) → y2 (actor2's style, action1's motion)
                    # D(E_M(x2), E_P(x1)) → y1 (actor1's style, action2's motion)
                    y2_cross = model.decoder(zm1,  zp2)
                    y1_cross = model.decoder(zm2,  zp1)
                    # Consistency from y-side
                    x1_cross = model.decoder(zm_y1, zp_y2)  # (p1,a2)motion + (p2,a1)priv ≈ x1? No...
                    x2_cross = model.decoder(zm_y2, zp_y1)
                    # Correct cross from y-side:
                    # y1=(p1,a2), y2=(p2,a1)
                    # D(E_M(y1), E_P(y2)) → (p2,a2) = x2
                    # D(E_M(y2), E_P(y1)) → (p1,a1) = x1
                    x1_ycross = model.decoder(zm_y2, zp_y1)  # (p1,a1)=x1
                    x2_ycross = model.decoder(zm_y1, zp_y2)  # (p2,a2)=x2

                    # ── Reconstruction loss ───────────────────────────────
                    l_rec = (_rec(x1_hat, x1) + _rec(x2_hat, x2) +
                             _rec(y1_hat, y1) + _rec(y2_hat, y2)) / 4
                    l_smooth = (_smooth(x1_hat) + _smooth(x2_hat) +
                                _smooth(y1_hat) + _smooth(y2_hat)) / 4

                    # ── Cross-reconstruction loss ─────────────────────────
                    l_cross = (_rec(y2_cross, y2) + _rec(y1_cross, y1) +
                               _rec(x1_ycross, x1) + _rec(x2_ycross, x2)) / 4

                    # ── Triplet loss ──────────────────────────────────────
                    # E_M: same action → close; different action → far
                    # x1(p1,a1) and y2(p2,a1) share action a1; x2(p2,a2) is negative
                    # x2(p2,a2) and y1(p1,a2) share action a2; x1(p1,a1) is negative
                    f1, f2 = _flatten(zm1), _flatten(zm2)
                    fy1, fy2 = _flatten(zm_y1), _flatten(zm_y2)
                    l_trip_m  = (triplet_loss(f1, fy2, f2) +   # a1 anchor: x1≈y2, x2 neg
                                 triplet_loss(f2, fy1, f1)) / 2  # a2 anchor: x2≈y1, x1 neg

                    # E_P: same actor → close; different actor → far
                    # x1(p1,a1) and y1(p1,a2) share actor p1; x2(p2,a2) is negative
                    g1, g2 = _flatten(zp1), _flatten(zp2)
                    gy1, gy2 = _flatten(zp_y1), _flatten(zp_y2)
                    l_trip_p  = (triplet_loss(g1, gy1, g2) +   # p1 anchor: x1≈y1, x2 neg
                                 triplet_loss(g2, gy2, g1)) / 2  # p2 anchor: x2≈y2, x1 neg

                    l_trip = l_trip_m + l_trip_p

                    # ── Latent consistency ────────────────────────────────
                    # E_M(x1) ≈ E_M(y2): both action a1 but different actors
                    # E_M(x2) ≈ E_M(y1): both action a2 but different actors
                    l_lat_m = (F.mse_loss(zm1, zm_y2.detach()) +
                               F.mse_loss(zm2, zm_y1.detach())) / 2
                    # E_P(x1) ≈ E_P(y1): same actor p1 different actions
                    # E_P(x2) ≈ E_P(y2): same actor p2 different actions
                    l_lat_p = (F.mse_loss(zp1, zp_y1.detach()) +
                               F.mse_loss(zp2, zp_y2.detach())) / 2
                    l_latent = l_lat_m + l_lat_p

                    # ── Classifier: prepare cooperative losses on detached z ──
                    # (computed before any optimizer step to avoid version conflicts)
                    l_cm = (ce_loss(motion_clf(zm1.detach()), act1_idx) +
                            ce_loss(motion_clf(zm2.detach()), act2_idx)) / 2
                    l_cp = (ce_loss(privacy_clf(zp1.detach()), a1_idx) +
                            ce_loss(privacy_clf(zp2.detach()), a2_idx)) / 2

                    # ── Adversarial + utility losses for AE ───────────────
                    # Freeze clf params so only encoder (zm) gets grad here.
                    for p in privacy_clf.parameters(): p.requires_grad_(False)
                    for p in motion_clf.parameters():  p.requires_grad_(False)
                    l_adv  = _adv_kl(privacy_clf, zm1, num_actors)
                    l_util = (ce_loss(motion_clf(zm1), act1_idx) +
                              ce_loss(motion_clf(zm2), act2_idx)) / 2
                    for p in privacy_clf.parameters(): p.requires_grad_(True)
                    for p in motion_clf.parameters():  p.requires_grad_(True)

                    # ── Total AE loss ─────────────────────────────────────
                    loss_ae = (args.alpha_rec    * l_rec
                               + args.alpha_smooth * l_smooth
                               + args.alpha_cross  * l_cross
                               + args.alpha_trip   * l_trip
                               + args.alpha_latent * l_latent
                               + args.alpha_coop   * l_util
                               + args.alpha_adv    * l_adv)

                    # Step AE first — before any clf step modifies clf weights
                    opt_ae.zero_grad()
                    loss_ae.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt_ae.step()

                    # Then step classifiers on their independently computed losses
                    opt_m_clf.zero_grad(); l_cm.backward(); opt_m_clf.step()
                    opt_p_clf.zero_grad(); l_cp.backward(); opt_p_clf.step()

                    for k, v in [("rec", l_rec), ("smooth", l_smooth), ("cross", l_cross),
                                 ("trip", l_trip), ("latent", l_latent), ("adv", l_adv),
                                 ("total", loss_ae)]:
                        totals[k] = totals.get(k, 0.0) + float(v.item() if hasattr(v, "item") else v)
                    n_bat += 1

            avg = {k: v / max(1, n_bat) for k, v in totals.items()}
            keys = ["rec", "cross", "trip", "latent", "adv", "total"]
            line = "  ".join(f"{k}={avg.get(k,0):.4f}" for k in keys if k in avg)
            print(f"  Ep {epoch_global:03d}/{total_epochs} [{stage_name}] {line}")

            save_ckpt(os.path.join(args.output_dir, "latest.pt"), stage_name, ep)

        stage_path = os.path.join(args.output_dir, f"{stage_name}.pt")
        save_ckpt(stage_path, stage_name, n_epochs - 1)
        print(f"  Stage checkpoint → {stage_path}")

    save_ckpt(os.path.join(args.output_dir, "final.pt"), "final", -1)
    print(f"\nFinal checkpoint → {args.output_dir}/final.pt")


if __name__ == "__main__":
    args = parse_args()
    print(f"Device: {DEVICE}")
    train(args)
