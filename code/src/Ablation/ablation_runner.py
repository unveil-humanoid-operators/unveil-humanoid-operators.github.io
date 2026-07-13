#!/usr/bin/env python3
"""
UNVEIL Ablation Experiment Runner (2026)
================================================

Runs 4 ablation variants × {gender, reid} tasks on a 100-actor subsample of
BONES-SEED, then prints a comparison table. Each variant disables a different
UNVEIL stage: the kinematic input streams, the hierarchical spatial graph, or
the temporal aggregation.

Usage:
    python Ablation/ablation_runner.py --task all --ablation all \
        --data-root . --splits-dir artifacts/splits

Re-run a single (task, ablation) pair without touching others:
    python Ablation/ablation_runner.py --task gender --ablation pos_only_v2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Path setup — make the release root importable so we can reuse its modules
# ---------------------------------------------------------------------------
_ABLATION_DIR = Path(__file__).resolve().parent
_BONES_ROOT = _ABLATION_DIR.parent

# src/ holds unveil.py; the ablation dir itself provides the config modules
for _p in [str(_BONES_ROOT), str(_BONES_ROOT / "src"), str(_ABLATION_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import everything we need from the reference implementation
from unveil import (  # noqa: E402
    # Model components
    UnveilBackbone,
    _sym_normalize,
    DOF_PER_JOINT,
    # Losses
    SupConLoss,
    SignatureContrastiveLoss,
    # Dataset / data utils
    BonesSeedDataset,
    collate_padded,
    compute_global_norm,
    load_manifests,
    load_g1_cache,
    # Evaluation
    eval_cls,
    eval_reid_closed_set,
    # Misc
    set_seed,
    DEVICE,
)

from configs import ALL_ABLATIONS, BASE_TRAIN_CFG, ABLATION_DISPLAY_NAMES  # noqa: E402

# ---------------------------------------------------------------------------
# A3 Model — no temporal aggregation
# ---------------------------------------------------------------------------

class Unveil_NoTemporal(UnveilBackbone):
    """UNVEIL with the multi-scale temporal aggregation replaced by identity
    and no temporal downsampling — each frame is processed independently and
    the final pooling is a plain temporal mean."""

    _DOWN_STAGES = ()  # no temporal stride

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for blk in self.blocks:
            blk.tcn = nn.Identity()


# ---------------------------------------------------------------------------
# A1: Position-only model — streams=1, truly no vel/acc in architecture
# ---------------------------------------------------------------------------

class Unveil_PosOnly(UnveilBackbone):
    """UNVEIL whose kinematic encoder sees a single stream (position only).

    The data BN and the encoder projection are built for one stream instead
    of three, so the model architecturally cannot encode velocity or
    acceleration.
    """

    _STREAMS = 1  # position only

    def forward(self, x, lengths=None):
        return super().forward(x[:, 0:1], lengths)  # keep the position stream


# ---------------------------------------------------------------------------
# A2: Subset+FC model — root+legs+waist joints, fully-connected graph
# ---------------------------------------------------------------------------

# The 20 retained G1 channels (root 0-5, legs 6-17, waist_yaw/roll 18-19)
# grouped into 9 semantic joints; missing DoFs are zero-padded (pad = 20).
_SUBSET_PAD = 20
_SUBSET_JOINT_CHANNELS = [
    ("root_trans", [0, 1, 2]),
    ("root_rot",   [3, 4, 5]),
    ("waist",      [18, 19, _SUBSET_PAD]),   # waist_pitch removed
    ("l_hip",      [6, 7, 8]),
    ("l_knee",     [9, _SUBSET_PAD, _SUBSET_PAD]),
    ("l_ankle",    [10, 11, _SUBSET_PAD]),
    ("r_hip",      [12, 13, 14]),
    ("r_knee",     [15, _SUBSET_PAD, _SUBSET_PAD]),
    ("r_ankle",    [16, 17, _SUBSET_PAD]),
]


class Unveil_SubsetFC(UnveilBackbone):
    """UNVEIL on the root+legs+waist joints with the kinematic hierarchy
    removed: a single fully-connected subgraph replaces the nine intra-limb /
    limb-torso subgraphs.

    SubsetJointsDataset must be applied upstream to reduce input channels
    from 35 → 20 before this model is called.
    """

    def _graph(self):
        J = len(_SUBSET_JOINT_CHANNELS)
        A = np.ones((J, J), dtype=np.float32)   # FC incl. self-loops
        return (_sym_normalize(A)[None],
                np.ones((1, J, J), dtype=np.float32),
                ["fc"])

    def _joint_index(self):
        idx = np.zeros((len(_SUBSET_JOINT_CHANNELS), DOF_PER_JOINT),
                       dtype=np.int64)
        for j, (_, chans) in enumerate(_SUBSET_JOINT_CHANNELS):
            for d, c in enumerate(chans):
                idx[j, d] = c
        return idx


# ---------------------------------------------------------------------------
# Dataset wrappers
# ---------------------------------------------------------------------------

class PositionOnlyDataset(Dataset):
    """Wraps BonesSeedDataset and zeros velocity + acceleration streams."""

    def __init__(self, base_dataset: BonesSeedDataset):
        self.base = base_dataset
        self.labels = base_dataset.labels
        self.task_ids = base_dataset.task_ids
        self.is_regression = base_dataset.is_regression

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        streams, label, task_id = self.base[idx]
        if isinstance(streams, np.ndarray):
            streams = streams.copy()
        else:
            streams = np.array(streams, copy=True)
        streams[1, :, :] = 0.0  # zero velocity
        streams[2, :, :] = 0.0  # zero acceleration
        return streams, label, task_id


class SubsetJointsDataset(Dataset):
    """Selects a subset of joint channels: (3, 35, T) → (3, len(indices), T)."""

    def __init__(self, base_dataset: BonesSeedDataset, indices: List[int]):
        self.base = base_dataset
        self.indices = indices
        self.labels = base_dataset.labels
        self.task_ids = base_dataset.task_ids
        self.is_regression = base_dataset.is_regression

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        streams, label, task_id = self.base[idx]
        if not isinstance(streams, np.ndarray):
            streams = np.array(streams)
        return streams[:, self.indices, :], label, task_id


class TimeAveragedDataset(Dataset):
    """Collapses the temporal axis to a single mean frame: (3, C, T) → (3, C, 1)."""

    def __init__(self, base_dataset: BonesSeedDataset):
        self.base = base_dataset
        self.labels = base_dataset.labels
        self.task_ids = base_dataset.task_ids
        self.is_regression = base_dataset.is_regression

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        streams, label, task_id = self.base[idx]
        if not isinstance(streams, np.ndarray):
            streams = np.array(streams)
        return streams.mean(axis=2, keepdims=True).astype(np.float32), label, task_id


# ---------------------------------------------------------------------------
# Actor subsampling
# ---------------------------------------------------------------------------

def subsample_actors(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_actors: int = 100,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]:
    """Select num_actors from the full pool, then build train/eval splits."""
    rng = np.random.default_rng(seed)

    parts = [df for df in [train_df, val_df, test_df] if not df.empty]
    all_df = pd.concat(parts, ignore_index=True)
    all_actors = sorted(all_df["actor_uid"].dropna().unique().tolist())

    n_select = min(num_actors, len(all_actors))
    selected = rng.choice(all_actors, size=n_select, replace=False).tolist()

    n_unseen = max(1, int(round(test_ratio * len(selected))))
    idx_perm = rng.permutation(len(selected))
    unseen_actors = set(np.array(selected)[idx_perm[:n_unseen]].tolist())
    seen_actors = set(np.array(selected)[idx_perm[n_unseen:]].tolist())

    seen_df = all_df[all_df["actor_uid"].isin(seen_actors)].reset_index(drop=True)
    unseen_df = all_df[all_df["actor_uid"].isin(unseen_actors)].reset_index(drop=True)

    train_parts, eval_parts = [], []
    for _, group in seen_df.groupby("actor_uid"):
        group = group.reset_index(drop=True)
        n = len(group)
        idx = rng.permutation(n)
        n_train = max(1, min(int(round(0.8 * n)), n - 1))
        train_parts.append(group.iloc[idx[:n_train]])
        eval_parts.append(group.iloc[idx[n_train:]])

    train_sub = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=all_df.columns)
    seen_unseen_demos = pd.concat(eval_parts, ignore_index=True) if eval_parts else pd.DataFrame(columns=all_df.columns)

    return train_sub, seen_unseen_demos, unseen_df, len(seen_actors), len(unseen_actors)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(
    ablation_cfg: Dict,
    num_classes: int,
    train_cfg: Dict,
) -> nn.Module:
    kwargs = dict(
        fmt=train_cfg["format"],
        num_classes=num_classes,
        emb_dim=train_cfg["emb_dim"],
        base_channels=train_cfg["base_channels"],
        num_stages=train_cfg["num_stages"],
        num_prototype=train_cfg["num_prototype"],
        dropout=train_cfg["dropout"],
    )
    model_class = ablation_cfg.get("model_class", "default")
    if model_class == "pos_only":
        model = Unveil_PosOnly(**kwargs)
    elif model_class == "subset_fc":
        model = Unveil_SubsetFC(**kwargs)
    elif model_class == "no_temporal":
        model = Unveil_NoTemporal(**kwargs)
    else:
        model = UnveilBackbone(**kwargs)
    return model.to(DEVICE)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    csc_loss_fn: Optional[nn.Module],
    best_metric: float,
    best_epoch: int,
    bad_epochs: int,
    best_state: Optional[Dict],
):
    payload = {
        "epoch": epoch,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "opt_state": opt.state_dict(),
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "bad_epochs": bad_epochs,
        "best_state": {k: v.cpu() for k, v in best_state.items()} if best_state else None,
    }
    if csc_loss_fn is not None:
        payload["csc_loss_state"] = {
            k: v.cpu() for k, v in csc_loss_fn.state_dict().items()
        }
    torch.save(payload, path)


def _load_checkpoint(
    ckpt_dir: Path,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    csc_loss_fn: Optional[nn.Module],
    log: logging.Logger,
) -> Tuple[int, float, int, int, Optional[Dict]]:
    """Load latest or best checkpoint.  Returns (start_epoch, best_metric, best_epoch, bad_epochs, best_state)."""
    latest = ckpt_dir / "latest_checkpoint.pt"
    best = ckpt_dir / "best_model.pt"

    load_path = None
    if latest.exists() and best.exists():
        ep_l = torch.load(latest, map_location="cpu", weights_only=True)["epoch"]
        ep_b = torch.load(best, map_location="cpu", weights_only=True)["epoch"]
        load_path = latest if ep_l >= ep_b else best
    elif latest.exists():
        load_path = latest
    elif best.exists():
        load_path = best

    if load_path is None:
        return 1, -1.0, 0, 0, None

    log.info(f"  Resuming from: {load_path}")
    ckpt = torch.load(load_path, map_location=DEVICE, weights_only=True)

    model.load_state_dict(ckpt["model_state"])
    opt.load_state_dict(ckpt["opt_state"])

    if csc_loss_fn is not None and "csc_loss_state" in ckpt:
        csc_loss_fn.load_state_dict(ckpt["csc_loss_state"])

    resume_epoch = int(ckpt["epoch"])
    best_metric = float(ckpt.get("best_metric", -1.0))
    best_epoch = int(ckpt.get("best_epoch", resume_epoch))
    bad_epochs = int(ckpt.get("bad_epochs", 0))
    best_state = ckpt.get("best_state")
    if best_state is not None:
        best_state = {k: v.clone() for k, v in best_state.items()}

    log.info(
        f"  Loaded epoch {resume_epoch} | best_metric={best_metric:.4f} "
        f"@ epoch {best_epoch} | bad_epochs={bad_epochs}"
    )
    return resume_epoch + 1, best_metric, best_epoch, bad_epochs, best_state


# ---------------------------------------------------------------------------
# Training loop (single run)
# ---------------------------------------------------------------------------

def run_ablation(
    task: str,
    ablation_cfg: Dict,
    train_cfg: Dict,
    train_df: pd.DataFrame,
    seen_unseen_demos_df: pd.DataFrame,
    unseen_actors_df: pd.DataFrame,
    n_seen_actors: int,
    n_unseen_actors: int,
    data_root: str,
    g1_cache_info: Optional[Dict],
    output_dir: Path,
    log: logging.Logger,
    resume: bool = False,
) -> Dict:
    ablation_name = ablation_cfg["ablation"]
    run_id = f"{task}_{ablation_name}"
    ckpt_dir = output_dir / "checkpoints" / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics" / f"{run_id}.json"
    epochs = train_cfg["epochs"]

    # When resuming: skip runs that already reached the requested epoch budget.
    # When not resuming: always train from scratch (ignore any existing checkpoints).
    if resume:
        _latest = ckpt_dir / "latest_checkpoint.pt"
        _best = ckpt_dir / "best_model.pt"
        _existing = _latest if _latest.exists() else (_best if _best.exists() else None)
        if _existing is not None:
            _saved_epoch = torch.load(_existing, map_location="cpu", weights_only=True)["epoch"]
            if _saved_epoch >= epochs and metrics_path.exists():
                log.info(
                    f"[{run_id}] Already complete at epoch {_saved_epoch} "
                    f"(requested {epochs}). Loading saved metrics."
                )
                with open(metrics_path) as f:
                    return json.load(f)

    set_seed(train_cfg["seed"])
    log.info(f"\n{'='*72}")
    log.info(f"[{run_id}] Starting: {ablation_cfg['description']}")
    log.info(f"  task={task}  ablation={ablation_name}  device={DEVICE}")
    log.info(f"  train={len(train_df):,}  seen-unseen-demos={len(seen_unseen_demos_df):,}  unseen-actors={len(unseen_actors_df):,}")
    log.info(f"{'='*72}")

    # ── Label setup ──────────────────────────────────────────────────────────
    label_col = "actor_uid" if task == "reid" else "actor_gender"
    path_col = "move_g1_mujoco_path"

    all_labels = sorted(train_df[label_col].dropna().unique().tolist())
    label_map = {lbl: i for i, lbl in enumerate(all_labels)}
    num_classes = len(label_map)
    log.info(f"  num_classes={num_classes}")

    # Filter eval sets to known labels
    seen_unseen_demos_df = seen_unseen_demos_df[
        seen_unseen_demos_df[label_col].isin(label_map)
    ].reset_index(drop=True)

    # For re-id, unseen actors aren't in the classifier head — evaluation only
    # on seen-actors-unseen-demos (closed-set). For gender, unseen actors are valid.
    if task == "gender":
        unseen_actors_df = unseen_actors_df[
            unseen_actors_df[label_col].isin(label_map)
        ].reset_index(drop=True)

    # ── Task map (dummy single task — all g1, no deconfounding needed) ───────
    task_map = {"g1": 0}
    task_col = "package" if "package" in train_df.columns else None

    if task_col and task_col in train_df.columns:
        all_tasks = sorted(set(train_df[task_col].dropna().unique().tolist()))
        task_map = {t: i for i, t in enumerate(all_tasks)}

    # ── Normalization ─────────────────────────────────────────────────────────
    downsample_factor = max(1, 120 // train_cfg["target_fps"])
    log.info("Computing normalization stats...")
    global_mean, global_std = compute_global_norm(
        train_df, data_root, "g1", None, downsample_factor,
        max_samples=3000, seed=train_cfg["seed"], g1_cache_info=g1_cache_info,
    )

    # ── Datasets ──────────────────────────────────────────────────────────────
    ds_kwargs = dict(
        data_root=data_root,
        fmt="g1",
        label_col=label_col,
        label_map=label_map,
        is_regression=False,
        task_col=task_col or "package",
        task_map=task_map,
        channel_indices=None,
        downsample_factor=downsample_factor,
        max_seq_len=train_cfg["max_seq_len"],
        min_seq_len=train_cfg["min_seq_len"],
        global_mean=global_mean,
        global_std=global_std,
        seed=train_cfg["seed"],
        g1_cache_info=g1_cache_info,
    )

    log.info("Loading datasets...")
    train_ds_base = BonesSeedDataset(train_df, train=True, **ds_kwargs)
    seen_unseen_ds_base = BonesSeedDataset(seen_unseen_demos_df, train=False, **ds_kwargs)
    unseen_actors_ds_base = (
        BonesSeedDataset(unseen_actors_df, train=False, **ds_kwargs)
        if task == "gender" else None
    )

    # Apply dataset wrapper based on ablation variant
    subset = ablation_cfg.get("subset_joints")
    if subset is not None:
        # A2: keep only specified joint channels
        def _wrap_subset(ds):
            return SubsetJointsDataset(ds, subset) if ds is not None else None
        train_ds = _wrap_subset(train_ds_base)
        seen_unseen_ds = _wrap_subset(seen_unseen_ds_base)
        unseen_actors_ds = _wrap_subset(unseen_actors_ds_base)
    elif ablation_cfg.get("time_average_input"):
        # A3: collapse temporal axis to single mean frame
        def _wrap_tavg(ds):
            return TimeAveragedDataset(ds) if ds is not None else None
        train_ds = _wrap_tavg(train_ds_base)
        seen_unseen_ds = _wrap_tavg(seen_unseen_ds_base)
        unseen_actors_ds = _wrap_tavg(unseen_actors_ds_base)
    else:
        # A0 (full) and A1 (pos_only): no dataset wrapper needed.
        # The PosOnly model keeps only the position stream in its forward().
        train_ds = train_ds_base
        seen_unseen_ds = seen_unseen_ds_base
        unseen_actors_ds = unseen_actors_ds_base

    _loader_kw = dict(num_workers=0, collate_fn=collate_padded, pin_memory=(DEVICE == "cuda"))
    train_loader = DataLoader(train_ds, train_cfg["batch_size"], shuffle=True, drop_last=True, **_loader_kw)
    seen_unseen_loader = DataLoader(seen_unseen_ds, train_cfg["batch_size"], shuffle=False, **_loader_kw)
    unseen_actors_loader = (
        DataLoader(unseen_actors_ds, train_cfg["batch_size"], shuffle=False, **_loader_kw)
        if unseen_actors_ds is not None else None
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(ablation_cfg, num_classes, train_cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  params={n_params:,}  layout={ablation_cfg['graph_layout']}")

    # ── Losses ────────────────────────────────────────────────────────────────
    ce_fn = nn.CrossEntropyLoss(label_smoothing=train_cfg["label_smoothing"])
    supcon_fn = SupConLoss(temperature=train_cfg["supcon_temp"])

    # Operator-contrastive loss on the distilled signature; its input width
    # follows the variant's joint count (V*V joint pairs).
    csc_loss_fn = SignatureContrastiveLoss(
        n_class=num_classes,
        in_dim=model.num_joints * model.num_joints,
        h_dim=min(256, train_cfg["emb_dim"]),
    ).to(DEVICE)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )

    # ── Resume from checkpoint if requested ───────────────────────────────────
    eval_every = train_cfg["eval_every"]
    early_stop = train_cfg["early_stop"]
    supcon_warmup = train_cfg["supcon_warmup"]

    if resume:
        start_epoch, best_metric, best_epoch, bad_epochs, best_state = _load_checkpoint(
            ckpt_dir, model, opt, csc_loss_fn, log
        )
        if start_epoch == 1:
            log.info("  No checkpoint found — training from scratch.")
    else:
        start_epoch, best_metric, best_epoch, bad_epochs, best_state = 1, -1.0, 0, 0, None
        log.info("  --resume not set — training from scratch.")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_sum = ce_sum = sc_sum = csc_sum = 0.0
        correct = total = skipped = 0
        t0 = time.time()

        for xb, lengths, yb, tids in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            if not torch.isfinite(xb).all():
                xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
                if not torch.isfinite(xb).all():
                    skipped += 1
                    continue

            logits, z, signature = model(xb)
            if not torch.isfinite(logits).all():
                skipped += 1
                continue

            loss_ce = ce_fn(logits, yb)
            ce_sum += loss_ce.item() * xb.size(0)
            loss = loss_ce

            lam_sc = train_cfg["lambda_supcon"] if epoch >= supcon_warmup else 0.0
            if lam_sc > 0:
                loss_sc = supcon_fn(z, yb)
                loss = loss + lam_sc * loss_sc
                sc_sum += loss_sc.item() * xb.size(0)

            lam_csc = train_cfg["lambda_csc"]
            if lam_csc > 0 and epoch >= supcon_warmup and signature is not None:
                try:
                    loss_csc = csc_loss_fn(signature, yb)
                    if torch.isfinite(loss_csc):
                        loss = loss + lam_csc * loss_csc
                        csc_sum += loss_csc.item() * xb.size(0)
                except Exception:
                    pass

            if not torch.isfinite(loss):
                skipped += 1
                continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            loss_sum += loss.item() * xb.size(0)
            correct += int((logits.argmax(1) == yb).sum())
            total += int(yb.numel())

        tr_acc = correct / max(1, total)
        tr_loss = loss_sum / max(1, total)
        elapsed = time.time() - t0

        if epoch % eval_every == 0 or epoch == epochs:
            if task == "reid":
                su_top1, su_top5, su_f1, su_auc, _, _ = eval_reid_closed_set(model, seen_unseen_loader)
                primary_metric = su_top1
                metric_str = f"seen-unseen top1={su_top1:.4f} top5={su_top5:.4f} f1={su_f1:.4f}"
            else:
                su_acc, su_c, su_n = eval_cls(model, seen_unseen_loader)
                if unseen_actors_loader is not None:
                    un_acc, un_c, un_n = eval_cls(model, unseen_actors_loader)
                    metric_str = f"seen-unseen={su_acc:.4f} unseen-actors={un_acc:.4f}"
                    # Primary: unseen-actors acc. Tiebreak with seen-unseen-demos.
                    primary_metric = un_acc + 1e-4 * su_acc
                else:
                    un_acc = 0.0
                    metric_str = f"seen-unseen={su_acc:.4f}"
                    primary_metric = su_acc

            marker = " <-- best" if primary_metric > best_metric + 1e-6 else ""
            log.info(
                f"  Ep {epoch:03d}/{epochs} [{elapsed:.1f}s] "
                f"loss={tr_loss:.4f} train={tr_acc:.4f} | {metric_str}{marker}"
            )

            if primary_metric > best_metric + 1e-6:
                best_metric = primary_metric
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
                _save_checkpoint(
                    ckpt_dir / "best_model.pt",
                    epoch, model, opt, csc_loss_fn,
                    best_metric, best_epoch, bad_epochs, best_state,
                )
            else:
                bad_epochs += 1
                if bad_epochs >= early_stop:
                    log.info(f"  Early stop at epoch {epoch} ({early_stop} evals without improvement)")
                    break

            # Always write latest so we can resume after any interruption
            _save_checkpoint(
                ckpt_dir / "latest_checkpoint.pt",
                epoch, model, opt, csc_loss_fn,
                best_metric, best_epoch, bad_epochs, best_state,
            )
        else:
            log.info(
                f"  Ep {epoch:03d}/{epochs} [{elapsed:.1f}s] "
                f"loss={tr_loss:.4f} train={tr_acc:.4f}"
            )

    # ── Final evaluation with best checkpoint ─────────────────────────────────
    eval_model = build_model(ablation_cfg, num_classes, {**train_cfg, "dropout": 0.0})
    if best_state is not None:
        eval_model.load_state_dict(best_state)
    eval_model.eval()

    log.info(f"\n[{run_id}] Final evaluation (best epoch {best_epoch})")

    result: Dict = {
        "task": task,
        "ablation": ablation_name,
        "description": ablation_cfg["description"],
        "num_actors_total": n_seen_actors + n_unseen_actors,
        "num_seen_actors": n_seen_actors,
        "num_unseen_actors": n_unseen_actors,
        "train_samples": len(train_ds),
        "seen_unseen_demos_samples": len(seen_unseen_ds),
        "unseen_actors_samples": len(unseen_actors_ds) if unseen_actors_ds else 0,
        "best_epoch": best_epoch,
    }

    if task == "reid":
        top1, top5, f1_, auc_, _, _ = eval_reid_closed_set(eval_model, seen_unseen_loader)
        result.update({
            "seen_actors_unseen_demos_top1": top1,
            "seen_actors_unseen_demos_top5": top5,
            "seen_actors_unseen_demos_f1_macro": f1_,
            "seen_actors_unseen_demos_roc_auc": auc_,
            "primary_metric": top1,
        })
        log.info(f"  seen-actors-unseen-demos: top1={top1:.4f} top5={top5:.4f} f1={f1_:.4f}")
    else:
        su_acc, _, _ = eval_cls(eval_model, seen_unseen_loader)
        result["seen_actors_unseen_demos_accuracy"] = su_acc
        log.info(f"  seen-actors-unseen-demos acc={su_acc:.4f}")
        if unseen_actors_loader is not None:
            un_acc, _, _ = eval_cls(eval_model, unseen_actors_loader)
            result["unseen_actors_accuracy"] = un_acc
            result["primary_metric"] = su_acc
            log.info(f"  unseen-actors acc={un_acc:.4f}")
        else:
            result["unseen_actors_accuracy"] = float("nan")
            result["primary_metric"] = su_acc

    # Save metrics
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"  Saved -> {metrics_path}")

    return result


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def print_summary(task: str, results: List[Dict], output_dir: Path, log: logging.Logger):
    ablation_order = ["full", "pos_only_v2", "subset_fc", "time_averaged"]

    if task == "reid":
        col_header = "Seen-Unseen-Demos top-1"
        col_key = "seen_actors_unseen_demos_top1"
        note = "(unseen-actors not reported for re-id: actors not in classifier head)"
    else:
        col_header = "Seen-Unseen-Demos acc"
        col_key = "seen_actors_unseen_demos_accuracy"

    by_ablation = {r["ablation"]: r for r in results}
    baseline = by_ablation.get("full", {})
    baseline_val = baseline.get(col_key, float("nan"))

    rows = []
    for abl in ablation_order:
        r = by_ablation.get(abl)
        if r is None:
            continue
        val = r.get(col_key, float("nan"))
        delta = (val - baseline_val) if abl != "full" else None
        rows.append((abl, ABLATION_DISPLAY_NAMES.get(abl, abl), val, delta))

    col_w = 24
    val_w = 26
    dlt_w = 8

    header = f"{'Ablation':<{col_w}} {'Metric (' + col_header + ')':<{val_w}} {'Δ':>{dlt_w}}"
    sep = "-" * (col_w + val_w + dlt_w + 2)
    log.info(f"\nTask: {task.upper()}")
    if task == "reid":
        log.info(f"  Note: {note}")
    log.info(sep)
    log.info(header)
    log.info(sep)
    for abl, display, val, delta in rows:
        val_str = f"{val:.4f}" if not (val != val) else "  n/a"
        dlt_str = f"{delta:+.4f}" if delta is not None and not (delta != delta) else "  —"
        log.info(f"  {display:<{col_w-2}} {val_str:<{val_w}} {dlt_str:>{dlt_w}}")
    log.info(sep)

    # Save CSV
    csv_rows = []
    for abl, display, val, delta in rows:
        row = {"ablation": abl, "display_name": display, col_key: val}
        if delta is not None:
            row["delta"] = delta
        csv_rows.append(row)
    csv_path = output_dir / f"summary_{task}.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    log.info(f"  Summary CSV -> {csv_path}")

    if task == "gender":
        un_key = "unseen_actors_accuracy"
        if any(un_key in r for r in results):
            log.info(f"\n  Unseen-actors accuracy:")
            log.info(f"  {'Ablation':<{col_w}} {'Unseen-Actors Acc':<{val_w}}")
            log.info(f"  {'-'*(col_w+val_w)}")
            for abl in ablation_order:
                r = by_ablation.get(abl)
                if r is None:
                    continue
                uval = r.get(un_key, float("nan"))
                ustr = f"{uval:.4f}" if not (uval != uval) else "  n/a"
                log.info(f"  {ABLATION_DISPLAY_NAMES.get(abl, abl):<{col_w}} {ustr}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UNVEIL Ablation Runner")
    p.add_argument("--task", choices=["gender", "reid", "all"], default="all")
    p.add_argument("--ablation",
                   choices=["full", "pos_only_v2", "subset_fc", "time_averaged", "all"],
                   default="all")
    p.add_argument("--num-actors", type=int, default=100)
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("--splits-dir", type=str, default=None)
    p.add_argument("--g1-cache-dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None,
                   help="Override device (cuda/cpu). Default: auto-detect.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output dir (default: ablation/outputs relative to script)")
    p.add_argument("--resume", action="store_true", default=False,
                   help="Resume training from saved checkpoints. "
                        "Default: start from scratch (safe after architecture changes).")
    return p.parse_args()


def setup_logger(log_path: Path, name: str) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


def main():
    args = parse_args()

    # Override device if requested (default: auto-detect via unveil.DEVICE)
    global DEVICE
    if args.device:
        import unveil as _ref
        _ref.DEVICE = args.device
        DEVICE = args.device  # rebind this module's copy too

    output_dir = Path(args.output_dir) if args.output_dir else _ABLATION_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    master_log = setup_logger(output_dir / "logs" / "ablation_master.log", "ablation_master")

    # ── Resolve paths ─────────────────────────────────────────────────────────
    data_root = str(Path(args.data_root).resolve())
    if args.splits_dir is None:
        from project_paths import default_splits_dir
        splits_dir = str(default_splits_dir(create=False))
    else:
        splits_dir = args.splits_dir

    g1_cache_info = None
    if args.g1_cache_dir:
        g1_cache_info = load_g1_cache(args.g1_cache_dir)
    else:
        from project_paths import default_g1_cache_dir
        g1_cache_info = load_g1_cache(str(default_g1_cache_dir(create=False)))

    if g1_cache_info:
        master_log.info(f"G1 cache: {len(g1_cache_info['index_map']):,} clips")
    else:
        master_log.info("G1 cache not found, falling back to CSV reads.")

    # ── Load manifests ────────────────────────────────────────────────────────

    class _FakeArgs:
        def __init__(self, splits_dir):
            self.splits_dir = splits_dir

    train_df, val_df, test_df = load_manifests(_FakeArgs(splits_dir))
    master_log.info(
        f"Manifests loaded: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}"
    )

    # ── Subsample 100 actors ──────────────────────────────────────────────────
    train_sub, seen_unseen_demos, unseen_actors, n_seen, n_unseen = subsample_actors(
        train_df, val_df, test_df,
        num_actors=args.num_actors,
        test_ratio=0.2,
        seed=args.seed,
    )
    master_log.info(
        f"100-actor subsample: seen={n_seen} unseen={n_unseen} | "
        f"train={len(train_sub):,} seen-unseen-demos={len(seen_unseen_demos):,} "
        f"unseen-actors={len(unseen_actors):,}"
    )

    # ── Select tasks and ablations to run ─────────────────────────────────────
    tasks = ["gender", "reid"] if args.task == "all" else [args.task]
    ablations = list(ALL_ABLATIONS.keys()) if args.ablation == "all" else [args.ablation]

    train_cfg = dict(BASE_TRAIN_CFG)  # shared hyperparameters
    all_results: Dict[str, List[Dict]] = {t: [] for t in tasks}

    # ── Run experiments ───────────────────────────────────────────────────────
    for task in tasks:
        for abl_name in ablations:
            abl_cfg = ALL_ABLATIONS[abl_name]
            run_log = setup_logger(
                output_dir / "logs" / f"{task}_{abl_name}.log",
                f"ablation_{task}_{abl_name}",
            )
            result = run_ablation(
                task=task,
                ablation_cfg=abl_cfg,
                train_cfg=train_cfg,
                train_df=train_sub.copy(),
                seen_unseen_demos_df=seen_unseen_demos.copy(),
                unseen_actors_df=unseen_actors.copy(),
                n_seen_actors=n_seen,
                n_unseen_actors=n_unseen,
                data_root=data_root,
                g1_cache_info=g1_cache_info,
                output_dir=output_dir,
                log=run_log,
                resume=args.resume,
            )
            all_results[task].append(result)

    # ── Print summary tables ──────────────────────────────────────────────────
    master_log.info("\n" + "=" * 72)
    master_log.info("ABLATION SUMMARY")
    master_log.info("=" * 72)
    for task in tasks:
        if all_results[task]:
            print_summary(task, all_results[task], output_dir, master_log)

    # Save combined summary JSON
    combined = {
        "tasks": tasks,
        "ablations": ablations,
        "num_actors": args.num_actors,
        "results": all_results,
    }
    with open(output_dir / "summary_all.json", "w") as f:
        json.dump(combined, f, indent=2)
    master_log.info(f"\nAll done. Results in {output_dir}")


if __name__ == "__main__":
    main()
