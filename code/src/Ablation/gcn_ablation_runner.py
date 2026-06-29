#!/usr/bin/env python3
"""
InveRT DS-GCN Ablation Experiment Runner (NeurIPS 2026)
=======================================================

Runs 4 ablation variants × {gender, reid} tasks on a 100-actor subsample of
BONES-SEED using the DS-GCN (DGSTGCN) backbone.

Usage:
    python ablation/dsgcn_ablation_runner.py --task all --ablation all \
        --data-root . --splits-dir artifacts/splits

Re-run a single pair without touching others:
    python ablation/dsgcn_ablation_runner.py --task gender --ablation fc_graph
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ABLATION_DIR = Path(__file__).resolve().parent
_BONES_ROOT = _ABLATION_DIR.parent

for _p in [str(_BONES_ROOT), str(_BONES_ROOT / "src")]:   # src/ holds pyskl + DS-GCN
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Import DS-GCN reference script.
# This import triggers the pyskl path-setup side-effects, making DGSTGCN
# and the real pyskl Graph available in sys.modules.
# ---------------------------------------------------------------------------
from dsgcn_bones_seed import (  # noqa: E402
    DGSTGCNBonesSeed,
    BonesSeedDataset,
    collate_padded,
    compute_global_norm,
    load_manifests,
    load_g1_cache,
    g1_cache_fetch,
    SupConLoss,
    eval_cls,
    eval_reid_closed_set,
    set_seed,
    DEVICE,
)

# Reuse actor-subsampling and position-only wrapper from the ProtoGCN runner
from ablation.ablation_runner import subsample_actors, PositionOnlyDataset

from ablation.configs_dsgcn import (
    ALL_ABLATIONS_DSGCN,
    BASE_TRAIN_CFG_DSGCN,
    ABLATION_DISPLAY_NAMES,
)

# ---------------------------------------------------------------------------
# Patch the pyskl Graph class to add the fully-connected G1 layout (A2)
# ---------------------------------------------------------------------------
# dsgcn_bones_seed imports load the real pyskl Graph via the stub and register
# it in sys.modules["pyskl.utils.graph"]. We extend that class in-place.

_pyskl_graph_mod = sys.modules.get("pyskl.utils.graph")
if _pyskl_graph_mod is None:
    import pyskl.utils as _psku  # noqa: F401
    _pyskl_graph_mod = sys.modules["pyskl.utils.graph"]

_PyGraph = _pyskl_graph_mod.Graph
_original_pyskl_get_layout = _PyGraph.get_layout


def _pyskl_extended_get_layout(self, layout: str):
    if layout == "bones_seed_g1_fc":
        self.num_node = 35
        self.center = 0
        self.inward = [(i, j) for i in range(35) for j in range(35) if i != j]
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward

        # node_type: all same (0) — uniform body-part type
        node_type = [0] * 35
        self.node_type = node_type

        # edge_type: compute the same way as other layouts
        idx = np.array(node_type).reshape(self.num_node, 1) + 1
        idx = idx * pow(-1, idx)
        edge_type_index = np.dot(idx, idx.T)
        unique, _ = np.unique(edge_type_index, return_counts=True)
        self.edge_type = np.zeros([self.num_node, self.num_node])
        for i in range(len(unique)):
            self.edge_type[edge_type_index == unique[i]] = i
        self.edge_type_num = unique
    else:
        _original_pyskl_get_layout(self, layout)


# Also need to lift the assertion in __init__ that checks allowed layouts
_original_pyskl_graph_init = _PyGraph.__init__


def _pyskl_patched_init(self, layout="coco", mode="spatial", max_hop=1,
                         nx_node=1, num_filter=3, init_std=0.02, init_off=0.04):
    from DS_GCN.pyskl.utils.graph import get_hop_distance  # noqa: F401 — may fail
    # Call patched get_layout without the assertion
    self.max_hop = max_hop
    self.layout = layout
    self.mode = mode
    self.num_filter = num_filter
    self.init_std = init_std
    self.init_off = init_off
    self.nx_node = nx_node

    assert nx_node == 1 or mode == "random", \
        "nx_node can be > 1 only if mode is 'random'"

    self.get_layout(layout)

    from DS_GCN.pyskl.utils.graph import get_hop_distance as _ghd
    self.hop_dis = _ghd(self.num_node, self.inward, max_hop)

    assert hasattr(self, mode), f"Do Not Exist This Mode: {mode}"
    self.A = getattr(self, mode)()


# Simpler: just patch get_layout and keep original __init__ (it calls get_layout).
# The original __init__ has an assert on layout names — bypass it by subclassing.

class _PatchedPyGraph(_PyGraph):
    """Subclass that adds bones_seed_g1_fc and removes the layout assertion."""

    def __init__(self, layout="coco", mode="spatial", max_hop=1,
                 nx_node=1, num_filter=3, init_std=0.02, init_off=0.04):
        # Store attrs the parent __init__ would set before calling get_layout
        self.max_hop = max_hop
        self.layout = layout
        self.mode = mode
        self.num_filter = num_filter
        self.init_std = init_std
        self.init_off = init_off
        self.nx_node = nx_node

        assert nx_node == 1 or mode == "random"

        self.get_layout(layout)

        # get_hop_distance is defined at module level in the real graph module
        hop_fn = getattr(_pyskl_graph_mod, "get_hop_distance", None)
        if hop_fn is None:
            # fall back: find it in the parent class module
            import inspect
            hop_fn = inspect.getmodule(_PyGraph).get_hop_distance
        self.hop_dis = hop_fn(self.num_node, self.inward, max_hop)

        assert hasattr(self, mode), f"Do Not Exist This Mode: {mode}"
        self.A = getattr(self, mode)()

    def get_layout(self, layout: str):
        _pyskl_extended_get_layout(self, layout)


# Register the patched class as the Graph used by DGSTGCN.
# DGSTGCN holds a reference to the class object via `from ...utils import Graph`.
# We replace it in the stub module and in sys.modules so new DGSTGCN instances
# created after this point use _PatchedPyGraph.
import pyskl.utils as _pyskl_utils_mod
_pyskl_utils_mod.Graph = _PatchedPyGraph
_pyskl_graph_mod.Graph = _PatchedPyGraph

# Also patch the dgstgcn module's reference to Graph so it uses the patched one
import sys as _sys
_dgstgcn_mod = _sys.modules.get("pyskl.models.gcns.utils.dgstgcn") or \
               _sys.modules.get("pyskl.models.gcns.dgstgcn")
if _dgstgcn_mod and hasattr(_dgstgcn_mod, "Graph"):
    _dgstgcn_mod.Graph = _PatchedPyGraph

# ---------------------------------------------------------------------------
# A3: No-temporal DS-GCN — replace TCN in each DGBlock with nn.Identity()
# ---------------------------------------------------------------------------

def _apply_no_temporal(model: DGSTGCNBonesSeed) -> DGSTGCNBonesSeed:
    """Post-process: replace TCN with identity and fix strided residuals."""
    from pyskl.models.gcns.utils import unit_tcn as pyskl_unit_tcn

    for block in model.backbone.gcn:
        # Replace temporal conv with identity
        block.tcn = nn.Identity()

        # Fix stride-2 residual projections: replace with stride-1 version
        # so temporal dimensions don't get downsampled.
        if (isinstance(block.residual, pyskl_unit_tcn)
                and getattr(block.residual, "stride", 1) == 2):
            in_ch = block.residual.conv.in_channels
            out_ch = block.residual.conv.out_channels
            block.residual = pyskl_unit_tcn(
                in_ch, out_ch, kernel_size=1, stride=1
            ).to(next(model.parameters()).device)

    return model


class DGSTGCNBonesSeed_NoTemporal(DGSTGCNBonesSeed):
    """DGSTGCNBonesSeed with all temporal convolutions replaced by identity."""

    def __init__(self, fmt, num_classes, emb_dim=256, base_channels=64,
                 num_stages=10, dropout=0.5):
        super().__init__(fmt=fmt, num_classes=num_classes, emb_dim=emb_dim,
                         base_channels=base_channels, num_stages=num_stages,
                         dropout=dropout)
        _apply_no_temporal(self)


# ---------------------------------------------------------------------------
# A2: FC-graph DS-GCN — builds model with bones_seed_g1_fc layout
# ---------------------------------------------------------------------------

class DGSTGCNBonesSeed_FCGraph(DGSTGCNBonesSeed):
    """DGSTGCNBonesSeed with fully-connected G1 graph (no kinematic bias)."""

    def __init__(self, fmt, num_classes, emb_dim=256, base_channels=64,
                 num_stages=10, dropout=0.5):
        # Call nn.Module init directly to rebuild backbone with fc graph
        nn.Module.__init__(self)
        from pyskl.models.gcns.dgstgcn import DGSTGCN

        self.fmt = fmt
        self.num_classes = num_classes

        if fmt == "g1":
            in_channels = 3
            self.num_joints = 35
            layout = "bones_seed_g1_fc"
        else:
            # BVH: keep kinematic graph (fc ablation only targets g1 in this study)
            in_channels = 9
            self.num_joints = 24
            layout = "bones_seed_bvh"

        graph_cfg = dict(layout=layout, mode="spatial")
        self._out_channels = int(base_channels * (2 ** 2))

        self.backbone = DGSTGCN(
            graph_cfg=graph_cfg,
            in_channels=in_channels,
            base_channels=base_channels,
            ch_ratio=2,
            num_stages=num_stages,
            inflate_stages=[5, 8],
            down_stages=[5, 8],
            data_bn_type="VC",
            num_person=1,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.fc = nn.Linear(self._out_channels, num_classes)
        self.z_proj = nn.Sequential(
            nn.Linear(self._out_channels, emb_dim),
            nn.LayerNorm(emb_dim),
        )

        # Bind _reshape_input and forward from the parent
        self._reshape_input = types.MethodType(
            DGSTGCNBonesSeed._reshape_input, self
        )
        self.forward = types.MethodType(DGSTGCNBonesSeed.forward, self)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model_dsgcn(
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
        dropout=train_cfg["dropout"],
    )

    if ablation_cfg["no_temporal"]:
        model = DGSTGCNBonesSeed_NoTemporal(**kwargs)
    elif ablation_cfg["graph_layout"] == "bones_seed_g1_fc":
        model = DGSTGCNBonesSeed_FCGraph(**kwargs)
    else:
        model = DGSTGCNBonesSeed(**kwargs)

    return model.to(DEVICE)


# ---------------------------------------------------------------------------
# Training loop (single run)
# ---------------------------------------------------------------------------

def run_ablation_dsgcn(
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
) -> Dict:
    ablation_name = ablation_cfg["ablation"]
    run_id = f"dsgcn_{task}_{ablation_name}"
    ckpt_dir = output_dir / "checkpoints" / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics" / f"{run_id}.json"
    if metrics_path.exists():
        log.info(f"[{run_id}] Already complete — skipping. "
                 f"Delete {metrics_path} to re-run.")
        with open(metrics_path) as f:
            return json.load(f)

    set_seed(train_cfg["seed"])
    log.info(f"\n{'='*72}")
    log.info(f"[{run_id}] {ablation_cfg['description']}")
    log.info(f"  task={task}  ablation={ablation_name}  device={DEVICE}")
    log.info(f"  train={len(train_df):,}  "
             f"seen-unseen-demos={len(seen_unseen_demos_df):,}  "
             f"unseen-actors={len(unseen_actors_df):,}")
    log.info(f"{'='*72}")

    # ── Label setup ──────────────────────────────────────────────────────────
    label_col = "actor_uid" if task == "reid" else "actor_gender"

    all_labels = sorted(train_df[label_col].dropna().unique().tolist())
    label_map = {lbl: i for i, lbl in enumerate(all_labels)}
    num_classes = len(label_map)

    seen_unseen_demos_df = seen_unseen_demos_df[
        seen_unseen_demos_df[label_col].isin(label_map)
    ].reset_index(drop=True)

    if task == "gender":
        unseen_actors_df = unseen_actors_df[
            unseen_actors_df[label_col].isin(label_map)
        ].reset_index(drop=True)

    # ── Task map ──────────────────────────────────────────────────────────────
    task_col = "package" if "package" in train_df.columns else None
    task_map = {"g1": 0}
    if task_col:
        all_tasks = sorted(train_df[task_col].dropna().unique().tolist())
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

    if ablation_cfg["pos_only"]:
        train_ds = PositionOnlyDataset(train_ds_base)
        seen_unseen_ds = PositionOnlyDataset(seen_unseen_ds_base)
        unseen_actors_ds = (
            PositionOnlyDataset(unseen_actors_ds_base)
            if unseen_actors_ds_base is not None else None
        )
    else:
        train_ds = train_ds_base
        seen_unseen_ds = seen_unseen_ds_base
        unseen_actors_ds = unseen_actors_ds_base

    _loader_kw = dict(
        num_workers=0, collate_fn=collate_padded,
        pin_memory=(DEVICE == "cuda"),
    )
    train_loader = DataLoader(
        train_ds, train_cfg["batch_size"], shuffle=True, drop_last=True,
        **_loader_kw,
    )
    seen_unseen_loader = DataLoader(
        seen_unseen_ds, train_cfg["batch_size"], shuffle=False, **_loader_kw
    )
    unseen_actors_loader = (
        DataLoader(unseen_actors_ds, train_cfg["batch_size"],
                   shuffle=False, **_loader_kw)
        if unseen_actors_ds is not None else None
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model_dsgcn(ablation_cfg, num_classes, train_cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  params={n_params:,}  layout={ablation_cfg['graph_layout']}")

    # ── Losses ────────────────────────────────────────────────────────────────
    ce_fn = nn.CrossEntropyLoss(label_smoothing=train_cfg["label_smoothing"])
    supcon_fn = SupConLoss(temperature=train_cfg["supcon_temp"])

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_metric = -1.0
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    epochs = train_cfg["epochs"]
    eval_every = train_cfg["eval_every"]
    early_stop = train_cfg["early_stop"]
    supcon_warmup = train_cfg["supcon_warmup"]

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = ce_sum = sc_sum = 0.0
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

            logits, z = model(xb)
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
                su_top1, su_top5, su_f1, su_auc, _, _ = eval_reid_closed_set(
                    model, seen_unseen_loader
                )
                primary_metric = su_top1
                metric_str = (
                    f"seen-unseen top1={su_top1:.4f} top5={su_top5:.4f} "
                    f"f1={su_f1:.4f}"
                )
            else:
                su_acc, su_c, su_n = eval_cls(model, seen_unseen_loader)
                metric_str = f"seen-unseen={su_acc:.4f} ({su_c}/{su_n})"
                if unseen_actors_loader is not None:
                    un_acc, un_c, un_n = eval_cls(model, unseen_actors_loader)
                    metric_str += f" | unseen-actors={un_acc:.4f} ({un_c}/{un_n})"
                primary_metric = su_acc

            marker = " <-- best" if primary_metric > best_metric + 1e-6 else ""
            log.info(
                f"  Ep {epoch:03d}/{epochs} [{elapsed:.1f}s] "
                f"loss={tr_loss:.4f} train={tr_acc:.4f} | {metric_str}{marker}"
            )

            if primary_metric > best_metric + 1e-6:
                best_metric = primary_metric
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                bad_epochs = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": best_state,
                        "best_metric": best_metric,
                    },
                    ckpt_dir / "best_model.pt",
                )
            else:
                bad_epochs += 1
                if bad_epochs >= early_stop:
                    log.info(
                        f"  Early stop at epoch {epoch} "
                        f"({early_stop} evals without improvement)"
                    )
                    break
        else:
            log.info(
                f"  Ep {epoch:03d}/{epochs} [{elapsed:.1f}s] "
                f"loss={tr_loss:.4f} train={tr_acc:.4f}"
            )

    # ── Final evaluation with best weights ────────────────────────────────────
    eval_model = build_model_dsgcn(
        ablation_cfg, num_classes, {**train_cfg, "dropout": 0.0}
    )
    if best_state is not None:
        eval_model.load_state_dict(best_state)
    eval_model.eval()

    log.info(f"\n[{run_id}] Final evaluation (best epoch {best_epoch})")

    result: Dict = {
        "task": task,
        "ablation": ablation_name,
        "model": "dsgcn",
        "description": ablation_cfg["description"],
        "num_actors_total": n_seen_actors + n_unseen_actors,
        "num_seen_actors": n_seen_actors,
        "num_unseen_actors": n_unseen_actors,
        "train_samples": len(train_ds),
        "seen_unseen_demos_samples": len(seen_unseen_ds),
        "unseen_actors_samples": (
            len(unseen_actors_ds) if unseen_actors_ds is not None else 0
        ),
        "best_epoch": best_epoch,
    }

    if task == "reid":
        top1, top5, f1_, auc_, _, _ = eval_reid_closed_set(
            eval_model, seen_unseen_loader
        )
        result.update({
            "seen_actors_unseen_demos_top1": top1,
            "seen_actors_unseen_demos_top5": top5,
            "seen_actors_unseen_demos_f1_macro": f1_,
            "seen_actors_unseen_demos_roc_auc": auc_,
            "primary_metric": top1,
        })
        log.info(
            f"  seen-actors-unseen-demos: "
            f"top1={top1:.4f} top5={top5:.4f} f1={f1_:.4f}"
        )
    else:
        su_acc, _, _ = eval_cls(eval_model, seen_unseen_loader)
        result["seen_actors_unseen_demos_accuracy"] = su_acc
        log.info(f"  seen-actors-unseen-demos acc={su_acc:.4f}")

        if unseen_actors_loader is not None:
            un_acc, _, _ = eval_cls(eval_model, unseen_actors_loader)
            result["unseen_actors_accuracy"] = un_acc
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
# Summary table (shared display logic with ProtoGCN runner)
# ---------------------------------------------------------------------------

def print_summary(
    task: str,
    results: List[Dict],
    output_dir: Path,
    log: logging.Logger,
):
    from ablation.ablation_runner import print_summary as _base_print_summary
    _base_print_summary(task, results, output_dir, log)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="InveRT DS-GCN Ablation Runner")
    p.add_argument("--task", choices=["gender", "reid", "all"], default="all")
    p.add_argument(
        "--ablation",
        choices=["full", "pos_only", "fc_graph", "no_temporal", "all"],
        default="all",
    )
    p.add_argument("--num-actors", type=int, default=100)
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("--splits-dir", type=str, default=None)
    p.add_argument("--g1-cache-dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def setup_logger(log_path: Path, name: str) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(
            logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


def main():
    args = parse_args()

    if args.device:
        import dsgcn_bones_seed as _ref
        _ref.DEVICE = args.device

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else _ABLATION_DIR / "outputs_dsgcn"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    master_log = setup_logger(
        output_dir / "logs" / "dsgcn_ablation_master.log",
        "dsgcn_ablation_master",
    )

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
        master_log.info(
            f"G1 cache: {len(g1_cache_info['index_map']):,} clips"
        )
    else:
        master_log.info("G1 cache not found — falling back to CSV reads.")

    # ── Load manifests ────────────────────────────────────────────────────────
    class _FakeArgs:
        def __init__(self, sd):
            self.splits_dir = sd

    train_df, val_df, test_df = load_manifests(_FakeArgs(splits_dir))
    master_log.info(
        f"Manifests: train={len(train_df):,}  "
        f"val={len(val_df):,}  test={len(test_df):,}"
    )

    # ── Subsample actors ──────────────────────────────────────────────────────
    train_sub, seen_unseen_demos, unseen_actors, n_seen, n_unseen = \
        subsample_actors(
            train_df, val_df, test_df,
            num_actors=args.num_actors,
            test_ratio=0.2,
            seed=args.seed,
        )
    master_log.info(
        f"100-actor subsample: seen={n_seen} unseen={n_unseen} | "
        f"train={len(train_sub):,}  "
        f"seen-unseen-demos={len(seen_unseen_demos):,}  "
        f"unseen-actors={len(unseen_actors):,}"
    )

    # ── Select tasks and ablations ────────────────────────────────────────────
    tasks = ["gender", "reid"] if args.task == "all" else [args.task]
    ablations = (
        list(ALL_ABLATIONS_DSGCN.keys())
        if args.ablation == "all"
        else [args.ablation]
    )

    train_cfg = dict(BASE_TRAIN_CFG_DSGCN)
    all_results: Dict[str, List[Dict]] = {t: [] for t in tasks}

    for task in tasks:
        for abl_name in ablations:
            abl_cfg = ALL_ABLATIONS_DSGCN[abl_name]
            run_log = setup_logger(
                output_dir / "logs" / f"dsgcn_{task}_{abl_name}.log",
                f"dsgcn_{task}_{abl_name}",
            )
            result = run_ablation_dsgcn(
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
            )
            all_results[task].append(result)

    # ── Print summary tables ──────────────────────────────────────────────────
    master_log.info("\n" + "=" * 72)
    master_log.info("DS-GCN ABLATION SUMMARY")
    master_log.info("=" * 72)
    for task in tasks:
        if all_results[task]:
            print_summary(task, all_results[task], output_dir, master_log)

    combined = {
        "model": "dsgcn",
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
