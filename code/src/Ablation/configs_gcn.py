"""Ablation experiment configurations for DS-GCN / InveRT (NeurIPS 2026).

Same 4 ablation variants as ProtoGCN but with DS-GCN-specific hyperparameters:
  - base_channels=64  (vs 96 for ProtoGCN)
  - supcon_warmup=20  (vs 8 for ProtoGCN)
  - no num_prototype / lambda_proto (DS-GCN has no PRN)
"""

from __future__ import annotations
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Shared hyperparameters — identical across all DS-GCN ablation runs
# ---------------------------------------------------------------------------
BASE_TRAIN_CFG_DSGCN: Dict[str, Any] = dict(
    format="g1",
    epochs=10,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    label_smoothing=0.05,
    lambda_supcon=0.1,
    supcon_warmup=20,
    supcon_temp=0.07,
    early_stop=40,
    eval_every=1,
    target_fps=30,
    max_seq_len=256,
    min_seq_len=16,
    base_channels=64,
    num_stages=10,
    emb_dim=256,
    dropout=0.5,
    seed=42,
    num_workers=2,
    no_compile=True,
)

# ---------------------------------------------------------------------------
# Ablation variants (same 4 as ProtoGCN — only graph_layout differs for A2)
# ---------------------------------------------------------------------------

ABLATION_FULL_DSGCN: Dict[str, Any] = dict(
    ablation="full",
    description="Full DS-GCN model (baseline)",
    graph_layout="bones_seed_g1",
    pos_only=False,
    no_temporal=False,
)

ABLATION_POS_ONLY_DSGCN: Dict[str, Any] = dict(
    ablation="pos_only",
    description="Position-only input (vel/acc streams zeroed)",
    graph_layout="bones_seed_g1",
    pos_only=True,
    no_temporal=False,
)

ABLATION_FC_GRAPH_DSGCN: Dict[str, Any] = dict(
    ablation="fc_graph",
    description="Fully-connected graph (no kinematic structure)",
    graph_layout="bones_seed_g1_fc",
    pos_only=False,
    no_temporal=False,
)

ABLATION_NO_TEMPORAL_DSGCN: Dict[str, Any] = dict(
    ablation="no_temporal",
    description="No temporal conv (spatial GCN + AdaptiveAvgPool only)",
    graph_layout="bones_seed_g1",
    pos_only=False,
    no_temporal=True,
)

ALL_ABLATIONS_DSGCN: Dict[str, Dict[str, Any]] = {
    "full":        ABLATION_FULL_DSGCN,
    "pos_only":    ABLATION_POS_ONLY_DSGCN,
    "fc_graph":    ABLATION_FC_GRAPH_DSGCN,
    "no_temporal": ABLATION_NO_TEMPORAL_DSGCN,
}

ABLATION_DISPLAY_NAMES = {
    "full":        "Full Model",
    "pos_only":    "Position-only",
    "fc_graph":    "FC Graph",
    "no_temporal": "No Temporal Conv",
}
