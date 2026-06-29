"""Ablation experiment configurations for InveRT (NeurIPS 2026).

Aggressive ablations designed to produce clearly visible performance drops
(~5–15%) to support the paper's ablation table.

Each variant degrades a DIFFERENT pipeline stage:
  full          — baseline, nothing degraded
  pos_only_v2   — input stage: architecture reduced to in_channels=1 (pos only)
  subset_fc     — spatial stage: 9/35 DoFs (root+waist) + FC graph
  time_averaged — temporal stage: full-clip mean collapsed to T=1 before backbone
"""

from __future__ import annotations
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Shared hyperparameters — identical across all runs
# ---------------------------------------------------------------------------
BASE_TRAIN_CFG: Dict[str, Any] = dict(
    format="g1",
    epochs=50,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    label_smoothing=0.05,
    lambda_supcon=0.1,
    lambda_proto=0.1,
    supcon_warmup=8,
    supcon_temp=0.07,
    early_stop=40,
    eval_every=1,
    target_fps=30,
    max_seq_len=256,
    min_seq_len=16,
    base_channels=96,
    num_stages=10,
    num_prototype=100,
    emb_dim=256,
    dropout=0.5,
    seed=42,
    num_workers=2,
    no_compile=True,
)

# ---------------------------------------------------------------------------
# Aggressive ablation variants
# ---------------------------------------------------------------------------

# A0: Full InveRT — nothing degraded
ABLATION_FULL: Dict[str, Any] = dict(
    ablation="full",
    description="Full InveRT (baseline — all components enabled)",
    model_class="default",
    graph_layout="bones_seed_g1",
    no_temporal=False,
    pos_only=False,
    subset_joints=None,
    time_average_input=False,
)

# A1: W/o motion dynamics — in_channels=1, position stream only
#     Architectural change: backbone truly sees only 1 channel, not zero-padded 3.
ABLATION_POS_ONLY_V2: Dict[str, Any] = dict(
    ablation="pos_only_v2",
    description="W/o motion dynamics: in_channels=1 (position only, no vel/acc)",
    model_class="pos_only",
    graph_layout="bones_seed_g1",
    no_temporal=False,
    pos_only=True,
    subset_joints=None,
    time_average_input=False,
)

# A2: W/o kinematic prior — 20/35 DoFs (root+legs+waist) with fully-connected graph
#     Removes both arms (14 DoFs) + waist_pitch (1 DoF) = 15 removed.
#     Keeps: root(0-5) + left_leg(6-11) + right_leg(12-17) + waist_yaw+roll(18-19).
_SUBSET_JOINTS: List[int] = list(range(20))  # indices 0–19, 20 DoFs

ABLATION_SUBSET_FC: Dict[str, Any] = dict(
    ablation="subset_fc",
    description="W/o kinematic prior: root+legs+waist (20/35 DoFs) + FC graph",
    model_class="subset_fc",
    graph_layout="bones_seed_g1_subset_fc",
    no_temporal=False,
    pos_only=False,
    subset_joints=_SUBSET_JOINTS,
    time_average_input=False,
)

# A3: W/o temporal modeling — temporal mean collapsed to single frame before backbone
#     The model sees exactly one averaged pose; no sequence information whatsoever.
ABLATION_TIME_AVERAGED: Dict[str, Any] = dict(
    ablation="time_averaged",
    description="W/o temporal modeling: input mean-collapsed to T=1 (single avg pose)",
    model_class="no_temporal",
    graph_layout="bones_seed_g1",
    no_temporal=True,
    pos_only=False,
    subset_joints=None,
    time_average_input=True,
)

ALL_ABLATIONS: Dict[str, Dict[str, Any]] = {
    "full":          ABLATION_FULL,
    "pos_only_v2":   ABLATION_POS_ONLY_V2,
    "subset_fc":     ABLATION_SUBSET_FC,
    "time_averaged": ABLATION_TIME_AVERAGED,
}

ABLATION_DISPLAY_NAMES: Dict[str, str] = {
    "full":          "Full InveRT",
    "pos_only_v2":   "W/o motion dynamics",
    "subset_fc":     "W/o kinematic prior",
    "time_averaged": "W/o temporal modeling",
}
