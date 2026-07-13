"""
dataset_g1_paired.py — G1 paired dataset for PMR cross-reconstruction training.

Builds an actor-action index from the manifests and produces 4-tuples:
    x1 = clip(actor_p1, action_a1)
    x2 = clip(actor_p2, action_a2)
    y1 = clip(actor_p1, action_a2)   ← same actor as x1, different action
    y2 = clip(actor_p2, action_a1)   ← same actor as x2, different action

Cross-reconstruction targets:
    D(E_M(x1), E_P(x2)) should look like y2  (p2's style + a1's motion)
    D(E_M(x2), E_P(x1)) should look like y1  (p1's style + a2's motion)

Also supports unpaired mode (for AE warmup / classifier pretrain) which
returns just (x, actor_idx, action_idx) from the full clip pool.
"""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))   # unveil.py lives under src/

from unveil import read_g1_motion, load_g1_cache, g1_cache_fetch

G1_CHANNELS = 35
T_WINDOW    = 256


def _load_clip(rel: str, data_root: str, g1_cache_info: Optional[dict]) -> Optional[np.ndarray]:
    """Load a G1 CSV clip → (T, 35) float32, or None on failure."""
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
            return None
    x = x.astype(np.float32)
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, 0.0)
    return x


def _sample_window(x: np.ndarray) -> np.ndarray:
    """Random T_WINDOW crop; pad short clips."""
    T = x.shape[0]
    if T < T_WINDOW:
        pad = np.zeros((T_WINDOW - T, G1_CHANNELS), dtype=np.float32)
        return np.concatenate([x, pad], axis=0)
    start = np.random.randint(0, T - T_WINDOW + 1)
    return x[start: start + T_WINDOW].copy()


def _normalise(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = (x - mean) / (std + 1e-6)
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, 0.0)
    return x


# ===========================================================================
# Actor-action index
# ===========================================================================

def _build_max_distance_pairs(
    df: pd.DataFrame,
    actor_set: set,
    top_k: int = 5,
) -> set:
    """
    For each actor p, find the top-K actors with the largest Euclidean distance
    in normalised attribute space and record them as valid pairs.

    Attribute vector (per actor):
        [age_zscore, height_zscore, weight_zscore, gender_binary]
    Missing values are imputed with 0 (population mean after z-scoring).

    With top_k=1 each actor gets exactly one partner (the maximally different one).
    With top_k=5 (default) we get more 4-tuples while still being targeted.
    """
    actors = sorted(str(a) for a in actor_set)

    # ── Build per-actor attribute vectors ────────────────────────────────────
    df_a = df.copy()
    df_a["_actor"] = df_a["actor_uid"].astype(str)

    vecs: dict = {}
    for actor in actors:
        rows = df_a[df_a["_actor"] == actor]
        v = []
        for col in ("actor_age_yr", "actor_height_cm", "actor_weight_kg"):
            val = rows[col].mean() if col in rows.columns else float("nan")
            v.append(float(val) if pd.notna(val) else float("nan"))
        if "actor_gender" in df_a.columns:
            mode = rows["actor_gender"].mode()
            v.append(1.0 if (len(mode) > 0 and mode.iloc[0] == "F") else 0.0)
        else:
            v.append(0.5)   # unknown → midpoint
        vecs[actor] = v

    actor_list = sorted(vecs.keys())
    mat = np.array([vecs[a] for a in actor_list], dtype=np.float32)   # (N, 4)

    # ── Z-score continuous dims, leave gender binary ──────────────────────────
    for col_i in range(3):   # age, height, weight only
        col = mat[:, col_i]
        valid = np.isfinite(col)
        if valid.sum() > 1:
            m, s = col[valid].mean(), col[valid].std()
            col[valid] = (col[valid] - m) / max(s, 1e-6)
        col[~valid] = 0.0   # impute missing with population mean
        mat[:, col_i] = col

    # ── Pairwise distance → top-K partners per actor ──────────────────────────
    pairs: set = set()
    N = len(actor_list)
    for i in range(N):
        diff = mat - mat[i]                     # (N, 4) broadcast
        dists = np.sqrt((diff ** 2).sum(axis=1))  # (N,)
        dists[i] = -1.0                         # exclude self
        # Top-K indices by distance (largest first)
        topk_idx = np.argsort(dists)[::-1][:top_k]
        for j in topk_idx:
            if dists[j] > 0:
                pairs.add(frozenset({actor_list[i], actor_list[j]}))

    return pairs


def _build_contrast_pairs(
    df: pd.DataFrame,
    actor_set: set,
    percentile: float = 0.33,
) -> set:
    """
    Return a set of frozenset({p1, p2}) pairs where the two actors sit at
    opposite extremes of at least one demographic attribute.

    Attributes used:
      age    — bottom <percentile> vs top <1-percentile>
      height — bottom <percentile> vs top <1-percentile>
      weight — bottom <percentile> vs top <1-percentile>
      gender — M vs F

    Only actors in actor_set are considered.
    """
    contrasting: set = set()

    for col in ("actor_age_yr", "actor_height_cm", "actor_weight_kg"):
        if col not in df.columns:
            continue
        per_actor = (
            df[df["actor_uid"].astype(str).isin(actor_set)]
            .groupby("actor_uid")[col]
            .mean()
            .dropna()
        )
        if len(per_actor) < 4:
            continue
        lo_thresh = per_actor.quantile(percentile)
        hi_thresh = per_actor.quantile(1.0 - percentile)
        lo_group  = set(per_actor[per_actor <= lo_thresh].index.astype(str))
        hi_group  = set(per_actor[per_actor >= hi_thresh].index.astype(str))
        for p1 in lo_group:
            for p2 in hi_group:
                if p1 != p2:
                    contrasting.add(frozenset({p1, p2}))

    if "actor_gender" in df.columns:
        genders = (
            df[df["actor_uid"].astype(str).isin(actor_set)]
            .groupby("actor_uid")["actor_gender"]
            .first()
            .dropna()
        )
        male   = set(genders[genders == "M"].index.astype(str))
        female = set(genders[genders == "F"].index.astype(str))
        for p1 in male:
            for p2 in female:
                contrasting.add(frozenset({p1, p2}))

    return contrasting


class ActorActionIndex:
    """
    Lookup table: actor_uid × package → [relative clip paths].
    Also pre-computes the list of valid 4-tuples for paired training.

    pairing_strategy:
      "random"             — all valid actor pairs considered (default)
      "attribute_contrast" — only pairs at demographic extremes: young↔old,
                             short↔tall, light↔heavy, M↔F.  Forces E_M to
                             encode motion without demographic cues.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        max_combos: int = 200_000,
        seed: int = 42,
        pairing_strategy: str = "random",
        contrast_percentile: float = 0.33,
        max_dist_topk: int = 5,
    ):
        # Build index
        self.clips: List[dict] = []          # flat list of all clips with metadata
        self.actor_action: Dict[str, Dict[str, List[int]]] = {}  # actor→action→clip_indices

        for _, row in df.iterrows():
            rel = str(row.get("move_g1_mujoco_path", ""))
            if not rel:
                continue
            actor  = str(row.get("actor_uid",  ""))
            action = str(row.get("package",    "unknown"))
            if not actor:
                continue
            idx = len(self.clips)
            self.clips.append({
                "rel_path":   rel,
                "actor":      actor,
                "action":     action,
                "actor_idx":  -1,   # filled below
                "action_idx": -1,
            })
            self.actor_action.setdefault(actor, {}).setdefault(action, []).append(idx)

        # Integer maps
        self.actors  = sorted(self.actor_action.keys())
        self.actions = sorted({c["action"] for c in self.clips})
        self.actor_map  = {a: i for i, a in enumerate(self.actors)}
        self.action_map = {a: i for i, a in enumerate(self.actions)}
        for c in self.clips:
            c["actor_idx"]  = self.actor_map[c["actor"]]
            c["action_idx"] = self.action_map[c["action"]]

        # Pre-compute valid 4-tuples: (actor_p1, actor_p2, action_a1, action_a2)
        # Condition: p1 ≠ p2, a1 ≠ a2, all four (pX, aY) combinations have ≥1 clip
        rng = np.random.default_rng(seed)

        # Build allowed actor-pair set
        if pairing_strategy == "attribute_contrast":
            allowed_pairs = _build_contrast_pairs(
                df, actor_set=set(self.actors), percentile=contrast_percentile
            )
            print(f"  [contrast] {len(allowed_pairs)} contrasting actor pairs "
                  f"(percentile={contrast_percentile:.0%})")
        elif pairing_strategy == "max_attribute_distance":
            allowed_pairs = _build_max_distance_pairs(
                df, actor_set=set(self.actors), top_k=max_dist_topk
            )
            print(f"  [max_dist] {len(allowed_pairs)} max-distance actor pairs "
                  f"(top_k={max_dist_topk})")
        else:
            allowed_pairs = None  # all pairs allowed

        combos: List[Tuple[str, str, str, str]] = []
        actor_list = self.actors

        for p1, p2 in itertools.combinations(actor_list, 2):
            if allowed_pairs is not None and frozenset({p1, p2}) not in allowed_pairs:
                continue
            shared = set(self.actor_action[p1].keys()) & set(self.actor_action[p2].keys())
            if len(shared) < 2:
                continue
            for a1, a2 in itertools.combinations(sorted(shared), 2):
                combos.append((p1, p2, a1, a2))

        if not combos and pairing_strategy in ("attribute_contrast", "max_attribute_distance"):
            print("  [warn] No contrasting 4-tuples found — falling back to random pairing")
            for p1, p2 in itertools.combinations(actor_list, 2):
                shared = set(self.actor_action[p1].keys()) & set(self.actor_action[p2].keys())
                if len(shared) < 2:
                    continue
                for a1, a2 in itertools.combinations(sorted(shared), 2):
                    combos.append((p1, p2, a1, a2))

        if len(combos) > max_combos:
            idx_arr = rng.choice(len(combos), max_combos, replace=False)
            combos = [combos[i] for i in idx_arr]

        self.valid_combos: List[Tuple[str, str, str, str]] = combos
        self.pairing_strategy = pairing_strategy
        print(f"  ActorActionIndex [{pairing_strategy}]: {len(self.clips):,} clips  "
              f"{len(self.actors)} actors  {len(self.actions)} actions  "
              f"{len(self.valid_combos):,} valid 4-tuples")

    def sample_clip_path(self, actor: str, action: str) -> Optional[str]:
        """Pick a random clip for (actor, action)."""
        idxs = self.actor_action.get(actor, {}).get(action)
        if not idxs:
            return None
        i = np.random.choice(idxs)
        return self.clips[i]["rel_path"]


# ===========================================================================
# Datasets
# ===========================================================================

class G1UnpairedDataset(Dataset):
    """
    Returns individual clips: (x, actor_idx, action_idx).
    Used for AE warmup and classifier pretrain stages.
    """

    def __init__(
        self,
        index: ActorActionIndex,
        data_root: str,
        global_mean: np.ndarray,
        global_std: np.ndarray,
        g1_cache_info: Optional[dict] = None,
    ):
        self.index       = index
        self.data_root   = data_root
        self.global_mean = global_mean
        self.global_std  = global_std
        self.cache       = g1_cache_info

    def __len__(self) -> int:
        return len(self.index.clips)

    def __getitem__(self, idx: int):
        rec = self.index.clips[idx]
        x = _load_clip(rec["rel_path"], self.data_root, self.cache)
        if x is None:
            x = np.zeros((T_WINDOW, G1_CHANNELS), dtype=np.float32)
        x = _normalise(_sample_window(x), self.global_mean, self.global_std)
        return (
            torch.from_numpy(x),
            torch.tensor(rec["actor_idx"],  dtype=torch.long),
            torch.tensor(rec["action_idx"], dtype=torch.long),
        )


class G1PairedDataset(Dataset):
    """
    Returns 4-tuples for cross-reconstruction training:
        (x1, x2, y1, y2, actor1_idx, actor2_idx, action1_idx, action2_idx)

    x1 = (p1, a1),  x2 = (p2, a2)
    y1 = (p1, a2),  y2 = (p2, a1)
    """

    def __init__(
        self,
        index: ActorActionIndex,
        data_root: str,
        global_mean: np.ndarray,
        global_std: np.ndarray,
        g1_cache_info: Optional[dict] = None,
    ):
        self.index       = index
        self.data_root   = data_root
        self.global_mean = global_mean
        self.global_std  = global_std
        self.cache       = g1_cache_info

    def __len__(self) -> int:
        return len(self.index.valid_combos)

    def _get(self, actor: str, action: str) -> torch.Tensor:
        path = self.index.sample_clip_path(actor, action)
        if path is None:
            x = np.zeros((T_WINDOW, G1_CHANNELS), dtype=np.float32)
        else:
            raw = _load_clip(path, self.data_root, self.cache)
            x = _normalise(_sample_window(raw if raw is not None
                                          else np.zeros((T_WINDOW, G1_CHANNELS), np.float32)),
                            self.global_mean, self.global_std)
        return torch.from_numpy(x)

    def __getitem__(self, idx: int):
        p1, p2, a1, a2 = self.index.valid_combos[idx]
        x1 = self._get(p1, a1)
        x2 = self._get(p2, a2)
        y1 = self._get(p1, a2)
        y2 = self._get(p2, a1)
        return (
            x1, x2, y1, y2,
            torch.tensor(self.index.actor_map[p1],  dtype=torch.long),
            torch.tensor(self.index.actor_map[p2],  dtype=torch.long),
            torch.tensor(self.index.action_map[a1], dtype=torch.long),
            torch.tensor(self.index.action_map[a2], dtype=torch.long),
        )
