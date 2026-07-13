from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score,
)

# === SECTION: EXTERNAL DEPENDENCY SETUP ===
# UNVEIL itself is self-contained (pure torch). The dsgcn comparison backbone
# wraps the DS-GCN reference implementation bundled under src/DS-GCN via the
# local pyskl stub.

_SCRIPT_DIR = Path(__file__).resolve().parent   # src/ — baseline libs live alongside
_RELEASE_ROOT = _SCRIPT_DIR.parent              # repo root (project_paths.py lives here)
_DYNGRAPH_ROOT = _SCRIPT_DIR / "DS-GCN"

for _p in [str(_RELEASE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from project_paths import DATA_ROOT as PROJECT_ROOT
from project_paths import default_g1_cache_dir, default_splits_dir

# Pre-register the local pyskl stub (sibling of this file) before DS-GCN is added
if str(_SCRIPT_DIR) in sys.path:
    sys.path.remove(str(_SCRIPT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR))

if str(_DYNGRAPH_ROOT) not in sys.path:
    sys.path.insert(1, str(_DYNGRAPH_ROOT))

# The dsgcn baseline (and its mmcv dependency) is optional — UNVEIL itself
# never touches it.
try:
    import pyskl            # noqa: F401 — library path, cannot rename
    import pyskl.utils      # noqa: F401
    import pyskl.models     # noqa: F401
    import pyskl.models.gcns  # noqa: F401
    import pyskl.models.gcns.utils  # noqa: F401
    from pyskl.models.gcns.dgstgcn import DGSTGCN as _DynGraphCore  # noqa
    _HAS_DYNGRAPH = True
except ImportError:
    _DynGraphCore = None  # type: ignore
    _HAS_DYNGRAPH = False


# === SECTION: CONSTANTS ===

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALL_TASKS = ["reid", "gender", "age", "height", "weight"]
REGRESSION_TASKS = {"age", "height", "weight"}
EPS = 1e-4

# Bin widths used to discretize regression targets for SupCon (which needs
# discrete labels for its equality-based positive-pair mask).
SUPCON_BIN_WIDTH = {"age": 5.0, "height": 5.0, "weight": 5.0}

SPATIAL_BACKBONE_DEFAULTS: Dict[str, Dict] = {
    # UNVEIL — the hierarchical spatiotemporal architecture from the paper
    # (kinematic encoding, hierarchical spatial graph convolution, multi-scale
    # temporal aggregation, signature bank). Hyperparameters follow Tab. A.3.
    "unveil": {
        "lr": 1e-3, "batch_size": 32, "supcon_warmup": 8,
        "base_channels": 96, "num_stages": 10, "num_prototype": 100,
        "emb_dim": 256, "lambda_csc": 0.1,
        "variance_percentile": 0.0,
    },
    "sgn": {
        "lr": 3e-4, "batch_size": 64, "supcon_warmup": 20,
        "dim1": 256, "seg": 64, "emb_dim": 256,
        "variance_percentile": 10.0,
    },
    "dsgcn": {
        "lr": 1e-3, "batch_size": 32, "supcon_warmup": 20,
        "base_channels": 64, "num_stages": 10, "emb_dim": 256,
        "variance_percentile": 0.0,
    },
}


# === SECTION: SKELETON DEFINITIONS ===

MAJOR_JOINT_CHANNELS = {
    "Root_rot":     [3, 4, 5],
    "Hips_rot":     [9, 10, 11],
    "Spine1":       [12, 13, 14],
    "Spine2":       [15, 16, 17],
    "Chest":        [18, 19, 20],
    "Neck1":        [21, 22, 23],
    "Neck2":        [24, 25, 26],
    "Head":         [27, 28, 29],
    "LeftShoulder": [42, 43, 44],
    "LeftArm":      [45, 46, 47],
    "LeftForeArm":  [48, 49, 50],
    "LeftHand":     [51, 52, 53],
    "RightShoulder": [126, 127, 128],
    "RightArm":     [129, 130, 131],
    "RightForeArm": [132, 133, 134],
    "RightHand":    [135, 136, 137],
    "LeftLeg":      [210, 211, 212],
    "LeftShin":     [213, 214, 215],
    "LeftFoot":     [216, 217, 218],
    "LeftToeBase":  [219, 220, 221],
    "RightLeg":     [225, 226, 227],
    "RightShin":    [228, 229, 230],
    "RightFoot":    [231, 232, 233],
    "RightToeBase": [234, 235, 236],
}
BVH_NUM_JOINTS = 24
BVH_NUM_CHANNELS = BVH_NUM_JOINTS * 3


# === SECTION: G1 SEMANTIC JOINT LAYOUT (Tab. A.5) ===
# UNVEIL operates on J=15 semantic joints. The 35-channel G1 layout
#   0-5   root: x, y, z, roll, pitch, yaw
#   6-11  left leg : hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
#   12-17 right leg: same as left
#   18-20 waist: yaw, roll, pitch
#   21-27 left arm : shoulder_pitch/roll/yaw, elbow, wrist_yaw/roll/pitch
#   28-34 right arm: same as left
# is grouped into 15 joints; joints with fewer than 3 rotation DoFs are
# zero-padded so every node carries a uniform D=3 input feature.

G1_NUM_CHANNELS = 35
NUM_JOINTS = 15
DOF_PER_JOINT = 3
_PAD = -1

# (name, [channel indices], kinematic group)
G1_JOINTS = [
    ("root_trans", [0, 1, 2],        "T"),   # 0
    ("root_rot",   [3, 4, 5],        "T"),   # 1
    ("waist",      [18, 19, 20],     "T"),   # 2
    ("l_hip",      [6, 7, 8],        "LL"),  # 3
    ("l_knee",     [9, _PAD, _PAD],  "LL"),  # 4
    ("l_ankle",    [10, 11, _PAD],   "LL"),  # 5
    ("r_hip",      [12, 13, 14],     "RL"),  # 6
    ("r_knee",     [15, _PAD, _PAD], "RL"),  # 7
    ("r_ankle",    [16, 17, _PAD],   "RL"),  # 8
    ("l_shoulder", [21, 22, 23],     "LA"),  # 9
    ("l_elbow",    [24, _PAD, _PAD], "LA"),  # 10
    ("l_wrist",    [25, 26, 27],     "LA"),  # 11
    ("r_shoulder", [28, 29, 30],     "RA"),  # 12
    ("r_elbow",    [31, _PAD, _PAD], "RA"),  # 13
    ("r_wrist",    [32, 33, 34],     "RA"),  # 14
]

BODY_GROUPS = {
    "T":  [0, 1, 2],
    "LA": [9, 10, 11],
    "RA": [12, 13, 14],
    "LL": [3, 4, 5],
    "RL": [6, 7, 8],
}

# Skeletal edges of the full kinematic graph G (within-group chains).
GROUP_EDGES = {
    "T":  [(0, 1), (1, 2)],           # root_trans - root_rot - waist
    "LA": [(9, 10), (10, 11)],        # shoulder - elbow - wrist
    "RA": [(12, 13), (13, 14)],
    "LL": [(3, 4), (4, 5)],           # hip - knee - ankle
    "RL": [(6, 7), (7, 8)],
}

# Proximal connecting edge e_k between limb k and the torso (Sec. 3.2):
# legs attach at the pelvis (root_rot), arms attach at the waist.
PROXIMAL_EDGES = {
    "LA": (2, 9),
    "RA": (2, 12),
    "LL": (1, 3),
    "RL": (1, 6),
}

LIMBS = ["LA", "RA", "LL", "RL"]
NUM_SUBGRAPHS = 9  # 5 intra-limb + 4 limb-torso


def build_joint_channel_index() -> np.ndarray:
    """(J, D) channel gather index; pads point at the zero channel (35)."""
    idx = np.zeros((NUM_JOINTS, DOF_PER_JOINT), dtype=np.int64)
    for j, (_, chans, _) in enumerate(G1_JOINTS):
        for d, c in enumerate(chans):
            idx[j, d] = G1_NUM_CHANNELS if c == _PAD else c
    return idx


def _sym_normalize(A: np.ndarray) -> np.ndarray:
    """D^-1/2 A D^-1/2; rows/cols of non-member joints (deg 0) stay zero."""
    deg = A.sum(axis=1)
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(deg > 0, deg ** -0.5, 0.0)
    return (A * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]


def build_subgraphs() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Nine subgraph adjacencies across the two hierarchies (Sec. 3.2).

    Returns:
        A_static : (9, J, J) symmetric-normalized binary adjacencies
                   (self-loops on members; non-members are all-zero)
        masks    : (9, J, J) member outer-product masks
        names    : subgraph names (5 intra-limb + 4 limb-torso)
    """
    adjs, masks, names = [], [], []

    def make(members: List[int], edges: List[Tuple[int, int]], name: str):
        A = np.zeros((NUM_JOINTS, NUM_JOINTS), dtype=np.float32)
        for j in members:
            A[j, j] = 1.0
        for (a, b) in edges:
            A[a, b] = 1.0
            A[b, a] = 1.0
        m = np.zeros(NUM_JOINTS, dtype=np.float32)
        m[members] = 1.0
        adjs.append(_sym_normalize(A))
        masks.append(np.outer(m, m))
        names.append(name)

    # Intra-limb hierarchy H1: torso + four limbs, isolated
    for g in ["T", "LA", "RA", "LL", "RL"]:
        make(BODY_GROUPS[g], GROUP_EDGES[g], f"intra:{g}")

    # Limb-torso hierarchy H2: limb + torso + proximal connecting edge
    for g in LIMBS:
        members = BODY_GROUPS[g] + BODY_GROUPS["T"]
        edges = GROUP_EDGES[g] + GROUP_EDGES["T"] + [PROXIMAL_EDGES[g]]
        make(members, edges, f"limb-torso:{g}")

    return np.stack(adjs), np.stack(masks), names


# === SECTION: SHARED UTILITIES ===

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_bvh_motion(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    motion_start = None
    for i, line in enumerate(lines):
        if line.strip() == "MOTION":
            motion_start = i
            break
    if motion_start is None:
        raise ValueError(f"MOTION section not found: {path}")
    num_frames = int(lines[motion_start + 1].split(":", 1)[1].strip())
    frame_lines = lines[motion_start + 3: motion_start + 3 + num_frames]
    return np.array(
        [[float(x) for x in ln.strip().split()] for ln in frame_lines],
        dtype=np.float32,
    )


def read_g1_motion(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    if "Frame" in df.columns:
        df = df.drop(columns=["Frame"])
    return df.to_numpy(dtype=np.float32)


def read_motion(path: str, fmt: str) -> np.ndarray:
    return read_g1_motion(path) if fmt == "g1" else read_bvh_motion(path)


def norm_relpath(path: str) -> str:
    return str(path).replace("\\", "/").strip().lower()


def load_g1_cache(cache_dir: str) -> Optional[Dict]:
    meta_path = os.path.join(cache_dir, "metadata.json")
    index_path = os.path.join(cache_dir, "motion_index.csv")
    data_path = os.path.join(cache_dir, "motion_data.f32")
    if not all(os.path.exists(p) for p in [meta_path, index_path, data_path]):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    idx_df = pd.read_csv(index_path)
    if not {"path", "offset", "length"}.issubset(set(idx_df.columns)):
        raise RuntimeError(f"Invalid G1 cache schema in {index_path}")
    index_map = {
        norm_relpath(str(r["path"])): (int(r["offset"]), int(r["length"]))
        for _, r in idx_df.iterrows()
    }
    return {
        "data_path": data_path,
        "index_map": index_map,
        "total_frames": int(meta["total_frames"]),
        "num_channels": int(meta["num_channels"]),
    }


def g1_cache_fetch(cache_info: Dict, rel_path: str, memmap_obj: Optional[np.memmap]):
    key = norm_relpath(rel_path)
    pos = cache_info["index_map"].get(key)
    if pos is None:
        return None, memmap_obj
    if memmap_obj is None:
        memmap_obj = np.memmap(
            cache_info["data_path"], dtype=np.float32, mode="r",
            shape=(cache_info["total_frames"], cache_info["num_channels"]),
        )
    off, length = pos
    return np.array(memmap_obj[off: off + length], dtype=np.float32, copy=True), memmap_obj


# === SECTION: CHANNEL SELECTION ===

def get_bvh_channel_indices() -> np.ndarray:
    indices = []
    for chs in MAJOR_JOINT_CHANNELS.values():
        indices.extend(chs)
    return np.array(sorted(indices), dtype=np.int64)


def refine_bvh_channels_by_variance(
    train_df: pd.DataFrame, data_root: str, fmt: str,
    base_indices: np.ndarray, variance_percentile: float,
    max_samples: int = 800, seed: int = 42,
) -> np.ndarray:
    if variance_percentile <= 0:
        return base_indices
    path_col = get_format_path_col(fmt)
    if path_col not in train_df.columns:
        return base_indices
    sample_df = train_df.sample(n=min(max_samples, len(train_df)), random_state=seed)
    var_rows = []
    for _, row in sample_df.iterrows():
        fp = os.path.join(data_root, str(row[path_col]))
        if not os.path.exists(fp):
            continue
        try:
            x = read_bvh_motion(fp)[:, base_indices]
            if x.shape[0] >= 2:
                var_rows.append(np.var(x, axis=0))
        except Exception:
            continue
    if not var_rows:
        return base_indices
    avg_var = np.mean(np.stack(var_rows, axis=0), axis=0)
    threshold = np.percentile(avg_var, variance_percentile)
    keep_mask = avg_var > threshold
    if int(keep_mask.sum()) < 12:
        top_idx = np.argsort(avg_var)[-min(len(avg_var), 24):]
        keep_mask = np.zeros_like(avg_var, dtype=bool)
        keep_mask[top_idx] = True
    return base_indices[keep_mask]


def get_format_path_col(fmt: str) -> str:
    return {
        "g1": "move_g1_mujoco_path",
        "uniform": "move_soma_uniform_path",
        "proportional": "move_soma_proportional_path",
    }[fmt]


# === SECTION: DATA SPLITTING ===

def load_manifests(args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = os.path.join(args.splits_dir, "train_manifest.csv")
    val_path = os.path.join(args.splits_dir, "val_manifest.csv")
    test_path = os.path.join(args.splits_dir, "test_manifest.csv")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Missing split manifests in {args.splits_dir}. Run Data_split/create_splits.py first."
        )
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path) if os.path.exists(val_path) else pd.DataFrame()
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df


def user_task_split(df_all, test_ratio, seed, min_motions, task_key="package"):
    rng = np.random.default_rng(seed)
    originals = df_all[df_all["is_mirror"] == False].copy()
    actor_counts = originals.groupby("actor_uid").size()
    eligible = actor_counts[actor_counts >= min_motions].index.tolist()
    originals = originals[originals["actor_uid"].isin(eligible)]
    train_parts, test_parts = [], []
    for actor_uid in eligible:
        actor_df = originals[originals["actor_uid"] == actor_uid]
        for _, task_group in actor_df.groupby(task_key):
            group = task_group.copy()
            n = len(group)
            if n < 2:
                train_parts.append(group)
                continue
            idx = np.arange(n)
            rng.shuffle(idx)
            n_test = max(1, min(int(round(test_ratio * n)), n - 1))
            test_parts.append(group.iloc[idx[:n_test]])
            train_parts.append(group.iloc[idx[n_test:]])
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    train_keys = set(train_df["move_name"].tolist())
    mirrors = df_all[(df_all["is_mirror"] == True) & df_all["actor_uid"].isin(eligible)].copy()
    mirrors["_orig"] = mirrors["move_name"].map(lambda n: n[:-2] if n.endswith("_M") else n)
    train_df = pd.concat(
        [train_df, mirrors[mirrors["_orig"].isin(train_keys)].drop(columns=["_orig"])],
        ignore_index=True,
    )
    return train_df, test_df


def split_seen_val_per_actor(seen_val_df: pd.DataFrame, train_ratio: float = 0.8, seed: int = 42):
    if seen_val_df.empty:
        empty = pd.DataFrame(columns=seen_val_df.columns)
        return empty, empty
    rng = np.random.default_rng(seed)
    train_parts, eval_parts = [], []
    for _, group in seen_val_df.groupby("actor_uid"):
        group = group.reset_index(drop=True)
        n = len(group)
        idx = rng.permutation(n)
        n_train = max(1, min(int(round(train_ratio * n)), n - 1))
        train_parts.append(group.iloc[idx[:n_train]])
        eval_parts.append(group.iloc[idx[n_train:]])
    return pd.concat(train_parts, ignore_index=True), pd.concat(eval_parts, ignore_index=True)


def prepare_data(args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if args.split_mode == "user":
        train_df, val_df, test_df = load_manifests(args)
        train_names = set(train_df["move_name"].astype(str))
        val_names = set(val_df["move_name"].astype(str)) if not val_df.empty else set()
        test_names = set(test_df["move_name"].astype(str))
        if (train_names & test_names) or (val_names & test_names):
            raise RuntimeError("Split leakage detected in manifests.")
        print(f"Phase 2 split: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")
    else:
        meta_path = os.path.join(args.data_root, "metadata", "seed_metadata_v003.parquet")
        df_all = pd.read_parquet(meta_path)
        train_df, test_df = user_task_split(
            df_all, args.test_ratio, args.seed, args.min_motions, args.deconfound_key
        )
        val_df = pd.DataFrame()
        # Note: in user_task mode the held-out split contains unseen *demos*
        # of seen actors; downstream logs label it "unseen" for uniformity.
        print("Using regenerated user_task split.")
    if args.max_test > 0 and len(test_df) > args.max_test:
        test_df = test_df.sample(n=args.max_test, random_state=args.seed)
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


# === SECTION: DATASET ===

class BonesSeedDataset(Dataset):
    def __init__(
        self, df, data_root, fmt, label_col, label_map, is_regression,
        task_col, task_map, channel_indices, downsample_factor,
        max_seq_len, min_seq_len, global_mean, global_std,
        train, seed, g1_cache_info=None,
        aug_noise_std=0.01, aug_time_mask_frac=0.05,
    ):
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.aug_noise_std = aug_noise_std
        self.aug_time_mask_frac = aug_time_mask_frac

        path_col = get_format_path_col(fmt)
        valid_mask = (
            df[label_col].notna() & df[path_col].notna() if is_regression
            else df[label_col].isin(label_map) & df[path_col].notna()
        )
        df_valid = df[valid_mask].reset_index(drop=True)

        streams_list: List[np.ndarray] = []
        labels_list: List = []
        tasks_list: List[int] = []
        g1_mm = None
        n_skip = 0
        t0 = time.time()

        for i in range(len(df_valid)):
            row = df_valid.iloc[i]
            rel_path = str(row[path_col])
            filepath = os.path.join(data_root, rel_path)
            try:
                if fmt == "g1" and g1_cache_info is not None:
                    x, g1_mm = g1_cache_fetch(g1_cache_info, rel_path, g1_mm)
                    if x is None:
                        x = read_motion(filepath, fmt)
                else:
                    x = read_motion(filepath, fmt)
            except Exception:
                n_skip += 1
                continue

            if not np.isfinite(x).all():
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                if not np.isfinite(x).all():
                    n_skip += 1
                    continue

            if channel_indices is not None:
                x = x[:, channel_indices]
            if downsample_factor > 1:
                x = x[::downsample_factor]
            if x.shape[0] < min_seq_len:
                repeats = int(np.ceil(min_seq_len / max(1, x.shape[0])))
                x = np.tile(x, (repeats, 1))[:min_seq_len]
            if not train and max_seq_len > 0 and x.shape[0] > max_seq_len:
                start = (x.shape[0] - max_seq_len) // 2
                x = x[start: start + max_seq_len]

            x = x - x.mean(axis=0, keepdims=True)
            if global_mean is not None and global_std is not None:
                x = (x - global_mean[None, :]) / (global_std[None, :] + 1e-6)
            if not np.isfinite(x).all():
                n_skip += 1
                continue

            pos = x.T.astype(np.float32)
            vel = np.zeros_like(pos); vel[:, 1:] = pos[:, 1:] - pos[:, :-1]
            acc = np.zeros_like(pos); acc[:, 2:] = vel[:, 2:] - vel[:, 1:-1]
            streams = np.stack([pos, vel, acc], axis=0)  # (3, C, T)

            if is_regression:
                label = float(row[label_col])
                if not np.isfinite(label):
                    n_skip += 1
                    continue
            else:
                label = label_map[row[label_col]]
            task_val = str(row.get(task_col, "unknown"))
            task_idx = task_map.get(task_val, 0)
            streams_list.append(streams)
            labels_list.append(label)
            tasks_list.append(task_idx)

        if n_skip > 0:
            print(f"  [preload] skipped {n_skip} clips")
        print(f"  [preload] {len(streams_list):,} clips in {time.time()-t0:.1f}s")

        self.streams = streams_list
        self.is_regression = is_regression
        self.labels = torch.tensor(
            labels_list, dtype=torch.float32 if is_regression else torch.long
        )
        self.task_ids = torch.tensor(tasks_list, dtype=torch.long)

    def __len__(self):
        return len(self.streams)

    def __getitem__(self, idx: int):
        streams = self.streams[idx]
        T = streams.shape[2]
        if T > self.max_seq_len:
            start = (int(self.rng.integers(0, T - self.max_seq_len + 1))
                     if self.train else (T - self.max_seq_len) // 2)
            streams = streams[:, :, start: start + self.max_seq_len]
            T = self.max_seq_len
        if self.train:
            streams = streams.copy()
            if self.aug_noise_std > 0:
                streams = streams + (self.aug_noise_std *
                                     self.rng.standard_normal(streams.shape)).astype(np.float32)
            if self.aug_time_mask_frac > 0:
                m = max(1, int(round(self.aug_time_mask_frac * T)))
                streams[:, :, self.rng.choice(T, m, replace=False)] = 0.0
        return streams, self.labels[idx], self.task_ids[idx]


def collate_padded(samples):
    xs, ys, tids = zip(*samples)
    B = len(xs)
    y_tensor = torch.stack(list(ys))
    tid_tensor = torch.stack(list(tids))
    lengths_list = [x.shape[2] for x in xs]
    T_max = max(lengths_list)
    if all(t == T_max for t in lengths_list):
        Xb = torch.from_numpy(np.stack(xs, axis=0))
        lengths = torch.full((B,), T_max, dtype=torch.long)
    else:
        S, C = xs[0].shape[0], xs[0].shape[1]
        Xb = torch.zeros(B, S, C, T_max)
        lengths = torch.zeros(B, dtype=torch.long)
        for i, x in enumerate(xs):
            t = x.shape[2]
            Xb[i, :, :, :t] = torch.from_numpy(x)
            lengths[i] = t
    return Xb, lengths, y_tensor, tid_tensor


def compute_global_norm(df, data_root, fmt, channel_indices, downsample_factor,
                        max_samples=5000, seed=42, g1_cache_info=None):
    path_col = get_format_path_col(fmt)
    sample_df = df.sample(n=min(max_samples, len(df)), random_state=seed)
    all_vals = []
    g1_mem = None
    for _, row in sample_df.iterrows():
        rel_path = str(row[path_col])
        fp = os.path.join(data_root, rel_path)
        if not os.path.exists(fp):
            continue
        try:
            if fmt == "g1" and g1_cache_info is not None:
                x, g1_mem = g1_cache_fetch(g1_cache_info, rel_path, g1_mem)
                if x is None:
                    x = read_motion(fp, fmt)
            else:
                x = read_motion(fp, fmt)
            if channel_indices is not None:
                x = x[:, channel_indices]
            if downsample_factor > 1:
                x = x[::downsample_factor]
            x = x - x.mean(axis=0, keepdims=True)
            if np.isfinite(x).all():
                all_vals.append(x)
        except Exception:
            continue
    if not all_vals:
        raise RuntimeError("No valid files found for normalization.")
    big = np.concatenate(all_vals, axis=0)
    return big.mean(axis=0).astype(np.float32), np.maximum(big.std(axis=0), 1e-6).astype(np.float32)


# === SECTION: LOSSES ===

def _supcon_labels(yb: torch.Tensor, task: str) -> torch.Tensor:
    """Map regression targets to bin ids so SupCon can find positive pairs.
    Classification labels pass through unchanged."""
    w = SUPCON_BIN_WIDTH.get(task)
    if w is None:
        return yb
    return torch.floor(yb.float() / w).long()


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        if B < 2:
            return z.new_zeros(1).squeeze()
        sim = z @ z.T / self.temp
        labels_col = labels.view(-1, 1)
        pos_mask = (labels_col == labels_col.T).float()
        pos_mask.fill_diagonal_(0.0)
        has_pos = pos_mask.sum(1) > 0
        if has_pos.sum() == 0:
            return z.new_zeros(1).squeeze()
        sim_max, _ = sim.detach().max(dim=1, keepdim=True)
        sim = sim - sim_max
        self_mask = torch.eye(B, device=z.device).bool()
        exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)
        loss = -(pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-12)
        return loss[has_pos].mean()


class SignatureContrastiveLoss(nn.Module):
    """Operator-contrastive loss L_csc on the distilled signature (Eq. A.4-A.5).

    Prototypical InfoNCE over a momentum-updated class-prototype bank: each
    class keeps a running prototype of its members' signatures, and every
    sample is pulled toward its own class prototype and pushed from the rest.
    """

    def __init__(self, n_class: int, in_dim: int, h_dim: int = 256,
                 temperature: float = 0.125, momentum: float = 0.9):
        super().__init__()
        self.fc = nn.Linear(in_dim, h_dim)
        self.tau = temperature
        self.mom = momentum
        self.n_class = n_class
        self.register_buffer(
            "bank", F.normalize(torch.randn(n_class, h_dim), dim=1)
        )

    def forward(self, signature: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        f = F.normalize(self.fc(signature), dim=1)          # (B, H)
        labels = labels.long().view(-1)

        # Momentum update of class prototypes from batch means (Eq. A.4)
        with torch.no_grad():
            f_det = f.detach()
            onehot = torch.zeros(
                f.size(0), self.n_class, device=f.device, dtype=f.dtype
            )
            onehot.scatter_(1, labels.unsqueeze(1), 1.0)
            counts = onehot.sum(dim=0)                       # (K,)
            sums = onehot.t() @ f_det                        # (K, H)
            present = counts > 0
            means = sums[present] / counts[present].unsqueeze(1)
            self.bank[present] = (
                self.mom * self.bank[present] + (1.0 - self.mom) * means
            )

        logits = f @ F.normalize(self.bank, dim=1).t() / self.tau  # (B, K)
        return F.cross_entropy(logits, labels)


# === SECTION: SGN BACKBONE ===

class _NormData(nn.Module):
    def __init__(self, dim: int, num_joint: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim * num_joint)

    def forward(self, x):
        B, C, J, T = x.shape
        return self.bn(x.view(B, C * J, T)).view(B, C, J, T).contiguous()


class _Conv1x1(nn.Module):
    def __init__(self, dim1, dim2, bias=True):
        super().__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.cnn(x)


class _Embed(nn.Module):
    def __init__(self, dim, dim1, num_joint, norm=True, bias=False):
        super().__init__()
        if norm:
            self.cnn = nn.Sequential(
                _NormData(dim, num_joint),
                _Conv1x1(dim, 64, bias=bias), nn.ReLU(),
                _Conv1x1(64, dim1, bias=bias), nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                _Conv1x1(dim, 64, bias=bias), nn.ReLU(),
                _Conv1x1(64, dim1, bias=bias), nn.ReLU(),
            )

    def forward(self, x):
        return self.cnn(x)


class GraphConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature, bias=False):
        super().__init__()
        self.bn = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU()
        self.w = _Conv1x1(in_feature, out_feature, bias=False)
        self.w1 = _Conv1x1(in_feature, out_feature, bias=bias)

    def forward(self, x1, g):
        x = x1.permute(0, 3, 2, 1).contiguous()
        x = g.matmul(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        return self.relu(self.bn(self.w(x) + self.w1(x1)))


class LearnedAdjacency(nn.Module):
    def __init__(self, dim1, dim2, bias=False):
        super().__init__()
        self.g1 = _Conv1x1(dim1, dim2, bias=bias)
        self.g2 = _Conv1x1(dim1, dim2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1):
        g1 = self.g1(x1).permute(0, 3, 2, 1).contiguous()
        g2 = self.g2(x1).permute(0, 3, 1, 2).contiguous()
        return self.softmax(g1.matmul(g2))


class TemporalBlock(nn.Module):
    def __init__(self, dim1, dim2, seg, bias=False):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d((1, seg))
        self.cnn1 = nn.Conv2d(dim1, dim1, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.pool(x)
        x = self.relu(self.bn1(self.cnn1(x)))
        x = self.dropout(x)
        return self.relu(self.bn2(self.cnn2(x)))


class AttentionPool(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.W = nn.Linear(in_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(2)
        x_t = x.permute(0, 2, 1)
        a = torch.softmax(self.v(torch.tanh(self.W(x_t))), dim=1)
        return (a * x_t).sum(dim=1)


class StreamAttnBackbone(nn.Module):
    """Three-stream input backbone with learned adjacency and attention pooling."""

    def __init__(self, num_classes, num_joint, dim1=256, seg=64, emb_dim=256, bias=True):
        super().__init__()
        self.dim1 = dim1
        self.seg = seg
        self.num_joint = num_joint
        self.num_classes = num_classes

        self.tem_embed = _Embed(seg, 64 * 4, num_joint, norm=False, bias=bias)
        self.spa_embed = _Embed(num_joint, 64, num_joint, norm=False, bias=bias)
        self.joint_embed = _Embed(1, 64, num_joint, norm=True, bias=bias)
        self.dif_embed = _Embed(1, 64, num_joint, norm=True, bias=bias)
        self.acc_embed = _Embed(1, 64, num_joint, norm=True, bias=bias)

        self.stream_fuse = nn.Sequential(
            _Conv1x1(256, dim1 // 2, bias=bias),
            nn.BatchNorm2d(dim1 // 2), nn.ReLU(inplace=True),
        )
        self.compute_adj = LearnedAdjacency(dim1 // 2, dim1, bias=bias)
        self.gcn1 = GraphConvBlock(dim1 // 2, dim1 // 2, bias=bias)
        self.gcn2 = GraphConvBlock(dim1 // 2, dim1, bias=bias)
        self.gcn3 = GraphConvBlock(dim1, dim1, bias=bias)
        self.cnn = TemporalBlock(dim1, dim1 * 2, seg, bias=bias)
        self.attn_pool = AttentionPool(dim1 * 2, 128)
        self.fc = nn.Linear(dim1 * 2, num_classes)
        self.z_proj = nn.Sequential(nn.Linear(dim1 * 2, emb_dim), nn.LayerNorm(emb_dim))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
        nn.init.constant_(self.gcn1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn2.w.cnn.weight, 0)
        nn.init.constant_(self.gcn3.w.cnn.weight, 0)

        spa_eye = torch.eye(num_joint).unsqueeze(0).unsqueeze(0)
        self.register_buffer("spa_oh_base", spa_eye.permute(0, 3, 2, 1))
        tem_eye = torch.eye(seg).unsqueeze(0).unsqueeze(0)
        self.register_buffer("tem_oh_base", tem_eye.permute(0, 3, 1, 2))

    def forward(self, x, lengths=None):
        B, S, C, T = x.shape
        J = C
        pos = x[:, 0:1]; vel = x[:, 1:2]; acc = x[:, 2:3]
        spa_oh = self.spa_oh_base.expand(B, -1, -1, self.seg)
        tem_oh = self.tem_oh_base.expand(B, -1, J, -1)
        pos_feat = self.joint_embed(pos)
        vel_feat = self.dif_embed(vel)
        acc_feat = self.acc_embed(acc)
        tem1 = self.tem_embed(tem_oh)
        spa1 = F.adaptive_max_pool2d(self.spa_embed(spa_oh), (J, T))
        dy = torch.cat([pos_feat, vel_feat, acc_feat, spa1], dim=1)
        inp2 = self.stream_fuse(dy)
        g = self.compute_adj(inp2)
        inp2 = self.gcn1(inp2, g)
        inp2 = self.gcn2(inp2, g)
        inp2 = self.gcn3(inp2, g)
        tem1 = F.adaptive_max_pool2d(tem1, (J, inp2.shape[-1]))
        inp2 = inp2 + tem1
        inp2 = self.cnn(inp2)
        feat = self.attn_pool(inp2)
        logits = self.fc(feat)
        z = F.normalize(self.z_proj(feat), dim=-1)
        return logits, z


# === SECTION: DSGCN BACKBONE ===

class DynGraphBackbone(nn.Module):
    """Wraps the dynamic-adjacency spatiotemporal network for BONES-SEED."""

    def __init__(self, fmt, num_classes, emb_dim=256, base_channels=64,
                 num_stages=10, dropout=0.5):
        super().__init__()
        if not _HAS_DYNGRAPH:
            raise ImportError(f"dsgcn dependency not found at {_DYNGRAPH_ROOT}")
        self.fmt = fmt
        self.num_classes = num_classes
        if fmt == "g1":
            layout, in_channels, self.num_joints = "bones_seed_g1", 3, 35
        else:
            layout, in_channels, self.num_joints = "bones_seed_bvh", 9, 24
        graph_cfg = dict(layout=layout, mode="spatial")
        self._out_channels = int(base_channels * (2 ** 2))
        self.backbone = _DynGraphCore(  # type: ignore
            graph_cfg=graph_cfg, in_channels=in_channels,
            base_channels=base_channels, ch_ratio=2, num_stages=num_stages,
            inflate_stages=[5, 8], down_stages=[5, 8],
            data_bn_type="VC", num_person=1,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.fc = nn.Linear(self._out_channels, num_classes)
        self.z_proj = nn.Sequential(nn.Linear(self._out_channels, emb_dim), nn.LayerNorm(emb_dim))

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C, T = x.shape
        if self.fmt == "g1":
            x = x.permute(0, 3, 2, 1).unsqueeze(1)
        else:
            x = x.permute(0, 3, 2, 1)
            x = x.reshape(B, T, self.num_joints, 3, S).reshape(B, T, self.num_joints, 9).unsqueeze(1)
        return x.contiguous()

    def forward(self, x, lengths=None):
        x = self._reshape_input(x)
        feat = self.backbone(x)
        N, M, C_out, T_out, V = feat.shape
        feat = self.pool(feat.reshape(N * M, C_out, T_out, V)).reshape(N, M, C_out).mean(1)
        if self.dropout is not None:
            feat = self.dropout(feat)
        logits = self.fc(feat)
        z = F.normalize(self.z_proj(feat), dim=-1)
        return logits, z


# === SECTION: UNVEIL BACKBONE ===
# The UNVEIL architecture (Sec. 3): kinematic encoding lifts per-joint
# pos/vel/acc streams into the backbone width (Sec. 3.1); a hierarchical
# spatial graph convolution propagates features over nine kinematic subgraphs
# spanning the intra-limb and limb-torso hierarchies (Sec. 3.2, Eq. 1-2);
# multi-scale dilated temporal convolutions aggregate frames (Sec. 3.3); a
# signature bank distills the final layer's motion-adaptive topology into the
# operator signature that drives the operator-contrastive loss (Sec. 3.4).


class KinematicEncoder(nn.Module):
    """Learned per-joint projection of pos/vel/acc streams into C_in features
    (Sec. 3.1).

    Input : (N, S=3, J, D=3, T)   [S streams: position, velocity, acceleration]
    Output: (N, C_out, T, J)
    """

    def __init__(self, num_joints: int = NUM_JOINTS, dof: int = DOF_PER_JOINT,
                 streams: int = 3, out_channels: int = 96):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(streams, num_joints, dof, out_channels)
        )
        self.bias = nn.Parameter(torch.zeros(num_joints, out_channels))
        nn.init.normal_(self.weight, std=math.sqrt(2.0 / (streams * dof)))
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N,S,J,D,T) x (S,J,D,C) -> (N,C,T,J)
        h = torch.einsum("nsjdt,sjdc->nctj", x, self.weight)
        h = h + self.bias.t()[None, :, None, :]
        return self.act(self.bn(h))


class HierarchicalGCN(nn.Module):
    """Hierarchical spatial graph convolution over the nine kinematic
    subgraphs (Sec. 3.2, Eq. 1-2, A.1-A.2).

    Each subgraph k gets its own projection (a slice of mid channels), its own
    static normalized adjacency, and its own trajectory-specific correlation /
    contrast adjacencies with learnable scales alpha_k / beta_k (init 0).
    Aggregation is masked to subgraph members; outputs sum over subgraphs.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 A_static: torch.Tensor, masks: torch.Tensor,
                 ratio: float = 0.125):
        super().__init__()
        K = A_static.shape[0]
        self.num_subgraphs = K
        mid = int(ratio * out_channels)
        self.mid_channels = mid

        self.register_buffer("A_static", A_static)   # (K, V, V)
        self.register_buffer("mask", masks)           # (K, V, V)

        # Per-subgraph feature projection W_k (grouped into one conv)
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, K * mid, 1),
            nn.BatchNorm2d(K * mid),
            nn.ReLU(),
        )
        # Query / key streams Phi_Q, Phi_K (per subgraph)
        self.conv_q = nn.Conv2d(in_channels, K * mid, 1)
        self.conv_k = nn.Conv2d(in_channels, K * mid, 1)

        # Per-subgraph scales, init 0 -> early training reduces to the static
        # skeleton (App. A.2)
        self.alpha = nn.Parameter(torch.zeros(K))
        self.beta = nn.Parameter(torch.zeros(K))

        self.post = nn.Conv2d(K * mid, out_channels, 1)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.down = nn.Identity()
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (N, C, T, V)
        n, _, t, v = x.shape
        K, mid = self.num_subgraphs, self.mid_channels
        res = self.down(x)

        pre = self.pre(x).view(n, K, mid, t, v)

        # Trajectory-level query/key: time-average over the full clip (App. A.2)
        q = self.conv_q(x).view(n, K, mid, t, v).mean(dim=3)   # (N,K,mid,V)
        k = self.conv_k(x).view(n, K, mid, t, v).mean(dim=3)   # (N,K,mid,V)

        # Correlation adjacency (Eq. A.1): softmax(QK^T), masked to members
        scores = torch.einsum("nkcv,nkcw->nkvw", q, k)          # (N,K,V,V)
        scores = scores.masked_fill(self.mask[None] == 0, -1e9)
        cor = torch.softmax(scores, dim=-2) * self.mask[None]   # (N,K,V,V)

        # Contrast adjacency (Eq. A.2): channel-wise tanh(Q [-] K), masked
        con = torch.tanh(q.unsqueeze(-1) - k.unsqueeze(-2))     # (N,K,mid,V,V)
        con = con * self.mask[None, :, None]

        # Composite adjacency (Eq. 1); learned part feeds the signature bank
        learned = (
            self.alpha.view(1, K, 1, 1, 1) * cor.unsqueeze(2)
            + self.beta.view(1, K, 1, 1, 1) * con
        )                                                        # (N,K,mid,V,V)
        A = self.A_static[None, :, None] + learned

        # Spatial aggregation per subgraph, then sum via the fusing 1x1 conv
        out = torch.einsum("nkctv,nkcvw->nkctw", pre, A).contiguous()
        out = out.reshape(n, K * mid, t, v)
        out = self.post(out)

        topology = learned.reshape(n, K * mid, v, v)
        return self.act(self.bn(out) + res), topology


class MultiScaleTCN(nn.Module):
    """Multi-scale temporal aggregation (Sec. 3.3, Tab. A.3): branches
    {(3,d=1),(3,d=2),(3,d=3),(3,d=4), maxpool3, conv1x1}, concatenated and
    fused by a BatchNorm-ReLU-Conv1x1 transform."""

    DILATIONS = (1, 2, 3, 4)

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        num_branches = len(self.DILATIONS) + 2
        assert out_channels % num_branches == 0, \
            f"out_channels {out_channels} not divisible by {num_branches}"
        bc = out_channels // num_branches

        branches = []
        for d in self.DILATIONS:
            branches.append(nn.Sequential(
                nn.Conv2d(in_channels, bc, 1),
                nn.BatchNorm2d(bc),
                nn.ReLU(),
                nn.Conv2d(bc, bc, (3, 1), stride=(stride, 1),
                          padding=(d, 0), dilation=(d, 1)),
                nn.BatchNorm2d(bc),
            ))
        branches.append(nn.Sequential(
            nn.Conv2d(in_channels, bc, 1),
            nn.BatchNorm2d(bc),
            nn.ReLU(),
            nn.MaxPool2d((3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(bc),
        ))
        branches.append(nn.Sequential(
            nn.Conv2d(in_channels, bc, 1, stride=(stride, 1)),
            nn.BatchNorm2d(bc),
        ))
        self.branches = nn.ModuleList(branches)
        self.fuse = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([b(x) for b in self.branches], dim=1)
        return self.fuse(y)


class STBlock(nn.Module):
    """Hierarchical spatial GCN + multi-scale TCN with residual connection."""

    def __init__(self, in_channels: int, out_channels: int,
                 A_static: torch.Tensor, masks: torch.Tensor, stride: int = 1):
        super().__init__()
        self.gcn = HierarchicalGCN(in_channels, out_channels, A_static, masks)
        self.tcn = MultiScaleTCN(out_channels, out_channels, stride=stride)
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.residual(x)
        y, topology = self.gcn(x)
        y = self.act(self.tcn(y) + res)
        return y, topology


class SignatureBank(nn.Module):
    """Signature bank (Sec. 3.4, Eq. 3 / A.3): learnable prototypes
    reconstruct the joint-pair descriptors of the learned topology; a light
    conv + channel pooling yields the distilled operator signature g_hat
    (one scalar per joint pair)."""

    def __init__(self, dim: int, n_prototypes: int = 100, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Linear(dim, n_prototypes, bias=False)   # W_q
        self.memory = nn.Linear(n_prototypes, dim, bias=False)  # W_m
        self.dropout = nn.Dropout(dropout)
        self.post = nn.Conv2d(dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.ReLU()

    def forward(self, topology: torch.Tensor) -> torch.Tensor:
        # topology: (N, G, V, V) -> descriptors g: (N, V*V, G)
        n, g_c, v, _ = topology.shape
        g = topology.reshape(n, g_c, v * v).transpose(1, 2)
        r = torch.softmax(self.query(g), dim=-1)   # prototype assignment weights (Eq. A.3)
        g_hat = self.dropout(self.memory(r))       # reconstruction
        g_hat = g_hat.transpose(1, 2).reshape(n, g_c, v, v)
        g_hat = self.act(self.bn(self.post(g_hat)))
        return g_hat.mean(dim=1).reshape(n, v * v)  # (N, V*V)


class UnveilBackbone(nn.Module):
    """The full UNVEIL network on the G1 semantic-joint graph (Tab. A.2):
    kinematic encoding (layer 1) + spatiotemporal layers with channel
    inflation and temporal stride 2 at layers 5 and 8, followed by the
    signature bank and the task heads.

    Input : (B, 3, 35, T)  — pos/vel/acc streams over the 35 G1 channels
    Output: (logits, z, signature)
      logits    : (B, num_classes)     task head
      z         : (B, emb_dim)         normalized trajectory embedding
      signature : (B, V*V)             distilled signature for L_csc
    """

    _INFLATE_STAGES = (5, 8)
    _DOWN_STAGES = (5, 8)

    _STREAMS = 3  # position, velocity, acceleration

    def __init__(self, fmt, num_classes, emb_dim=256, base_channels=96,
                 num_stages=10, num_prototype=100, dropout=0.5):
        super().__init__()
        if fmt != "g1":
            raise ValueError(
                "UNVEIL's kinematic hierarchy is defined on the G1 skeleton; "
                "use --format g1"
            )
        self.fmt = fmt
        self.num_classes = num_classes

        self.register_buffer(
            "joint_idx", torch.from_numpy(self._joint_index())
        )
        A_np, M_np, self.subgraph_names = self._graph()
        A = torch.tensor(A_np, dtype=torch.float32)
        M = torch.tensor(M_np, dtype=torch.float32)
        self.num_joints = int(A_np.shape[1])
        num_subgraphs = int(A_np.shape[0])

        self.data_bn = nn.BatchNorm1d(
            self._STREAMS * self.num_joints * DOF_PER_JOINT
        )
        self.encoder = KinematicEncoder(
            num_joints=self.num_joints, out_channels=base_channels,
            streams=self._STREAMS,
        )

        # Layers 2..num_stages of Tab. A.2: (in, out, temporal stride)
        specs = []
        cur, inflate = base_channels, 0
        for i in range(2, num_stages + 1):
            if i in self._INFLATE_STAGES:
                inflate += 1
            out = base_channels * (2 ** inflate)
            specs.append((cur, out, 2 if i in self._DOWN_STAGES else 1))
            cur = out
        self.blocks = nn.ModuleList(
            [STBlock(i, o, A, M, stride=s) for (i, o, s) in specs]
        )
        self._out_channels = cur
        # Learned-topology channels of the final layer (input to the bank)
        topology_channels = num_subgraphs * int(0.125 * cur)
        self.signature_bank = SignatureBank(
            topology_channels, n_prototypes=num_prototype
        )

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.fc = nn.Linear(self._out_channels, num_classes)
        self.z_proj = nn.Sequential(
            nn.Linear(self._out_channels, emb_dim), nn.LayerNorm(emb_dim)
        )

    def _graph(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """(A_static, masks, names) — the nine-subgraph hierarchy (Sec. 3.2)."""
        return build_subgraphs()

    def _joint_index(self) -> np.ndarray:
        """(J, D) channel gather index mapping input channels to joints."""
        return build_joint_channel_index()

    def _regroup(self, x: torch.Tensor) -> torch.Tensor:
        """(B, S, 35, T) -> (B, S, J, D, T) with zero-padded missing DoFs."""
        b, s, _, t = x.shape
        x = torch.cat([x, x.new_zeros(b, s, 1, t)], dim=2)  # zero channel @35
        return x[:, :, self.joint_idx, :]                    # (B,S,J,D,T)

    def forward(self, x, lengths=None):
        x = self._regroup(x)
        n, s, j, d, t = x.shape
        x = self.data_bn(x.reshape(n, s * j * d, t)).reshape(n, s, j, d, t)
        x = self.encoder(x)                    # (N, C, T, V)
        topology = None
        for block in self.blocks:
            x, topology = block(x)
        feat = x.mean(dim=(2, 3))              # GAP over time and joints
        signature = self.signature_bank(topology)

        if self.dropout is not None:
            feat = self.dropout(feat)
        logits = self.fc(feat)
        z = F.normalize(self.z_proj(feat), dim=-1)
        return logits, z, signature


# === SECTION: UNVEIL WRAPPER ===

class UNVEIL(nn.Module):
    """Selects the spatial backbone by name; forward always returns (logits, z, aux)."""

    def __init__(self, spatial_backbone: str, **kwargs):
        super().__init__()
        self.spatial_backbone = spatial_backbone
        if spatial_backbone == "sgn":
            self.backbone = StreamAttnBackbone(**kwargs)
        elif spatial_backbone == "dsgcn":
            self.backbone = DynGraphBackbone(**kwargs)
        elif spatial_backbone == "unveil":
            self.backbone = UnveilBackbone(**kwargs)
        else:
            raise ValueError(f"Unknown spatial_backbone: {spatial_backbone!r}")
        self.num_classes = self.backbone.num_classes
        self.num_joints = getattr(self.backbone, "num_joints",
                                   getattr(self.backbone, "num_joint", None))
        self.z_proj = self.backbone.z_proj

    def forward(self, x, lengths=None):
        out = self.backbone(x, lengths)
        if len(out) == 2:
            logits, z = out
            return logits, z, None
        return out  # (logits, z, signature) for the unveil backbone


# === SECTION: DECONFOUNDING ===

def deconfound_embeddings(Z_train, Z_test, t_train, t_test, num_tasks, mode="residual"):
    if mode == "none":
        return Z_train, Z_test
    train_finite = np.isfinite(Z_train).all(axis=1) & np.isfinite(t_train)
    test_finite = np.isfinite(Z_test).all(axis=1) & np.isfinite(t_test)
    if int(train_finite.sum()) < 2 or int(test_finite.sum()) == 0:
        print("  [warn] deconfound skipped (insufficient finite embeddings)")
        Ztr = np.nan_to_num(Z_train, nan=0.0, posinf=0.0, neginf=0.0)
        Zte = np.nan_to_num(Z_test, nan=0.0, posinf=0.0, neginf=0.0)
        Ztr /= np.linalg.norm(Ztr, axis=1, keepdims=True) + 1e-12
        Zte /= np.linalg.norm(Zte, axis=1, keepdims=True) + 1e-12
        return Ztr, Zte
    Z_train_fit, t_train_fit = Z_train[train_finite], t_train[train_finite]
    Z_test_fit, t_test_fit = Z_test[test_finite], t_test[test_finite]
    oh_tr = np.eye(num_tasks)[t_train_fit]
    oh_te = np.eye(num_tasks)[t_test_fit]
    reg = Ridge(alpha=1.0, fit_intercept=True)
    reg.fit(oh_tr, Z_train_fit)
    Ztr = np.zeros_like(Z_train)
    Zte = np.zeros_like(Z_test)
    Ztr[train_finite] = Z_train_fit - reg.predict(oh_tr)
    Zte[test_finite] = Z_test_fit - reg.predict(oh_te)
    r2 = 1.0 - np.var(Ztr[train_finite], axis=0).sum() / (np.var(Z_train_fit, axis=0).sum() + 1e-12)
    print(f"  Task deconfound R²: {r2:.4f}")
    Ztr /= np.linalg.norm(Ztr, axis=1, keepdims=True) + 1e-12
    Zte /= np.linalg.norm(Zte, axis=1, keepdims=True) + 1e-12
    return Ztr, Zte


# === SECTION: EVALUATION ===

@torch.no_grad()
def eval_cls(model, loader) -> Tuple[float, int, int]:
    model.eval()
    correct, total = 0, 0
    for xb, _, yb, _ in loader:
        xb = xb.to(DEVICE); yb = yb.to(DEVICE)
        logits, _, _ = model(xb)
        correct += int((logits.argmax(1) == yb).sum())
        total += int(yb.numel())
    return correct / max(1, total), correct, total


@torch.no_grad()
def eval_reg(model, loader) -> Tuple[float, float]:
    model.eval()
    preds, trues = [], []
    for xb, _, yb, _ in loader:
        xb = xb.to(DEVICE); yb = yb.to(DEVICE)
        pred, _, _ = model(xb)
        pred = pred.squeeze(1)
        fm = torch.isfinite(pred) & torch.isfinite(yb)
        if int(fm.sum().item()) == 0:
            continue
        preds.append(pred[fm].cpu().numpy())
        trues.append(yb[fm].cpu().numpy())
    if not preds:
        return float("nan"), float("nan")
    y_true = np.concatenate(trues, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return mae, 1.0 - ss_res / (ss_tot + 1e-12)


@torch.no_grad()
def eval_reid_closed_set(model, loader):
    model.eval()
    y_pred_all, y_true_all, scores_all = [], [], []
    for xb, _, yb, _ in loader:
        xb = xb.to(DEVICE)
        logits, _, _ = model(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()
        y_pred_all.append(preds); y_true_all.append(yb.numpy()); scores_all.append(probs)
    if not y_true_all:
        return 0.0, 0.0, 0.0, float("nan"), np.array([]), np.array([])
    ya = np.concatenate(y_true_all)
    yp = np.concatenate(y_pred_all)
    sc = np.concatenate(scores_all, axis=0)
    top1 = float(np.mean(ya == yp))
    k = min(5, sc.shape[1])
    top5 = float(np.mean([ya[i] in np.argpartition(sc[i], -k)[-k:] for i in range(len(ya))]))
    f1 = f1_score(ya, yp, average="macro", zero_division=0)
    try:
        classes_present = np.unique(ya)
        auc = roc_auc_score(
            np.searchsorted(classes_present, ya),
            sc[:, classes_present], multi_class="ovr", average="macro",
        )
    except (ValueError, IndexError):
        auc = float("nan")
    return top1, top5, f1, auc, ya, yp


@torch.no_grad()
def collect_embeddings(model, loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    Z_list, Y_list, T_list = [], [], []
    for xb, _, yb, tids in loader:
        xb = xb.to(DEVICE)
        if not torch.isfinite(xb).all():
            xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
        _, z, _ = model(xb)
        z_np = z.cpu().numpy()
        y_np = yb.numpy()
        t_np = tids.numpy()
        fm = np.isfinite(z_np).all(axis=1)
        if np.issubdtype(y_np.dtype, np.number):
            fm &= np.isfinite(y_np)
        if int(fm.sum()) == 0:
            continue
        Z_list.append(z_np[fm]); Y_list.append(y_np[fm]); T_list.append(t_np[fm])
    emb_dim = model.z_proj[0].out_features if hasattr(model, "z_proj") else 256
    if not Z_list:
        return (np.zeros((0, emb_dim), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64))
    return (np.concatenate(Z_list, axis=0),
            np.concatenate(Y_list, axis=0),
            np.concatenate(T_list, axis=0))


def rank1_accuracy(Z_gallery, y_gallery, Z_probe, y_probe):
    classes = np.unique(y_gallery)
    if len(classes) == 0:
        return 0.0, 0, 0
    centroids = np.stack([Z_gallery[y_gallery == c].mean(0) for c in classes])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    pred = classes[np.argmax(Z_probe @ centroids.T, axis=1)]
    correct = int(np.sum(pred == y_probe))
    return correct / max(1, len(y_probe)), correct, len(y_probe)


def rank_k_accuracy(Z_gallery, y_gallery, Z_probe, y_probe, k=5):
    classes = np.unique(y_gallery)
    if len(classes) == 0 or len(y_probe) == 0:
        return 0.0, 0, len(y_probe)
    k = min(k, len(classes))
    if k <= 0:
        return 0.0, 0, len(y_probe)
    centroids = np.stack([Z_gallery[y_gallery == c].mean(0) for c in classes])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    sims = Z_probe @ centroids.T
    topk_idx = np.argpartition(sims, -k, axis=1)[:, -k:]
    topk_classes = classes[topk_idx]
    hits = np.array([y in row for y, row in zip(y_probe, topk_classes)])
    return float(hits.mean()), int(hits.sum()), len(y_probe)


def compute_regression_bin_metrics(y_true, y_pred, ref_targets, num_bins=5):
    if len(y_true) == 0 or len(y_pred) == 0 or len(ref_targets) == 0:
        return {"bin_acc": float("nan"), "bin_bal_acc": float("nan"), "bin_f1": float("nan"),
                "adjacent_bin_acc": float("nan"), "chance": float("nan"),
                "num_bins": 0, "bin_edges": [], "per_bin": []}
    bin_edges = np.unique(np.quantile(ref_targets, np.linspace(0, 1, num_bins + 1)))
    if len(bin_edges) < num_bins + 1:
        bin_edges = np.linspace(ref_targets.min(), ref_targets.max(), num_bins + 1)
    actual_bins = len(bin_edges) - 1
    if actual_bins <= 0:
        return {"bin_acc": float("nan"), "bin_bal_acc": float("nan"), "bin_f1": float("nan"),
                "adjacent_bin_acc": float("nan"), "chance": float("nan"),
                "num_bins": 0, "bin_edges": [], "per_bin": []}
    y_true_bins = np.clip(np.digitize(y_true, bin_edges) - 1, 0, actual_bins - 1)
    y_pred_bins = np.clip(np.digitize(y_pred, bin_edges) - 1, 0, actual_bins - 1)
    bin_acc = float(accuracy_score(y_true_bins, y_pred_bins))
    bin_bal_terms = []
    per_bin = []
    for b in range(actual_bins):
        mask = y_true_bins == b
        n_b = int(mask.sum())
        acc_b = float(accuracy_score(y_true_bins[mask], y_pred_bins[mask])) if n_b > 0 else float("nan")
        if n_b > 0:
            bin_bal_terms.append(acc_b)
        per_bin.append({"bin_idx": int(b), "low": float(bin_edges[b]),
                        "high": float(bin_edges[b + 1]), "acc": acc_b, "n": n_b})
    return {
        "bin_acc": bin_acc,
        "bin_bal_acc": float(np.mean(bin_bal_terms)) if bin_bal_terms else float("nan"),
        "bin_f1": float(f1_score(y_true_bins, y_pred_bins, average="macro", zero_division=0.0)),
        "adjacent_bin_acc": float(np.mean(np.abs(y_pred_bins - y_true_bins) <= 1)),
        "chance": float(1.0 / max(1, actual_bins)),
        "num_bins": int(actual_bins),
        "bin_edges": [float(x) for x in bin_edges.tolist()],
        "per_bin": per_bin,
    }


# === SECTION: CHECKPOINTING ===

def save_checkpoint(model, opt, epoch, metrics, args, filename):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    path = os.path.join(args.checkpoint_dir, filename)
    raw_model = getattr(model, "_orig_mod", model)
    torch.save({
        "epoch": epoch,
        "model_state": {k: v.cpu() for k, v in raw_model.state_dict().items()},
        "opt_state": opt.state_dict(),
        "metrics": metrics,
        "config": {
            "spatial_backbone": args.spatial_backbone, "format": args.format, "task": args.task,
            "num_joints": getattr(raw_model, "num_joints", None),
            "num_classes": raw_model.num_classes,
            "emb_dim": args.emb_dim, "base_channels": args.base_channels,
            "num_prototype": args.num_prototype, "target_fps": args.target_fps,
            "max_seq_len": args.max_seq_len, "split_mode": args.split_mode,
            "deconfound": args.deconfound,
        },
    }, path)
    print(f"  Checkpoint saved -> {path}")


def resume_checkpoint(model, opt, args):
    ckpt_dir = args.checkpoint_dir
    if not os.path.isdir(ckpt_dir):
        print("No checkpoint directory found. Starting from scratch.")
        return 1, -1.0, -1, None, 0
    best_path = os.path.join(ckpt_dir, "best_model.pt")
    periodic = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch*.pt")))
    latest = periodic[-1] if periodic else None
    load_path = None
    if latest and os.path.exists(best_path):
        ep_b = torch.load(best_path, map_location="cpu", weights_only=True)["epoch"]
        ep_p = torch.load(latest, map_location="cpu", weights_only=True)["epoch"]
        load_path = latest if ep_p >= ep_b else best_path
    elif os.path.exists(best_path):
        load_path = best_path
    elif latest:
        load_path = latest
    if load_path is None:
        print("No existing checkpoints found. Starting from scratch.")
        return 1, -1.0, -1, None, 0
    print(f"Resuming from: {load_path}")
    ckpt = torch.load(load_path, map_location=DEVICE, weights_only=True)
    ckpt_cfg = ckpt.get("config", {})
    for key, val in [("format", args.format), ("task", args.task), ("spatial_backbone", args.spatial_backbone)]:
        if ckpt_cfg.get(key, val) != val:
            print(f"  [WARN] Checkpoint {key}={ckpt_cfg.get(key)!r} != {val!r}. Starting fresh.")
            return 1, -1.0, -1, None, 0
    raw_model = getattr(model, "_orig_mod", model)
    raw_model.load_state_dict(ckpt["model_state"])
    opt.load_state_dict(ckpt["opt_state"])
    resume_epoch = int(ckpt["epoch"])
    start_epoch = resume_epoch + 1
    best_metric = float(ckpt.get("metrics", {}).get("best_metric", -1.0))
    best_epoch = resume_epoch
    best_state = {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}
    if os.path.exists(best_path):
        best_ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
        if "model_state" in best_ckpt:
            best_state = {k: v.detach().cpu().clone() for k, v in best_ckpt["model_state"].items()}
        best_epoch = int(best_ckpt.get("epoch", best_epoch))
        best_metric = float(best_ckpt.get("metrics", {}).get("best_metric", best_metric))
    print(f"  Loaded epoch {resume_epoch} -> start {start_epoch} | best={best_metric:.4f} @ {best_epoch}")
    return start_epoch, best_metric, best_epoch, best_state, resume_epoch


# === SECTION: MAIN TASK RUNNER ===

def run_single_task(args: argparse.Namespace) -> Dict:
    set_seed(args.seed)
    print("=" * 80)
    print(f"UNVEIL | spatial_backbone={args.spatial_backbone} | task={args.task} | format={args.format} | split={args.split_mode}")
    print("=" * 80)

    train_df, seen_val_df, unseen_val_df = prepare_data(args)
    seen_val_train_df, seen_actors_unseen_df = split_seen_val_per_actor(seen_val_df, 0.8, args.seed)
    train_combined = pd.concat([train_df, seen_val_train_df], ignore_index=True).reset_index(drop=True)

    if args.max_train > 0 and len(train_combined) > args.max_train:
        train_combined = train_combined.sample(n=args.max_train, random_state=args.seed).reset_index(drop=True)
        print(f"  --max-train: {len(train_combined):,} sampled rows")

    print(f"3-way split: train={len(train_combined):,} "
          f"sa_unseen={len(seen_actors_unseen_df):,} unseen={len(unseen_val_df):,}")

    g1_cache_info = None
    if args.format == "g1" and not args.no_g1_cache:
        g1_cache_info = load_g1_cache(args.g1_cache_dir)
        if g1_cache_info:
            print(f"G1 cache: {args.g1_cache_dir} ({len(g1_cache_info['index_map']):,} clips)")
        else:
            print("G1 cache not found, falling back to CSV reads.")

    is_regression = args.task in REGRESSION_TASKS
    label_col = {
        "reid": "actor_uid", "gender": "actor_gender",
        "age": "actor_age_yr", "height": "actor_height_cm", "weight": "actor_weight_kg",
    }[args.task]

    if is_regression:
        for dfp in [train_combined, seen_val_train_df, seen_actors_unseen_df, unseen_val_df]:
            dfp.dropna(subset=[label_col], inplace=True)
        label_map = None
        num_classes = 1
    else:
        all_labels = sorted(train_combined[label_col].dropna().unique().tolist())
        label_map = {lbl: i for i, lbl in enumerate(all_labels)}
        num_classes = len(label_map)
        for dfp in [seen_val_train_df, seen_actors_unseen_df, unseen_val_df]:
            bad = ~dfp[label_col].isin(label_map)
            dfp.drop(dfp.index[bad], inplace=True)
            dfp.reset_index(drop=True, inplace=True)

    print(f"Label: {label_col} | num_outputs: {num_classes}")

    task_col = args.deconfound_key
    if task_col not in train_df.columns:
        print(f"[WARN] Column '{task_col}' not found. Deconfounding disabled.")
        args.deconfound = "none"
        all_tasks = ["unknown"]
    else:
        all_tasks = sorted(
            set(train_df[task_col].dropna().unique())
            | set(seen_actors_unseen_df[task_col].dropna().unique())
            | set(unseen_val_df[task_col].dropna().unique())
        )
    task_map = {t: i for i, t in enumerate(all_tasks)}
    num_tasks = len(task_map)

    if args.format in ("uniform", "proportional"):
        base_indices = get_bvh_channel_indices()
        channel_indices = refine_bvh_channels_by_variance(
            train_df, args.data_root, args.format, base_indices,
            args.variance_percentile, seed=args.seed,
        )
        num_channels = len(channel_indices)
        print(f"BVH channels: {num_channels} (variance_percentile={args.variance_percentile})")
    else:
        channel_indices = None
        num_channels = 35
        print(f"G1 channels: {num_channels}")

    native_fps = 120
    downsample_factor = max(1, native_fps // args.target_fps)
    effective_fps = native_fps // downsample_factor
    print(f"FPS: {native_fps} -> {effective_fps} (factor={downsample_factor}) | max_seq={args.max_seq_len}")

    print("Computing global normalization stats...")
    global_mean, global_std = compute_global_norm(
        train_combined, args.data_root, args.format, channel_indices, downsample_factor,
        max_samples=5000, seed=args.seed, g1_cache_info=g1_cache_info,
    )
    print(f"  Mean [{global_mean.min():.4f}, {global_mean.max():.4f}] | "
          f"Std [{global_std.min():.4f}, {global_std.max():.4f}]")

    ds_kwargs = dict(
        data_root=args.data_root, fmt=args.format, label_col=label_col,
        label_map=label_map, is_regression=is_regression,
        task_col=task_col, task_map=task_map, channel_indices=channel_indices,
        downsample_factor=downsample_factor, max_seq_len=args.max_seq_len,
        min_seq_len=args.min_seq_len, global_mean=global_mean, global_std=global_std,
        seed=args.seed, g1_cache_info=g1_cache_info,
    )
    print("Loading datasets...")
    train_ds = BonesSeedDataset(train_combined, train=True, **ds_kwargs)
    sa_seen_ds = BonesSeedDataset(seen_val_train_df, train=False, **ds_kwargs)
    sa_unseen_ds = BonesSeedDataset(seen_actors_unseen_df, train=False, **ds_kwargs)
    unseen_ds = BonesSeedDataset(unseen_val_df, train=False, **ds_kwargs)

    _persist = args.num_workers > 0
    _lkw = dict(num_workers=args.num_workers, collate_fn=collate_padded,
                pin_memory=(DEVICE == "cuda"), persistent_workers=_persist)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=True, **_lkw)
    sa_seen_loader = DataLoader(sa_seen_ds, args.batch_size, shuffle=False, **_lkw)
    sa_unseen_loader = DataLoader(sa_unseen_ds, args.batch_size, shuffle=False, **_lkw)
    unseen_loader = DataLoader(unseen_ds, args.batch_size, shuffle=False, **_lkw)
    print(f"  train={len(train_ds):,} sa_seen={len(sa_seen_ds):,} "
          f"sa_unseen={len(sa_unseen_ds):,} unseen={len(unseen_ds):,}")

    model_kwargs: Dict = dict(num_classes=num_classes, emb_dim=args.emb_dim)
    if args.spatial_backbone == "sgn":
        model_kwargs.update(num_joint=num_channels, dim1=args.dim1, seg=args.seg)
    else:
        model_kwargs.update(fmt=args.format, base_channels=args.base_channels,
                            num_stages=args.num_stages, dropout=args.dropout)
        if args.spatial_backbone == "unveil":
            model_kwargs["num_prototype"] = args.num_prototype

    model = UNVEIL(spatial_backbone=args.spatial_backbone, **model_kwargs).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nUNVEIL ({args.spatial_backbone}): {n_params:,} parameters | device={DEVICE}")

    if not args.no_compile and hasattr(torch, "compile"):
        # torch.compile fails lazily at the first forward if triton is missing
        # (common on Windows), so probe for it up front.
        try:
            import triton  # noqa: F401
            model = torch.compile(model)
            print("  torch.compile: enabled")
        except ImportError:
            print("  torch.compile: triton not available, running eager")
        except Exception as e:
            print(f"  torch.compile: failed ({e}), continuing without")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce_loss_fn = nn.MSELoss() if is_regression else nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    supcon_fn = SupConLoss(temperature=args.supcon_temp)

    # Operator-contrastive loss on the distilled signature (classification
    # only: it contrasts class prototypes, so it needs discrete labels).
    # L_csc deliberately shares --supcon-warmup, and its projection `fc` is a
    # fixed random projection (not in the optimizer) — this matches the
    # training protocol used for the paper's reported numbers.
    csc_loss_fn = None
    if args.spatial_backbone == "unveil" and not is_regression and num_classes > 1:
        raw_m = getattr(model, "_orig_mod", model)
        n_joints_val = getattr(raw_m, "num_joints", NUM_JOINTS)
        csc_loss_fn = SignatureContrastiveLoss(
            n_class=num_classes, in_dim=n_joints_val * n_joints_val,
            h_dim=min(256, args.emb_dim),
            temperature=args.csc_temp, momentum=args.csc_momentum,
        ).to(DEVICE)

    start_epoch, best_metric, best_epoch, best_state, resume_epoch = resume_checkpoint(model, opt, args)
    if is_regression and best_metric == -1.0:
        best_metric = float("-inf")
    if best_state is None:
        raw_m = getattr(model, "_orig_mod", model)
        best_state = {k: v.detach().cpu().clone() for k, v in raw_m.state_dict().items()}
        best_epoch = max(0, start_epoch - 1)
    bad_epochs = max(0, (resume_epoch - max(0, best_epoch)) // max(1, args.eval_every))

    print(f"\nTraining {args.epochs} epochs | SupCon warmup={args.supcon_warmup} | early_stop={args.early_stop}")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_sum = ce_sum = sc_sum = csc_sum = 0.0
        correct = total = skipped = 0
        t0 = time.time()

        for xb, lengths, yb, tids in train_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            if not torch.isfinite(xb).all():
                xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
                if not torch.isfinite(xb).all():
                    skipped += 1; continue
            logits, z, aux = model(xb)
            if not torch.isfinite(logits).all():
                skipped += 1; continue

            if is_regression:
                loss_ce = ce_loss_fn(logits.squeeze(1), yb)
                sc_labels = _supcon_labels(yb, args.task)
            else:
                loss_ce = ce_loss_fn(logits, yb)
                sc_labels = yb
            ce_sum += loss_ce.item() * xb.size(0)
            loss = loss_ce
            lam_sc = args.lambda_supcon if epoch >= args.supcon_warmup else 0.0
            if lam_sc > 0:
                loss_sc = supcon_fn(z, sc_labels)
                loss = loss + lam_sc * loss_sc
                sc_sum += loss_sc.item() * xb.size(0)
            if (not is_regression and csc_loss_fn is not None and args.lambda_csc > 0
                    and epoch >= args.supcon_warmup and aux is not None):
                try:
                    loss_csc = csc_loss_fn(aux, yb)
                    if torch.isfinite(loss_csc):
                        loss = loss + args.lambda_csc * loss_csc
                        csc_sum += loss_csc.item() * xb.size(0)
                except Exception:
                    pass

            if not torch.isfinite(loss):
                skipped += 1; continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            loss_sum += loss.item() * xb.size(0)
            if not is_regression:
                correct += int((logits.argmax(1) == yb).sum())
            total += int(yb.numel())

        if skipped > 0:
            print(f"  [warn] skipped {skipped} non-finite batches in epoch {epoch}")
        tr_loss = loss_sum / max(1, total)
        tr_ce = ce_sum / max(1, total)
        tr_acc = correct / max(1, total) if not is_regression else 0.0
        elapsed = time.time() - t0

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            te_acc = 0.0
            if is_regression:
                sa_mae, sa_r2 = eval_reg(model, sa_seen_loader)
                su_mae, su_r2 = eval_reg(model, sa_unseen_loader)
                un_mae, un_r2 = eval_reg(model, unseen_loader)
                primary_metric = -un_mae
                metric_str = (f"sa_mae={sa_mae:.4f} r2={sa_r2:.4f} | "
                              f"su_mae={su_mae:.4f} | unseen_mae={un_mae:.4f} r2={un_r2:.4f}")
            elif args.task == "reid":
                if args.reid_eval == "closed-set":
                    sa_t1, sa_t5, _, _, _, _ = eval_reid_closed_set(model, sa_seen_loader)
                    su_t1, su_t5, _, _, _, _ = eval_reid_closed_set(model, sa_unseen_loader)
                    primary_metric = su_t1
                    metric_str = f"sa top1={sa_t1:.4f} t5={sa_t5:.4f} | su top1={su_t1:.4f} t5={su_t5:.4f}"
                    te_acc = su_t1
                else:
                    Z_gal, y_gal, t_gal = collect_embeddings(model, sa_seen_loader)
                    Z_prb, y_prb, t_prb = collect_embeddings(model, sa_unseen_loader)
                    Z_gal_dc, Z_prb_dc = deconfound_embeddings(Z_gal, Z_prb, t_gal, t_prb, num_tasks, args.deconfound)
                    primary_metric, r1c, r1n = rank1_accuracy(Z_gal_dc, y_gal, Z_prb_dc, y_prb)
                    metric_str = f"centroid rank1={primary_metric:.4f} ({r1c}/{r1n})"
                    te_acc = primary_metric
            else:
                sa_acc, _, _ = eval_cls(model, sa_seen_loader)
                su_acc, _, _ = eval_cls(model, sa_unseen_loader)
                un_acc, un_c, un_n = eval_cls(model, unseen_loader)
                primary_metric = un_acc
                metric_str = f"sa={sa_acc:.4f} su={su_acc:.4f} unseen={un_acc:.4f} ({un_c}/{un_n})"
                te_acc = un_acc

            marker = " <-- best" if primary_metric > best_metric + 1e-6 else ""
            print(f"Epoch {epoch:03d} [{elapsed:.1f}s] | loss={tr_loss:.4f} ce={tr_ce:.4f}"
                  + (f" sc={sc_sum / max(1, total):.4f}" if sc_sum > 0 else "")
                  + (f" csc={csc_sum / max(1, total):.4f}" if csc_sum > 0 else "")
                  + (f" tr={tr_acc:.4f}" if not is_regression else "")
                  + f" | {metric_str}{marker}")

            if primary_metric > best_metric + 1e-6:
                best_metric = primary_metric
                best_epoch = epoch
                raw_m = getattr(model, "_orig_mod", model)
                best_state = {k: v.detach().cpu().clone() for k, v in raw_m.state_dict().items()}
                bad_epochs = 0
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                torch.save({
                    "epoch": best_epoch, "model_state": best_state,
                    "opt_state": opt.state_dict(),
                    "metrics": {"best_metric": best_metric, "test_acc": te_acc},
                    "config": {"spatial_backbone": args.spatial_backbone, "format": args.format,
                                "task": args.task, "num_classes": num_classes,
                                "emb_dim": args.emb_dim, "base_channels": args.base_channels,
                                "num_prototype": args.num_prototype},
                }, os.path.join(args.checkpoint_dir, "best_model.pt"))
                print(f"  Best checkpoint saved (epoch {best_epoch})")
            else:
                bad_epochs += 1
                if bad_epochs >= args.early_stop:
                    print(f"Early stopping at epoch {epoch} ({args.early_stop} evals without improvement)")
                    break

            if epoch % args.save_every == 0:
                save_checkpoint(model, opt, epoch,
                                {"best_metric": best_metric, "test_acc": te_acc},
                                args, f"checkpoint_epoch{epoch:03d}.pt")
        else:
            print(f"Epoch {epoch:03d} [{elapsed:.1f}s] | loss={tr_loss:.4f}"
                  + (f" tr={tr_acc:.4f}" if not is_regression else ""))

    # ── Final evaluation ──────────────────────────────────────────────────────
    eval_kwargs = {**model_kwargs}
    if "dropout" in eval_kwargs:
        eval_kwargs["dropout"] = 0.0
    eval_model = UNVEIL(spatial_backbone=args.spatial_backbone, **eval_kwargs).to(DEVICE)
    if best_state is not None:
        eval_model.load_state_dict(best_state)
    eval_model.eval()
    model = eval_model

    print("\n" + "=" * 80 + "\nFINAL EVALUATION (best checkpoint)\n" + "=" * 80)

    def make_loader(ds):
        return DataLoader(ds, args.batch_size, shuffle=False,
                          num_workers=args.num_workers, collate_fn=collate_padded,
                          pin_memory=(DEVICE == "cuda"))

    fl_sa_seen = make_loader(sa_seen_ds)
    fl_sa_unseen = make_loader(sa_unseen_ds)
    fl_unseen = make_loader(unseen_ds)

    final_metrics = {
        "task": args.task, "format": args.format, "spatial_backbone": args.spatial_backbone,
        "split_mode": args.split_mode, "best_epoch": best_epoch,
        "train_samples": len(train_ds), "sa_seen_samples": len(sa_seen_ds),
        "sa_unseen_samples": len(sa_unseen_ds), "unseen_samples": len(unseen_ds),
    }

    if args.task == "reid":
        print(f"\nRe-ID Results ({args.format}, {args.reid_eval}):")
        if args.reid_eval == "closed-set":
            for ldr, name, key in [
                (fl_sa_seen, "seen-actors-seen-demos", "sa_seen"),
                (fl_sa_unseen, "seen-actors-unseen-demos", "sa_unseen"),
            ]:
                t1, t5, f1_, auc_, ya, yp = eval_reid_closed_set(model, ldr)
                print(f"  {name}: top1={t1:.4f} top5={t5:.4f} F1={f1_:.4f} AUC={auc_:.4f}")
                final_metrics.update({f"{key}_top1": t1, f"{key}_top5": t5,
                                      f"{key}_f1_macro": f1_, f"{key}_roc_auc": auc_})
        else:
            Z_tr, y_tr, t_tr = collect_embeddings(model, make_loader(train_ds))
            for ldr, name, key in [
                (fl_sa_unseen, "seen-actors-unseen-demos", "sa_unseen"),
                (fl_unseen, "unseen-actors", "unseen"),
            ]:
                Z_prb, y_prb, t_prb = collect_embeddings(model, ldr)
                Z_tr_dc, Z_prb_dc = deconfound_embeddings(Z_tr, Z_prb, t_tr, t_prb, num_tasks, args.deconfound)
                r1, r1c, r1n = rank1_accuracy(Z_tr_dc, y_tr, Z_prb_dc, y_prb)
                r5, _, _ = rank_k_accuracy(Z_tr_dc, y_tr, Z_prb_dc, y_prb, k=5)
                print(f"  {name}: rank1={r1:.4f} ({r1c}/{r1n}) rank5={r5:.4f}")
                final_metrics.update({f"{key}_rank1": r1, f"{key}_rank5": r5})
        print(f"  Chance: {1/max(1,num_classes):.4f} ({num_classes} classes) | Best epoch: {best_epoch}")
        final_metrics["num_classes"] = num_classes

    elif args.task == "gender":
        def _gender_metrics(loader):
            y_pred_l, y_true_l, sc_l = [], [], []
            with torch.no_grad():
                for xb, _, yb, _ in loader:
                    xb = xb.to(DEVICE)
                    logits, _, _ = model(xb)
                    y_pred_l.extend(logits.argmax(1).cpu().tolist())
                    y_true_l.extend(yb.tolist())
                    if logits.shape[1] == 2:
                        sc_l.extend(logits[:, 1].cpu().tolist())
            ya, yp = np.array(y_true_l), np.array(y_pred_l)
            acc = accuracy_score(ya, yp)
            f1 = f1_score(ya, yp, average="macro")
            try:
                auc = roc_auc_score(ya, np.array(sc_l))
            except (ValueError, IndexError):
                auc = float("nan")
            return acc, f1, auc, ya, yp

        sa_acc, sa_f1, sa_auc, _, _ = _gender_metrics(fl_sa_seen)
        su_acc, su_f1, su_auc, _, _ = _gender_metrics(fl_sa_unseen)
        un_acc, un_f1, un_auc, y_true_all, y_pred = _gender_metrics(fl_unseen)
        inv_label = {v: k for k, v in label_map.items()}
        target_names = [inv_label[i] for i in range(num_classes)]
        print(f"\nGender Results ({args.format}):")
        print(f"  sa_seen:   acc={sa_acc:.4f} F1={sa_f1:.4f} AUC={sa_auc:.4f}")
        print(f"  sa_unseen: acc={su_acc:.4f} F1={su_f1:.4f} AUC={su_auc:.4f}")
        print(f"  unseen:    acc={un_acc:.4f} F1={un_f1:.4f} AUC={un_auc:.4f}")
        print(f"  Best epoch: {best_epoch}")
        print(classification_report(y_true_all, y_pred, target_names=target_names))
        final_metrics.update({
            "sa_seen_accuracy": sa_acc, "sa_seen_f1": sa_f1,
            "sa_unseen_accuracy": su_acc, "sa_unseen_f1": su_f1,
            "unseen_accuracy": un_acc, "unseen_f1_macro": un_f1,
            "unseen_roc_auc": un_auc,
            "confusion_matrix": confusion_matrix(y_true_all, y_pred).tolist(),
        })

    else:
        def _collect_reg(loader):
            preds, trues = [], []
            with torch.no_grad():
                for xb, _, yb, _ in loader:
                    xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                    pred, _, _ = model(xb)
                    pred = pred.squeeze(1)
                    fm = torch.isfinite(pred) & torch.isfinite(yb)
                    if int(fm.sum().item()) > 0:
                        preds.append(pred[fm].cpu().numpy())
                        trues.append(yb[fm].cpu().numpy())
            if not preds:
                return np.array([]), np.array([])
            return np.concatenate(preds), np.concatenate(trues)

        BIN_COUNTS = [5, 10]

        def _reg_metrics(y_pred, y_true, ref):
            if len(y_pred) == 0:
                return {"mae": float("nan"), "r2": float("nan"),
                        "bins": {nb: compute_regression_bin_metrics(np.array([]), np.array([]), ref, nb)
                                 for nb in BIN_COUNTS}}
            mae = float(np.mean(np.abs(y_true - y_pred)))
            ss_res = float(np.sum((y_true - y_pred) ** 2))
            ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
            return {"mae": mae, "r2": 1.0 - ss_res / (ss_tot + 1e-12),
                    "bins": {nb: compute_regression_bin_metrics(y_true, y_pred, ref, nb)
                             for nb in BIN_COUNTS}}

        sa_pred, sa_true = _collect_reg(fl_sa_seen)
        su_pred, su_true = _collect_reg(fl_sa_unseen)
        un_pred, un_true = _collect_reg(fl_unseen)
        sa_m = _reg_metrics(sa_pred, sa_true, sa_true)
        su_m = _reg_metrics(su_pred, su_true, su_true)
        un_m = _reg_metrics(un_pred, un_true, un_true)

        def _print_reg(prefix, m):
            print(f"  {prefix}: MAE={m['mae']:.4f} R²={m['r2']:.4f}")
            for nb, bm in sorted(m["bins"].items()):
                print(f"    [{nb}-bin] acc={bm['bin_acc']:.4f} bal={bm['bin_bal_acc']:.4f} "
                      f"adj={bm['adjacent_bin_acc']:.4f} (chance={bm['chance']:.4f})")

        print(f"\nRegression Results ({args.format}, {args.task}):")
        _print_reg("sa_seen", sa_m); _print_reg("sa_unseen", su_m); _print_reg("unseen", un_m)
        print(f"  Best epoch: {best_epoch}")

        def _flatten_reg(prefix, m):
            flat = {f"{prefix}_mae": m["mae"], f"{prefix}_r2": m["r2"]}
            for nb, bm in m["bins"].items():
                flat.update({f"{prefix}_{nb}bin_acc": bm["bin_acc"],
                             f"{prefix}_{nb}bin_bal_acc": bm["bin_bal_acc"],
                             f"{prefix}_{nb}bin_f1": bm["bin_f1"],
                             f"{prefix}_{nb}bin_adj_acc": bm["adjacent_bin_acc"],
                             f"{prefix}_{nb}bin_chance": bm["chance"]})
            return flat

        final_metrics.update({**_flatten_reg("sa_seen", sa_m),
                               **_flatten_reg("sa_unseen", su_m),
                               **_flatten_reg("unseen", un_m)})

    metrics_path = os.path.join(args.checkpoint_dir, f"final_metrics_{args.format}_{args.task}.json")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")
    save_checkpoint(model, opt, best_epoch, final_metrics, args, f"final_{args.format}_{args.task}.pt")
    print("Done.")
    return final_metrics


# === SECTION: ARGUMENT PARSING ===

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UNVEIL: privacy modeling on BONES-SEED")

    p.add_argument("--spatial-backbone",
                   choices=["unveil", "sgn", "dsgcn"],
                   default="unveil",
                   help="unveil (the paper's model) or a comparison backbone")
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("--splits-dir", type=str, default=None)
    p.add_argument("--g1-cache-dir", type=str, default=None)
    p.add_argument("--no-g1-cache", action="store_true")
    p.add_argument("--format", choices=["g1", "uniform", "proportional"], default="g1")
    p.add_argument("--task", choices=["all"] + ALL_TASKS, default="reid")
    p.add_argument("--split-mode", choices=["user", "user_task"], default="user")
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--min-motions", type=int, default=20)
    p.add_argument("--deconfound", choices=["none", "residual"], default="residual")
    p.add_argument("--deconfound-key", choices=["package", "category"], default="package")
    p.add_argument("--target-fps", type=int, default=30)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--min-seq-len", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--lambda-supcon", type=float, default=0.1)
    p.add_argument("--supcon-temp", type=float, default=0.07)
    p.add_argument("--early-stop", type=int, default=40)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-test", type=int, default=0)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--reid-eval", choices=["centroid", "closed-set"], default=None)

    # Backbone-specific args with None defaults (filled from SPATIAL_BACKBONE_DEFAULTS after parsing)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--emb-dim", type=int, default=None)
    p.add_argument("--supcon-warmup", type=int, default=None)
    p.add_argument("--variance-percentile", type=float, default=None)
    p.add_argument("--dim1", type=int, default=None)
    p.add_argument("--seg", type=int, default=None)
    p.add_argument("--base-channels", type=int, default=None)
    p.add_argument("--num-stages", type=int, default=None)
    p.add_argument("--num-prototype", type=int, default=None,
                   help="signature-bank size (number of learnable prototypes)")
    p.add_argument("--lambda-csc", type=float, default=None,
                   help="operator-contrastive loss weight (unveil only)")
    p.add_argument("--csc-temp", type=float, default=0.125)
    p.add_argument("--csc-momentum", type=float, default=0.9)

    args = p.parse_args()

    vd = SPATIAL_BACKBONE_DEFAULTS[args.spatial_backbone]
    for key, val in vd.items():
        attr = key.replace("-", "_")
        if getattr(args, attr, None) is None:
            setattr(args, attr, val)

    # Fallbacks for any remaining None args
    if args.lr is None: args.lr = 3e-4
    if args.batch_size is None: args.batch_size = 64
    if args.emb_dim is None: args.emb_dim = 256
    if args.supcon_warmup is None: args.supcon_warmup = 20
    if args.variance_percentile is None: args.variance_percentile = 10.0
    if args.dim1 is None: args.dim1 = 256
    if args.seg is None: args.seg = 64
    if args.base_channels is None: args.base_channels = 64
    if args.num_stages is None: args.num_stages = 10
    if args.num_prototype is None: args.num_prototype = 100
    if args.lambda_csc is None: args.lambda_csc = 0.0

    if args.reid_eval is None:
        args.reid_eval = "centroid" if args.spatial_backbone == "sgn" else "closed-set"

    # UNVEIL's kinematic hierarchy is defined on the G1 skeleton (Tab. A.5)
    if args.spatial_backbone == "unveil" and args.format != "g1":
        p.error("the unveil backbone requires --format g1")

    # Graph-based spatial backbones need all BVH channels (exact joint count required)
    if args.spatial_backbone == "dsgcn" and args.format in ("uniform", "proportional"):
        if args.variance_percentile != 0.0:
            print(f"[INFO] Forcing variance_percentile=0.0 for {args.spatial_backbone} (needs {BVH_NUM_JOINTS} joints)")
            args.variance_percentile = 0.0

    data_root_resolved = Path(args.data_root).resolve()

    if args.splits_dir is None:
        if data_root_resolved == PROJECT_ROOT:
            args.splits_dir = str(default_splits_dir(create=False))
        else:
            args.splits_dir = os.path.join(args.data_root, "splits")

    if args.g1_cache_dir is None:
        if data_root_resolved == PROJECT_ROOT:
            args.g1_cache_dir = str(default_g1_cache_dir(create=False))
        else:
            args.g1_cache_dir = os.path.join(args.data_root, "cache", "g1_motions")

    if args.checkpoint_dir is None:
        if data_root_resolved == PROJECT_ROOT:
            try:
                from project_paths import MODELS_DIR
                args.checkpoint_dir = str(
                    MODELS_DIR / "unveil" / args.spatial_backbone / f"actor_holdout_split_{args.format}"
                )
            except Exception:
                args.checkpoint_dir = os.path.join(str(PROJECT_ROOT), "checkpoints_unveil", args.spatial_backbone)
        else:
            args.checkpoint_dir = os.path.join(args.data_root, "checkpoints_unveil", args.spatial_backbone)

    return args


# === SECTION: MAIN ===

def main():
    args = parse_args()
    tasks = ALL_TASKS if args.task == "all" else [args.task]
    base_ckpt_dir = args.checkpoint_dir
    all_results = {}
    for task in tasks:
        task_args = argparse.Namespace(**vars(args))
        task_args.task = task
        task_args.checkpoint_dir = os.path.join(base_ckpt_dir, task)
        all_results[task] = run_single_task(task_args)
    if len(tasks) > 1:
        os.makedirs(base_ckpt_dir, exist_ok=True)
        summary_path = os.path.join(base_ckpt_dir, f"summary_{args.format}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"format": args.format, "spatial_backbone": args.spatial_backbone,
                       "tasks": tasks, "results": all_results}, f, indent=2)
        print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
