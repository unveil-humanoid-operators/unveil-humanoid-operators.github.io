#!/usr/bin/env python3
"""
Linkage Attack Evaluation
=========================

Given L2-normalized re-ID embeddings from UNVEIL, evaluates whether an
attacker can determine if two motion clips came from the same person.

Integration: a post-hoc evaluation after re-ID training. No retraining
needed — uses cosine similarity on existing embeddings.

Pair sampling:
  - Positive pairs: two clips from the same actor (different motions)
  - Negative pairs: two clips from different actors (matched 1:1)
  - All pairs from unseen_val only (true held-out)

Metrics:
  - AUC-ROC: overall discriminability
  - EER: Equal Error Rate (where FAR = FRR)
  - TAR@FAR=1%: True Accept Rate at 1% False Accept Rate
  - TAR@FAR=0.1%: True Accept Rate at 0.1% False Accept Rate

Usage:
  After collecting embeddings Z and labels y from the re-ID final eval,
  call evaluate_linkage(Z, y).
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import roc_curve, roc_auc_score


def sample_linkage_pairs(
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_pos_per_actor: int = 50,
    max_total_pairs: int = 100_000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample balanced positive/negative pairs from embeddings.
    
    Args:
        embeddings: (N, D) L2-normalized embeddings
        labels: (N,) integer class labels
        max_pos_per_actor: cap positive pairs per actor to avoid dominance
        max_total_pairs: cap total pairs (pos + neg)
        seed: random seed
        
    Returns:
        sims: (P,) cosine similarities for each pair
        pair_labels: (P,) binary labels (1 = same person, 0 = different)
        pair_info: not returned, but pairs are balanced 1:1
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    
    # Index: class -> list of sample indices
    class_to_idx = {}
    for c in classes:
        class_to_idx[c] = np.where(labels == c)[0]
    
    # Sample positive pairs
    pos_i, pos_j = [], []
    for c in classes:
        idx = class_to_idx[c]
        if len(idx) < 2:
            continue
        # Sample pairs within this class
        n_pairs = min(max_pos_per_actor, len(idx) * (len(idx) - 1) // 2)
        for _ in range(n_pairs):
            a, b = rng.choice(len(idx), size=2, replace=False)
            pos_i.append(idx[a])
            pos_j.append(idx[b])
    
    pos_i = np.array(pos_i)
    pos_j = np.array(pos_j)
    n_pos = len(pos_i)
    
    if n_pos == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Cap if too many
    if n_pos > max_total_pairs // 2:
        keep = rng.choice(n_pos, size=max_total_pairs // 2, replace=False)
        pos_i = pos_i[keep]
        pos_j = pos_j[keep]
        n_pos = len(pos_i)
    
    # Sample negative pairs (same count as positive)
    neg_i, neg_j = [], []
    all_classes = list(class_to_idx.keys())
    attempts = 0
    while len(neg_i) < n_pos and attempts < n_pos * 10:
        c1, c2 = rng.choice(len(all_classes), size=2, replace=False)
        c1, c2 = all_classes[c1], all_classes[c2]
        if len(class_to_idx[c1]) == 0 or len(class_to_idx[c2]) == 0:
            attempts += 1
            continue
        a = rng.choice(class_to_idx[c1])
        b = rng.choice(class_to_idx[c2])
        neg_i.append(a)
        neg_j.append(b)
        attempts += 1
    
    neg_i = np.array(neg_i[:n_pos])
    neg_j = np.array(neg_j[:n_pos])
    n_neg = len(neg_i)
    
    # Compute cosine similarities
    # Embeddings are already L2-normalized, so cosine sim = dot product
    pos_sims = np.sum(embeddings[pos_i] * embeddings[pos_j], axis=1)
    neg_sims = np.sum(embeddings[neg_i] * embeddings[neg_j], axis=1)
    
    sims = np.concatenate([pos_sims, neg_sims])
    pair_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    
    return sims, pair_labels


def compute_eer(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> Tuple[float, float]:
    """Compute Equal Error Rate from ROC curve."""
    fnr = 1.0 - tpr
    # Find where FPR and FNR cross
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    eer_threshold = float(thresholds[idx])
    return eer, eer_threshold


def compute_tar_at_far(
    fpr: np.ndarray, tpr: np.ndarray, target_far: float
) -> float:
    """Compute TAR (True Accept Rate) at a given FAR (False Accept Rate)."""
    # Find the largest TPR where FPR <= target_far
    valid = fpr <= target_far
    if not valid.any():
        return 0.0
    return float(tpr[valid][-1])


def evaluate_linkage(
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_pos_per_actor: int = 50,
    max_total_pairs: int = 100_000,
    seed: int = 42,
) -> Dict[str, object]:
    """Full linkage attack evaluation.
    
    Args:
        embeddings: (N, D) L2-normalized embeddings (from unseen_val)
        labels: (N,) integer class labels
        
    Returns:
        Dictionary with AUC, EER, TAR@FAR metrics and pair statistics.
    """
    sims, pair_labels = sample_linkage_pairs(
        embeddings, labels, max_pos_per_actor, max_total_pairs, seed
    )
    
    if len(sims) == 0:
        return {
            "auc": float("nan"),
            "eer": float("nan"),
            "eer_threshold": float("nan"),
            "tar_at_far_1pct": float("nan"),
            "tar_at_far_01pct": float("nan"),
            "n_positive_pairs": 0,
            "n_negative_pairs": 0,
            "mean_pos_sim": float("nan"),
            "mean_neg_sim": float("nan"),
        }
    
    n_pos = int(pair_labels.sum())
    n_neg = len(pair_labels) - n_pos
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(pair_labels, sims)
    auc = float(roc_auc_score(pair_labels, sims))
    
    # EER
    eer, eer_thresh = compute_eer(fpr, tpr, thresholds)
    
    # TAR@FAR
    tar_1pct = compute_tar_at_far(fpr, tpr, 0.01)
    tar_01pct = compute_tar_at_far(fpr, tpr, 0.001)
    tar_5pct = compute_tar_at_far(fpr, tpr, 0.05)
    
    # Similarity statistics
    pos_sims = sims[:n_pos]
    neg_sims = sims[n_pos:]
    
    # Binary accuracy at EER threshold
    pred_at_eer = (sims >= eer_thresh).astype(np.int64)
    true_int = pair_labels.astype(np.int64)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = float(accuracy_score(true_int, pred_at_eer))
    prec = float(precision_score(true_int, pred_at_eer, zero_division=0.0))
    rec = float(recall_score(true_int, pred_at_eer, zero_division=0.0))
    f1 = float(f1_score(true_int, pred_at_eer, zero_division=0.0))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "eer": eer,
        "eer_threshold": eer_thresh,
        "tar_at_far_5pct": tar_5pct,
        "tar_at_far_1pct": tar_1pct,
        "tar_at_far_01pct": tar_01pct,
        "n_positive_pairs": n_pos,
        "n_negative_pairs": n_neg,
        "mean_pos_sim": float(pos_sims.mean()),
        "std_pos_sim": float(pos_sims.std()),
        "mean_neg_sim": float(neg_sims.mean()),
        "std_neg_sim": float(neg_sims.std()),
    }


def print_linkage_results(metrics: Dict[str, object], prefix: str = ""):
    """Pretty-print linkage results."""
    p = f"  {prefix}" if prefix else "  "
    print(f"{p}Linkage Attack Results:")
    print(f"{p}  Accuracy at EER threshold: {metrics['accuracy']:.4f}")
    print(f"{p}  Precision at EER threshold: {metrics['precision']:.4f}")
    print(f"{p}  Recall at EER threshold: {metrics['recall']:.4f}")
    print(f"{p}  F1 at EER threshold: {metrics['f1']:.4f}")
    print(f"{p}  AUC-ROC:        {metrics['auc']:.4f}")
    print(f"{p}  EER:            {metrics['eer']:.4f} (threshold={metrics['eer_threshold']:.4f})")
    print(f"{p}  TAR@FAR=5%:     {metrics['tar_at_far_5pct']:.4f}")
    print(f"{p}  TAR@FAR=1%:     {metrics['tar_at_far_1pct']:.4f}")
    print(f"{p}  TAR@FAR=0.1%:   {metrics['tar_at_far_01pct']:.4f}")
    print(f"{p}  Pairs: {metrics['n_positive_pairs']:,} pos / {metrics['n_negative_pairs']:,} neg")
    print(f"{p}  Pos sim: {metrics['mean_pos_sim']:.4f} ± {metrics['std_pos_sim']:.4f}")
    print(f"{p}  Neg sim: {metrics['mean_neg_sim']:.4f} ± {metrics['std_neg_sim']:.4f}")



if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    
    # Simulate 100 actors, 20 clips each, 128-dim embeddings
    n_actors = 100
    clips_per = 20
    dim = 128
    
    # Each actor gets a random centroid, clips are noisy versions
    labels = np.repeat(np.arange(n_actors), clips_per)
    centroids = np.random.randn(n_actors, dim)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    
    noise = 0.3 * np.random.randn(n_actors * clips_per, dim)
    embeddings = centroids[labels] + noise
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    print("Synthetic test: 100 actors, 20 clips each, noise=0.3")
    metrics = evaluate_linkage(embeddings, labels)
    print_linkage_results(metrics)
    
    # Higher noise = harder
    noise2 = 0.8 * np.random.randn(n_actors * clips_per, dim)
    embeddings2 = centroids[labels] + noise2
    embeddings2 /= np.linalg.norm(embeddings2, axis=1, keepdims=True)
    
    print("\nSynthetic test: 100 actors, 20 clips each, noise=0.8")
    metrics2 = evaluate_linkage(embeddings2, labels)
    print_linkage_results(metrics2)
