#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from project_paths import DATA_ROOT, SPLITS_DIR

METADATA_PATH = DATA_ROOT / "metadata" / "seed_metadata_v003.parquet"
OUT_DIR = SPLITS_DIR

RANDOM_STATE = 42
UNSEEN_TEST_FRAC = 0.2  # 20% of actors held out completely
SEEN_VAL_FRAC = 0.25    # 25% of remaining training actors used for seen_val
MIN_MOTIONS_PER_ACTOR = 20


def canonical_motion_key(move_name: str) -> str:
    """Map mirrored and original variants to one canonical key.

    BONES-SEED mirrors are marked with trailing "_M".
    """
    if move_name.endswith("_M"):
        return move_name[:-2]
    return move_name


def split_actors_into_groups(
    actor_ids: List[str],
    unseen_test_frac: float,
    seen_val_frac: float,
    rng: np.random.Generator,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split actor list into three groups:
    - unseen_test: completely held out (unseen_test_frac)
    - pure_train: training only ((1 - unseen_test_frac) * (1 - seen_val_frac))
    - seen_val: in training + validation ((1 - unseen_test_frac) * seen_val_frac)
    """
    actor_ids = list(actor_ids)
    n_actors = len(actor_ids)
    
    # Shuffle for randomness
    idx = np.arange(n_actors)
    rng.shuffle(idx)
    shuffled_actors = [actor_ids[i] for i in idx]
    
    # Split unseen_test
    n_unseen_test = int(np.ceil(unseen_test_frac * n_actors))
    unseen_test = shuffled_actors[:n_unseen_test]
    
    # Split remaining into pure_train and seen_val
    remaining_actors = shuffled_actors[n_unseen_test:]
    n_remaining = len(remaining_actors)
    n_seen_val = int(np.ceil(seen_val_frac * n_remaining))
    
    seen_val = remaining_actors[:n_seen_val]
    pure_train = remaining_actors[n_seen_val:]
    
    return pure_train, seen_val, unseen_test


def main() -> None:
    print("=" * 80)
    print("PHASE 2: THREE-WAY SPLIT GENERATION (unseen_test / pure_train / seen_val)")
    print("=" * 80)

    print(f"Loading metadata: {METADATA_PATH}")
    df = pd.read_parquet(METADATA_PATH)
    print(f"Loaded rows: {len(df):,}")

    # Add canonical key for mirror-safe grouping.
    df = df.copy()
    df["canonical_motion_key"] = df["move_name"].map(canonical_motion_key)

    # Work with original motions to identify eligible actors.
    originals = df[df["is_mirror"] == False].copy()
    actor_counts = originals.groupby("actor_uid").size().sort_values()

    eligible_actor_ids = actor_counts[actor_counts >= MIN_MOTIONS_PER_ACTOR].index.tolist()
    skipped_actor_ids = actor_counts[actor_counts < MIN_MOTIONS_PER_ACTOR].index.tolist()

    print(f"Eligible actors (>= {MIN_MOTIONS_PER_ACTOR} originals): {len(eligible_actor_ids)}")
    print(f"Skipped actors: {len(skipped_actor_ids)}")

    # Perform three-way actor split
    rng = np.random.default_rng(RANDOM_STATE)
    pure_train_actors, seen_val_actors, unseen_test_actors = split_actors_into_groups(
        eligible_actor_ids,
        unseen_test_frac=UNSEEN_TEST_FRAC,
        seen_val_frac=SEEN_VAL_FRAC,
        rng=rng,
    )

    print(f"\nActor split:")
    print(f"  - unseen_test: {len(unseen_test_actors)} actors ({100*len(unseen_test_actors)/len(eligible_actor_ids):.1f}%)")
    print(f"  - pure_train: {len(pure_train_actors)} actors ({100*len(pure_train_actors)/len(eligible_actor_ids):.1f}%)")
    print(f"  - seen_val: {len(seen_val_actors)} actors ({100*len(seen_val_actors)/len(eligible_actor_ids):.1f}%)")

    # Build manifests
    # Test: unseen_test actors' originals only (no mirrors)
    test_df = originals[originals["actor_uid"].isin(unseen_test_actors)].copy()
    test_df["split"] = "test"

    # Train: pure_train + seen_val actors, all motions (originals + mirrors)
    train_canonical_keys = set(
        df[df["actor_uid"].isin(pure_train_actors + seen_val_actors)]["canonical_motion_key"].tolist()
    )
    train_df = df[df["canonical_motion_key"].isin(train_canonical_keys)].copy()
    train_df["split"] = "train"

    # Val: seen_val actors only, originals only (no mirrors)
    val_df = originals[originals["actor_uid"].isin(seen_val_actors)].copy()
    val_df["split"] = "val"
    val_canonical_keys = set(val_df["canonical_motion_key"].tolist())

    # Safety checks
    # Check 1: no canonical overlap between test and train/val
    test_canonical = set(test_df["canonical_motion_key"].tolist())
    train_test_overlap = train_canonical_keys.intersection(test_canonical)
    if train_test_overlap:
        raise RuntimeError(f"Leakage detected: {len(train_test_overlap)} canonical keys overlap between train/test.")

    val_test_overlap = val_canonical_keys.intersection(test_canonical)
    if val_test_overlap:
        raise RuntimeError(f"Leakage detected: {len(val_test_overlap)} canonical keys overlap between val/test.")

    # Check 2: test contains only originals
    if bool(test_df["is_mirror"].any()):
        raise RuntimeError("Test set contains mirrored motions; this should never happen.")

    # Check 3: pure_train and seen_val have no actor overlap
    pure_train_actor_set = set(pure_train_actors)
    seen_val_actor_set = set(seen_val_actors)
    actor_overlap = pure_train_actor_set.intersection(seen_val_actor_set)
    if actor_overlap:
        raise RuntimeError(f"Actor overlap between pure_train and seen_val: {len(actor_overlap)} actors.")

    # Keep only columns needed by downstream loaders plus useful metadata.
    keep_cols = [
        "split",
        "actor_uid",
        "actor_gender",
        "actor_age_yr",
        "actor_height_cm",
        "actor_weight_kg",
        "move_name",
        "canonical_motion_key",
        "is_mirror",
        "package",
        "category",
        "content_type_of_movement",
        "content_body_position",
        "content_uniform_style",
        "move_duration_frames",
        "move_soma_proportional_path",
        "move_soma_uniform_path",
        "move_g1_mujoco_path",
    ]

    train_manifest = train_df[keep_cols].copy()
    val_manifest = val_df[keep_cols].copy()
    test_manifest = test_df[keep_cols].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUT_DIR / "train_manifest.csv"
    val_path = OUT_DIR / "val_manifest.csv"
    test_path = OUT_DIR / "test_manifest.csv"
    summary_path = OUT_DIR / "split_summary.json"

    train_manifest.to_csv(train_path, index=False)
    val_manifest.to_csv(val_path, index=False)
    test_manifest.to_csv(test_path, index=False)

    # Compute per-actor statistics
    per_actor = []
    all_split_actors = set(pure_train_actors + seen_val_actors + unseen_test_actors)
    for actor_uid in all_split_actors:
        a_train = int((train_manifest["actor_uid"] == actor_uid).sum())
        a_val = int((val_manifest["actor_uid"] == actor_uid).sum())
        a_test = int((test_manifest["actor_uid"] == actor_uid).sum())
        
        # Determine actor group
        if actor_uid in unseen_test_actors:
            actor_group = "unseen_test"
        elif actor_uid in seen_val_actors:
            actor_group = "seen_val"
        else:
            actor_group = "pure_train"
        
        per_actor.append({
            "actor_uid": actor_uid,
            "group": actor_group,
            "train_rows": a_train,
            "val_rows": a_val,
            "test_rows": a_test,
        })

    summary: Dict[str, object] = {
        "config": {
            "random_state": RANDOM_STATE,
            "unseen_test_frac": UNSEEN_TEST_FRAC,
            "seen_val_frac": SEEN_VAL_FRAC,
            "min_motions_per_actor": MIN_MOTIONS_PER_ACTOR,
            "test_originals_only": True,
            "train_val_includes_mirrors": True,
        },
        "splits": {
            "unseen_test_actors": len(unseen_test_actors),
            "pure_train_actors": len(pure_train_actors),
            "seen_val_actors": len(seen_val_actors),
        },
        "counts": {
            "total_rows": int(len(df)),
            "original_rows": int((df["is_mirror"] == False).sum()),
            "mirror_rows": int((df["is_mirror"] == True).sum()),
            "eligible_actors": int(len(eligible_actor_ids)),
            "skipped_actors": int(len(skipped_actor_ids)),
            "train_rows": int(len(train_manifest)),
            "val_rows": int(len(val_manifest)),
            "test_rows": int(len(test_manifest)),
            "train_original_rows": int((train_manifest["is_mirror"] == False).sum()),
            "train_mirror_rows": int((train_manifest["is_mirror"] == True).sum()),
            "val_original_rows": int((val_manifest["is_mirror"] == False).sum()),
            "val_mirror_rows": int((val_manifest["is_mirror"] == True).sum()),
            "test_original_rows": int((test_manifest["is_mirror"] == False).sum()),
            "test_mirror_rows": int((test_manifest["is_mirror"] == True).sum()),
        },
        "integrity": {
            "train_test_canonical_overlap": int(len(train_test_overlap)),
            "val_test_canonical_overlap": int(len(val_test_overlap)),
            "train_val_actor_overlap": int(len(actor_overlap)),
            "no_mirrors_in_test": bool((test_manifest["is_mirror"] == False).all()),
        },
        "paths": {
            "train_manifest": str(train_path.relative_to(DATA_ROOT)),
            "val_manifest": str(val_path.relative_to(DATA_ROOT)),
            "test_manifest": str(test_path.relative_to(DATA_ROOT)),
            "summary": str(summary_path.relative_to(DATA_ROOT)),
        },
        "per_actor_counts": per_actor,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSplit outputs written:")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print(f"  - {test_path}")
    print(f"  - {summary_path}")
    print("\nIntegrity checks:")
    print(f"  - Train/test canonical overlap: {summary['integrity']['train_test_canonical_overlap']}")
    print(f"  - Val/test canonical overlap: {summary['integrity']['val_test_canonical_overlap']}")
    print(f"  - Train/val actor overlap: {summary['integrity']['train_val_actor_overlap']}")
    print(f"  - Mirrors in test: {summary['counts']['test_mirror_rows']}")
    print("\nDone.")


if __name__ == "__main__":
    main()
