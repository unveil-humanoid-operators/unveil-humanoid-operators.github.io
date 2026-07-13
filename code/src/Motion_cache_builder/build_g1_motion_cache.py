#!/usr/bin/env python3
"""
Build a one-time memmap cache for BONES-SEED G1 motion CSV files.

Outputs in cache directory:
- motion_data.f32      float32 memmap array of shape (total_frames, num_channels)
- motion_index.csv     per-clip row with path, offset, length
- metadata.json        summary metadata
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# project_paths.py lives at the repo root, one level above this script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA_ROOT, default_g1_cache_dir, default_splits_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build BONES-SEED G1 memmap cache")
    p.add_argument("--data-root", type=str, default=".", help="Path to the dataset root")
    p.add_argument("--splits-dir", type=str, default=None, help="Path to split manifests directory")
    p.add_argument("--output-dir", type=str, default=None, help="Cache output directory")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files")
    p.add_argument("--progress-every", type=int, default=2000, help="Progress print interval")
    return p.parse_args()


def read_g1_motion(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    if "Frame" in df.columns:
        df = df.drop(columns=["Frame"])
    return df.to_numpy(dtype=np.float32)


def norm_relpath(path: str) -> str:
    return str(path).replace("\\", "/").strip().lower()


def collect_g1_paths(splits_dir: str) -> List[str]:
    train_path = os.path.join(splits_dir, "train_manifest.csv")
    val_path = os.path.join(splits_dir, "val_manifest.csv")
    test_path = os.path.join(splits_dir, "test_manifest.csv")

    # val_manifest is optional (for backwards compatibility)
    required = [train_path, test_path]
    if not all(os.path.exists(p) for p in required):
        raise FileNotFoundError("Missing train/test manifests in splits directory")

    dfs = [pd.read_csv(train_path), pd.read_csv(test_path)]
    if os.path.exists(val_path):
        dfs.append(pd.read_csv(val_path))

    col = "move_g1_mujoco_path"
    for df in dfs:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in split manifests")

    paths = pd.concat([df[col] for df in dfs], ignore_index=True).dropna().astype(str)
    paths = sorted(set(norm_relpath(p) for p in paths))
    return paths


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root).resolve()
    if args.splits_dir:
        splits_dir = Path(args.splits_dir)
    elif data_root == DATA_ROOT:
        splits_dir = default_splits_dir(create=False)
    else:
        splits_dir = data_root / "splits"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif data_root == DATA_ROOT:
        output_dir = default_g1_cache_dir(create=True)
    else:
        output_dir = data_root / "cache" / "g1_motions"

    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = output_dir / "motion_data.f32"
    index_path = output_dir / "motion_index.csv"
    meta_path = output_dir / "metadata.json"

    if not args.overwrite and (data_path.exists() or index_path.exists() or meta_path.exists()):
        raise FileExistsError(
            "Cache files already exist. Use --overwrite to rebuild."
        )

    print(f"Data root: {data_root}")
    print(f"Splits dir: {splits_dir}")
    print(f"Output dir: {output_dir}")

    rel_paths = collect_g1_paths(str(splits_dir))
    print(f"Unique G1 paths from manifests: {len(rel_paths):,}")

    # Pass 1: collect valid clips and lengths
    valid: List[Tuple[str, int]] = []
    num_channels = None

    for i, rel in enumerate(rel_paths, start=1):
        fp = data_root / rel
        if not fp.exists():
            continue
        try:
            x = read_g1_motion(str(fp))
            if x.ndim != 2 or x.shape[0] == 0:
                continue
            if num_channels is None:
                num_channels = int(x.shape[1])
            if int(x.shape[1]) != int(num_channels):
                continue
            valid.append((rel, int(x.shape[0])))
        except Exception:
            continue

        if args.progress_every > 0 and i % args.progress_every == 0:
            print(f"Pass1: {i:,}/{len(rel_paths):,} scanned | valid={len(valid):,}")

    if not valid:
        raise RuntimeError("No valid G1 clips found while building cache.")

    total_frames = int(sum(length for _, length in valid))
    print(f"Valid clips: {len(valid):,}")
    print(f"Num channels: {num_channels}")
    print(f"Total frames: {total_frames:,}")

    # Pass 2: write memmap and index
    mem = np.memmap(str(data_path), dtype=np.float32, mode="w+", shape=(total_frames, num_channels))
    rows = []
    offset = 0

    for i, (rel, length) in enumerate(valid, start=1):
        fp = data_root / rel
        x = read_g1_motion(str(fp))
        mem[offset: offset + length] = x
        rows.append({"path": rel, "offset": offset, "length": length})
        offset += length

        if args.progress_every > 0 and i % args.progress_every == 0:
            print(f"Pass2: {i:,}/{len(valid):,} written")

    mem.flush()

    pd.DataFrame(rows).to_csv(index_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_clips": len(valid),
                "total_frames": total_frames,
                "num_channels": int(num_channels),
                "dtype": "float32",
                "data_file": data_path.name,
                "index_file": index_path.name,
            },
            f,
            indent=2,
        )

    print("Cache build complete:")
    print(f"- {data_path}")
    print(f"- {index_path}")
    print(f"- {meta_path}")


if __name__ == "__main__":
    main()
