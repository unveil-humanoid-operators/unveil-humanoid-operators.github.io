#!/usr/bin/env python3
"""Centralized paths for the UNVEIL release.

By default every path resolves relative to this repository root: the
BONES-SEED dataset directories (``metadata/``, ``g1/``, ...) are expected
here, and all generated artifacts (splits, motion cache, checkpoints) are
written under ``artifacts/``. Set the ``BONES_SEED_ROOT`` environment
variable to point at a dataset checkout that lives elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get("BONES_SEED_ROOT", Path(__file__).resolve().parent)
).resolve()

ARTIFACTS_DIR = DATA_ROOT / "artifacts"
SPLITS_DIR = ARTIFACTS_DIR / "splits"
CACHE_DIR = ARTIFACTS_DIR / "cache"
MODELS_DIR = ARTIFACTS_DIR / "models"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_splits_dir(create: bool = False) -> Path:
    return ensure_dir(SPLITS_DIR) if create else SPLITS_DIR


def default_g1_cache_dir(create: bool = False) -> Path:
    cache = CACHE_DIR / "g1_motions"
    return ensure_dir(cache) if create else cache
