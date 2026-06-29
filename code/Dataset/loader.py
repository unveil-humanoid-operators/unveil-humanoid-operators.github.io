"""
loader.py — Download, extract, and verify the BONES-SEED dataset.

BONES-SEED (522 operators, 142K G1-retargeted humanoid motion sequences) is
the dataset used in the UNVEIL paper. It is hosted on Hugging Face at
https://huggingface.co/datasets/bones-studio/seed.

Usage
-----
    # Standard: download to ./bones-seed, auto-unzip, then verify.
    python loader.py

    # Custom destination.
    python loader.py --dest /data/bones-seed

    # Inspect the remote without downloading anything.
    python loader.py --dry-run

    # Skip downloading; only verify an existing local copy.
    python loader.py --verify-only --dest /data/bones-seed

    # Use a specific revision / branch / tag.
    python loader.py --revision main

Dependencies
------------
    pip install huggingface_hub tqdm
    # Optional, for full metadata verification:
    pip install pyarrow

Notes
-----
The full dataset is large (tens of GB). Use --dry-run first to see the file
list, and consider --allow-patterns / --ignore-patterns if you only want a
subset (e.g. only the G1 retargeting):

    python loader.py --allow "g1/**" "metadata/**"
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

REPO_ID = "bones-studio/seed"
REPO_TYPE = "dataset"
DEFAULT_DEST = Path("./bones-seed")

# Top-level entries we expect after a successful extract. Used by verify().
# Sourced from the dataset layout described in the UNVEIL paper supplementary.
EXPECTED_ENTRIES = [
    "g1",                  # Unitree G1 retargeted trajectories (CSV)
    "soma_uniform",        # SOMA Uniform skeleton BVH motions
    "soma_proportional",   # SOMA Proportional (per-actor) BVH motions
    "soma_shapes",         # SMPL shape parameters (.npz)
    "metadata",            # Per-sequence metadata parquet + temporal labels
]

# Key metadata file used as a deeper integrity check.
META_PARQUET = Path("metadata") / "seed_metadata_v003.parquet"

# Expected row count in the metadata parquet (from the paper: 142,220 rows).
EXPECTED_META_ROWS = 142_220


# ── Download ─────────────────────────────────────────────────────────────

def cmd_dry_run() -> None:
    """List remote files without downloading them."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub is required: pip install huggingface_hub")

    api = HfApi()
    info = api.dataset_info(REPO_ID)
    siblings = info.siblings or []
    total_size = sum((s.size or 0) for s in siblings)

    print(f"Repo       : {REPO_ID} ({REPO_TYPE})")
    print(f"Files      : {len(siblings)}")
    if total_size:
        print(f"Total size : {total_size / (1024 ** 3):.2f} GB")
    print()
    print("First entries:")
    for s in siblings[:40]:
        size_str = f"{(s.size or 0) / (1024 ** 2):8.1f} MB" if s.size else "       —  "
        print(f"  {size_str}  {s.rfilename}")
    if len(siblings) > 40:
        print(f"  ... and {len(siblings) - 40} more")


def cmd_download(
    dest: Path,
    revision: str | None,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
) -> None:
    """Pull the dataset snapshot into ``dest``."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is required: pip install huggingface_hub")

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} → {dest}")
    if revision:
        print(f"Revision   : {revision}")
    if allow_patterns:
        print(f"Allow      : {allow_patterns}")
    if ignore_patterns:
        print(f"Ignore     : {ignore_patterns}")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(dest),
        revision=revision,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    print("Download complete.")


# ── Extract ──────────────────────────────────────────────────────────────

def cmd_unzip(dest: Path) -> None:
    """Extract every .zip at the dataset root next to itself, skipping if
    a same-named directory already exists."""
    zips = sorted(dest.glob("*.zip"))
    if not zips:
        print("No .zip files at dataset root — nothing to unzip.")
        return
    for z in zips:
        out_dir = dest / z.stem
        if out_dir.exists():
            print(f"Skip   {z.name}  (already extracted to {out_dir.name}/)")
            continue
        print(f"Unzip  {z.name}  →  {out_dir.name}/")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out_dir)


# ── Verify ───────────────────────────────────────────────────────────────

def _format_gb(bytes_: int) -> str:
    return f"{bytes_ / (1024 ** 3):.2f} GB"


def cmd_verify(dest: Path) -> bool:
    """Sanity-check that the dataset looks like a complete BONES-SEED tree.

    Returns True on success, False if any check fails.
    """
    print(f"Verifying : {dest}")
    if not dest.exists():
        print(f"  FAIL  destination does not exist")
        return False

    # 1. Expected top-level entries.
    missing, present = [], []
    for name in EXPECTED_ENTRIES:
        (present if (dest / name).exists() else missing).append(name)
    print(f"  Top-level entries: {len(present)}/{len(EXPECTED_ENTRIES)} present")
    for name in present:
        print(f"    ✓ {name}")
    for name in missing:
        print(f"    ✗ {name}  (missing)")

    # 2. Sizes and counts.
    file_count = sum(1 for _ in dest.rglob("*") if _.is_file())
    total_size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"  Files: {file_count:,}    total: {_format_gb(total_size)}")

    if file_count == 0:
        print("  FAIL  no files under destination")
        return False

    # 3. Metadata parquet schema + row count.
    meta_path = dest / META_PARQUET
    meta_ok = True
    if not meta_path.exists():
        print(f"  WARN  {META_PARQUET} missing — metadata-driven loaders won't work")
        meta_ok = False
    else:
        try:
            import pyarrow.parquet as pq  # type: ignore
        except ImportError:
            print(
                f"  Note  {META_PARQUET.name} present "
                f"({meta_path.stat().st_size / (1024 ** 2):.1f} MB); "
                "install pyarrow to validate its schema"
            )
        else:
            try:
                table = pq.read_table(meta_path)
                rows, cols = table.num_rows, len(table.column_names)
                print(f"  Metadata: {rows:,} rows × {cols} cols")
                if rows != EXPECTED_META_ROWS:
                    print(
                        f"    WARN  expected {EXPECTED_META_ROWS:,} rows, "
                        f"got {rows:,} (dataset may have been re-revised)"
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL  could not read {META_PARQUET}: {exc}")
                meta_ok = False

    ok = not missing and meta_ok
    print()
    print(f"Result    : {'OK' if ok else 'INCOMPLETE'}")
    return ok


# ── CLI ──────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Download, extract, and verify the BONES-SEED dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                    help=f"Destination directory (default: {DEFAULT_DEST})")
    ap.add_argument("--revision", default=None,
                    help="Hub revision / branch / tag to pin (default: latest main)")
    ap.add_argument("--allow", dest="allow_patterns", nargs="+", default=None,
                    metavar="PATTERN",
                    help="Glob(s) of files to include (e.g. 'g1/**' 'metadata/**')")
    ap.add_argument("--ignore", dest="ignore_patterns", nargs="+", default=None,
                    metavar="PATTERN",
                    help="Glob(s) of files to exclude")
    ap.add_argument("--dry-run", action="store_true",
                    help="List remote files only; download nothing")
    ap.add_argument("--verify-only", action="store_true",
                    help="Skip download; just verify --dest")
    ap.add_argument("--skip-unzip", action="store_true",
                    help="Don't auto-extract .zip files after download")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.dry_run:
        cmd_dry_run()
        return 0

    if args.verify_only:
        return 0 if cmd_verify(args.dest) else 1

    cmd_download(
        dest=args.dest,
        revision=args.revision,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns,
    )
    if not args.skip_unzip:
        cmd_unzip(args.dest)
    return 0 if cmd_verify(args.dest) else 1


if __name__ == "__main__":
    sys.exit(main())
