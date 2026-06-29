"""Convert G1 CSV motion data into the pickle format consumed by the unveil
action classifier. See Action_Recognition/README.md for usage."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # Action_Recognition/
_ROOT = _HERE.parent                             # bones-seed/
_DATA = _HERE / 'data' / 'pkl'                  # Action_Recognition/data/pkl/

# Maps variant name -> subdirectory under bones-seed/
VARIANT_ROOTS: Dict[str, Optional[str]] = {
    'original':    None,                        # bones-seed/ itself
    'pmr_contrast': 'g1_sanitized_pmr_contrast',
    'sanitized':   'g1_sanitized_pmr_contrast', # backward-compat alias
    'pmr_random':  'g1_sanitized_pmr',
    'grl':         'g1_sanitized_100',
}

# ---------------------------------------------------------------------------
# Action category label map  (20 classes, alphabetical order)
# ---------------------------------------------------------------------------
CATEGORIES: List[str] = [
    'Advanced Locomotion',       # 0
    'Baseline',                  # 1
    'Basic Locomotion Neutral',  # 2
    'Basic Locomotion Styles',   # 3
    'Communication',             # 4
    'Complex Actions',           # 5
    'Consuming',                 # 6
    'Dancing',                   # 7
    'Environments',              # 8
    'Gestures',                  # 9
    'Household',                 # 10
    'Looking and Pointing',      # 11
    'Magic',                     # 12
    'Martial Arts',              # 13
    'Object Interaction',        # 14
    'Object Manipulation',       # 15
    'Other',                     # 16
    'Sports',                    # 17
    'Stunts',                    # 18
    'Unusual Locomotion',        # 19
]
CAT2IDX: Dict[str, int] = {c: i for i, c in enumerate(CATEGORIES)}

G1_CHANNELS = 35

LABEL_COLS = {
    'category':    'category',
    'action_type': 'content_type_of_movement',
}


def build_label_map(manifests: List[Path], label_type: str) -> Tuple[Dict[str, int], List[str]]:
    """
    Build {label_string -> int} mapping.
    'category'    : fixed 20-class CATEGORIES list.
    'action_type' : all unique content_type_of_movement values across manifests (sorted).
    Returns (label2idx, idx2label_list).
    """
    if label_type == 'category':
        return CAT2IDX, CATEGORIES

    col = LABEL_COLS['action_type']
    values: set = set()
    for p in manifests:
        if p.exists():
            df = pd.read_csv(p, usecols=[col]).dropna()
            values.update(df[col].unique())
    idx2label = sorted(values)
    label2idx = {v: i for i, v in enumerate(idx2label)}
    return label2idx, idx2label


# ---------------------------------------------------------------------------
# CSV -> keypoint tensor
# ---------------------------------------------------------------------------

def csv_to_keypoint(csv_path: str) -> Optional[np.ndarray]:
    """
    Read one G1 CSV and return (1, T, 35, 3) float32.
    Last axis: [position, velocity (1st diff), acceleration (2nd diff)].
    Returns None on missing file, read error, or < 2 frames.
    """
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if 'Frame' in df.columns:
        df = df.drop(columns=['Frame'])
    if df.shape[1] != G1_CHANNELS:
        return None

    x = df.values.astype(np.float32)
    if x.shape[0] < 2:
        return None

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    vel = np.zeros_like(x)
    acc = np.zeros_like(x)
    vel[:-1] = np.diff(x, axis=0);  vel[-1] = vel[-2]
    acc[:-1] = np.diff(vel, axis=0); acc[-1] = acc[-2]

    kp = np.stack([x, vel, acc], axis=-1)    # (T, 35, 3)
    return kp[np.newaxis].astype(np.float32)  # (1, T, 35, 3)


# ---------------------------------------------------------------------------
# Helpers shared by both modes
# ---------------------------------------------------------------------------

def _df_to_annotations(df: pd.DataFrame, data_root: Path,
                        label_col: str, label2idx: Dict[str, int],
                        desc: str, verbose: bool) -> List[dict]:
    df = df[df[label_col].isin(label2idx)].reset_index(drop=True)
    anns, skipped = [], 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc, disable=not verbose):
        csv_path = data_root / str(row['move_g1_mujoco_path'])
        kp = csv_to_keypoint(str(csv_path))
        if kp is None:
            skipped += 1
            continue
        anns.append({
            'frame_dir':    str(row['move_name']),
            'total_frames': int(kp.shape[1]),
            'label':        label2idx[row[label_col]],
            'keypoint':     kp,
        })
    if verbose and skipped:
        print(f'  Skipped {skipped:,} missing / unreadable CSVs')
    return anns


def _label_table(label_stats: Dict[str, Dict[int, int]],
                 splits: List[str], idx2label: List[str]) -> None:
    header = f'{"Label":<36} ' + '  '.join(f'{s:>8}' for s in splits if s in label_stats)
    print(header)
    for idx, name in enumerate(idx2label):
        row_str = f'{name:<36} '
        row_str += '  '.join(
            f'{label_stats[s].get(idx, 0):>8}'
            for s in splits if s in label_stats
        )
        if any(label_stats[s].get(idx, 0) > 0 for s in splits if s in label_stats):
            print(row_str)


# ---------------------------------------------------------------------------
# Normal mode: convert one manifest split
# ---------------------------------------------------------------------------

def convert_split(manifest_path: str, data_root: Path,
                  label_col: str, label2idx: Dict[str, int],
                  max_clips: int = 0, verbose: bool = True) -> List[dict]:
    df = pd.read_csv(manifest_path)
    df = df[df['move_g1_mujoco_path'].notna()]
    if max_clips > 0:
        df = df.head(max_clips)
    return _df_to_annotations(df, data_root, label_col, label2idx,
                               os.path.basename(manifest_path), verbose)


# ---------------------------------------------------------------------------
# Fast mode: balanced 15k sample -> stratified 12k/3k train/test split
# ---------------------------------------------------------------------------

def fast_sample_balanced(
    manifest_dir: Path,
    data_root: Path,
    label_col: str,
    label2idx: Dict[str, int],
    target_total: int = 30_000,
    train_ratio: float = 0.8,
    seed: int = 42,
    min_clips: int = 50,
    verbose: bool = True,
) -> Tuple[List[dict], List[dict]]:
    """
    Merge all three manifests, sample up to target_total clips with balanced
    label representation, then do a stratified train/test split.
    Labels with fewer than min_clips clips in the pool are skipped entirely.
    Returns (train_annotations, test_annotations).
    """
    frames = []
    for name in ['train_manifest.csv', 'val_manifest.csv', 'test_manifest.csv']:
        p = manifest_dir / name
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(f'No manifests found in {manifest_dir}')

    pool = pd.concat(frames, ignore_index=True)
    pool = pool[pool['move_g1_mujoco_path'].notna()].copy()
    pool[label_col] = pool[label_col].astype(str).str.strip()
    pool = pool[pool[label_col].isin(label2idx)]
    pool = pool.drop_duplicates(subset='move_name').reset_index(drop=True)

    # groupby is more robust than per-row == comparisons
    groups = {cat: grp.reset_index(drop=True)
              for cat, grp in pool.groupby(label_col, sort=False)}

    # Drop labels that don't have enough clips to be meaningful
    skipped = {cat: len(g) for cat, g in groups.items() if len(g) < min_clips}
    groups  = {cat: g for cat, g in groups.items() if len(g) >= min_clips}

    if verbose and skipped:
        print(f'  Skipped {len(skipped)} labels with < {min_clips} clips: '
              f'{", ".join(f"{c}({n})" for c, n in sorted(skipped.items()))}')

    avail_cats  = sorted(groups.keys())
    n_cats      = len(avail_cats)
    per_cat_cap = target_total // n_cats

    if verbose:
        print(f'  Pool       : {len(pool):,} unique clips, {n_cats} labels (>= {min_clips} clips each)')
        print(f'  Target     : {target_total:,} total  ({per_cat_cap}/category cap)')
        print(f'  Split      : {train_ratio:.0%} train / {1-train_ratio:.0%} test  (stratified)')

    rng = np.random.default_rng(seed)

    train_rows: List[pd.DataFrame] = []
    test_rows:  List[pd.DataFrame] = []
    actual_counts: Dict[str, Tuple[int, int]] = {}

    for cat in avail_cats:
        cat_df = groups[cat]
        n      = min(len(cat_df), per_cat_cap)
        sample = cat_df.sample(n=n, random_state=int(rng.integers(1_000_000)))

        n_train = round(n * train_ratio)
        n_test  = n - n_train

        train_rows.append(sample.iloc[:n_train])
        test_rows.append(sample.iloc[n_train:])
        actual_counts[cat] = (n_train, n_test)

    train_df = pd.concat(train_rows, ignore_index=True).sample(
        frac=1, random_state=int(rng.integers(1_000_000))
    )
    test_df = pd.concat(test_rows, ignore_index=True).sample(
        frac=1, random_state=int(rng.integers(1_000_000))
    )

    if verbose:
        print(f'  After sampling: {len(train_df):,} train clips, {len(test_df):,} test clips')
        print()
        print(f'  {"Label":<36} {"train":>8} {"test":>8}')
        print(f'  {"-"*36} {"--------":>8} {"--------":>8}')
        for cat in avail_cats:
            tr, te = actual_counts[cat]
            print(f'  {cat:<36} {tr:>8,} {te:>8,}')
        print()
        print('Reading train CSVs...')

    train_anns = _df_to_annotations(train_df, data_root, label_col, label2idx, 'train', verbose)

    if verbose:
        print('Reading test CSVs...')
    test_anns = _df_to_annotations(test_df, data_root, label_col, label2idx, 'test', verbose)

    return train_anns, test_anns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Convert G1 CSV data to the unveil action classifier pickle format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--variant',
        choices=list(VARIANT_ROOTS.keys()),
        default='original',
        help=(
            'Which dataset to convert.\n'
            '  original    : bones-seed/g1/csv/\n'
            '  pmr_contrast: g1_sanitized_pmr_contrast/  (PMR + contrastive loss)\n'
            '  pmr_random  : g1_sanitized_pmr/            (PMR random)\n'
            '  grl         : g1_sanitized_100/            (GRL defense)\n'
            '  sanitized   : alias for pmr_contrast (backward compat)'
        ),
    )
    # Fast mode
    p.add_argument('--fast', action='store_true',
                   help=(
                       'Balanced subset mode: sample --fast-total clips equally '
                       'across all categories, then split into 80%% train / 20%% test. '
                       'Output goes to data/pkl/{variant}_fast/'
                   ))
    p.add_argument('--fast-total', type=int, default=30_000,
                   help='Total clips to sample in --fast mode (default: 30000 -> 24k train / 6k test)')
    p.add_argument('--fast-seed', type=int, default=42,
                   help='Random seed for --fast sampling (default: 42)')
    p.add_argument('--min-clips', type=int, default=50,
                   help='Skip labels with fewer than this many clips in the pool (default: 50)')

    # Normal mode
    p.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                   help='Which manifest splits to convert (normal mode only)')
    p.add_argument('--max-per-split', type=int, default=0,
                   help='Cap clips per split (0 = all, normal mode only)')

    # Label type
    p.add_argument('--label-type', choices=['category', 'action_type'], default='category',
                   help=(
                       'category    : 20 action categories (default)\n'
                       'action_type : 149 fine-grained content_type_of_movement values'
                   ))

    # Path overrides
    p.add_argument('--data-root', type=str, default=None,
                   help='Override dataset root (auto-detected from --variant)')
    p.add_argument('--splits-dir', type=str, default=None,
                   help='Override path to the manifest CSVs directory')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Override output directory for the .pkl files')
    p.add_argument('--quiet', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    # -- Resolve data root ---------------------------------------------------
    if args.data_root:
        data_root = Path(args.data_root).resolve()
    else:
        subdir = VARIANT_ROOTS[args.variant]
        data_root = _ROOT if subdir is None else _ROOT / subdir

    splits_dir = Path(args.splits_dir).resolve() if args.splits_dir \
                 else data_root / 'artifacts' / 'splits'

    # -- Build label map -----------------------------------------------------
    all_manifests = [splits_dir / f'{s}_manifest.csv'
                     for s in ['train', 'val', 'test']]
    label2idx, idx2label = build_label_map(all_manifests, args.label_type)
    label_col = LABEL_COLS[args.label_type]
    num_classes = len(idx2label)

    # -- Resolve output dir (includes label_type suffix for non-default) -----
    # Use canonical variant name for output dir (sanitized -> pmr_contrast)
    canonical = 'pmr_contrast' if args.variant == 'sanitized' else args.variant
    lt_suffix  = '' if args.label_type == 'category' else f'_{args.label_type}'
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    elif args.fast:
        out_dir = _DATA / f'{canonical}_fast{lt_suffix}'
    else:
        out_dir = _DATA / f'{canonical}{lt_suffix}'

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Variant    : {args.variant}{"  [FAST]" if args.fast else ""}')
    print(f'Label type : {args.label_type}  ({num_classes} classes)')
    print(f'Label col  : {label_col}')
    print(f'Data root  : {data_root}')
    print(f'Splits dir : {splits_dir}')
    print(f'Output dir : {out_dir}')

    # -- Save label map alongside pkl files ----------------------------------
    label_map = {
        'label_type':  args.label_type,
        'column':      label_col,
        'num_classes': num_classes,
        'idx2label':   idx2label,
        'label2idx':   label2idx,
    }
    with open(out_dir / 'label_map.json', 'w') as f:
        json.dump(label_map, f, indent=2)
    print(f'Label map  : {out_dir}/label_map.json')

    label_stats: Dict[str, Dict[int, int]] = {}

    # -----------------------------------------------------------------------
    # FAST mode
    # -----------------------------------------------------------------------
    if args.fast:
        print(f'\nSampling {args.fast_total:,} balanced clips '
              f'(seed={args.fast_seed})...')
        train_anns, test_anns = fast_sample_balanced(
            manifest_dir=splits_dir,
            data_root=data_root,
            label_col=label_col,
            label2idx=label2idx,
            target_total=args.fast_total,
            train_ratio=0.8,
            seed=args.fast_seed,
            min_clips=args.min_clips,
            verbose=not args.quiet,
        )
        for split, anns in [('train', train_anns), ('test', test_anns)]:
            pkl_path = out_dir / f'{split}.pkl'
            with open(pkl_path, 'wb') as f:
                pickle.dump(anns, f, protocol=4)
            print(f'  Saved {split}.pkl  ({len(anns):,} clips)  -> {pkl_path}')
            counts: Dict[int, int] = {}
            for ann in anns:
                counts[ann['label']] = counts.get(ann['label'], 0) + 1
            label_stats[split] = counts

        print('\n=== Label distribution (fast subset) ===')
        _label_table(label_stats, ['train', 'test'], idx2label)

    # -----------------------------------------------------------------------
    # Normal mode
    # -----------------------------------------------------------------------
    else:
        split_files = {
            'train': 'train_manifest.csv',
            'val':   'val_manifest.csv',
            'test':  'test_manifest.csv',
        }
        for split in args.splits:
            manifest = splits_dir / split_files[split]
            if not manifest.exists():
                print(f'[skip] {manifest} not found')
                continue

            print(f'\n=== {split} ===')
            anns = convert_split(
                str(manifest), data_root,
                label_col=label_col, label2idx=label2idx,
                max_clips=args.max_per_split,
                verbose=not args.quiet,
            )
            pkl_path = out_dir / f'{split}.pkl'
            with open(pkl_path, 'wb') as f:
                pickle.dump(anns, f, protocol=4)
            print(f'  Saved -> {pkl_path}  ({len(anns):,} clips)')

            counts: Dict[int, int] = {}
            for ann in anns:
                counts[ann['label']] = counts.get(ann['label'], 0) + 1
            label_stats[split] = counts

        print('\n=== Label distribution ===')
        _label_table(label_stats, args.splits, idx2label)

    # -----------------------------------------------------------------------
    # Next-step hint
    # -----------------------------------------------------------------------
    lt_flag  = f' --label-type {args.label_type}' if args.label_type != 'category' else ''
    fast_flag = ' --fast' if args.fast else ''
    print(f'\nDone.  Pickle files in: {out_dir}')
    print('\nNext step -- train + evaluate:')
    print(f'  python Action_Recognition/eval_action_classifier.py{fast_flag}{lt_flag}')


if __name__ == '__main__':
    main()
