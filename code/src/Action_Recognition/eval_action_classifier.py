"""Train the action classifier on original G1 and on each anonymized
G1 variant, then compare their accuracy. See Action_Recognition/README.md
for usage and the rationale behind this utility check."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE       = Path(__file__).resolve().parent
_ROOT       = _HERE.parent
_MODEL_SRC  = _ROOT / 'src' / 'ProtoGCN'   # backbone implementation lives here
_PKL        = _HERE / 'data' / 'pkl'

for _p in [str(_HERE), str(_MODEL_SRC), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from g1_to_action_pkl import CATEGORIES, VARIANT_ROOTS    # noqa: E402

CLIP_LEN = 100
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Human-readable labels for the comparison table. The GRL baseline was
# produced by a separate GRL sanitizer not included in this release; its
# variant is skipped when the corresponding dataset directory is absent.
VARIANT_LABELS = {
    'pmr_contrast': 'PMR Contrast',
    'sanitized':    'PMR Contrast',   # alias
    'pmr_random':   'PMR Random',
    'grl':          'GRL',
}


def load_label_map(pkl_dir: Path) -> Tuple[List[str], int]:
    """
    Load label_map.json written by the converter.
    Falls back to the default 20 CATEGORIES if the file doesn't exist.
    Returns (idx2label list, num_classes).
    """
    p = pkl_dir / 'label_map.json'
    if p.exists():
        with open(p) as f:
            lm = json.load(f)
        return lm['idx2label'], lm['num_classes']
    return CATEGORIES, len(CATEGORIES)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class _G1Dataset(Dataset):
    def __init__(self, data: List[dict], clip_len: int, augment: bool):
        self.data, self.clip_len, self.augment = data, clip_len, augment

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        ann = self.data[idx]
        kp  = ann['keypoint']
        T   = kp.shape[1]

        if T <= self.clip_len:
            reps = (self.clip_len // T) + 1
            kp   = np.tile(kp, (1, reps, 1, 1))[:, :self.clip_len]
        else:
            start = (np.random.randint(0, T - self.clip_len + 1)
                     if self.augment else (T - self.clip_len) // 2)
            kp = kp[:, start: start + self.clip_len]

        return torch.from_numpy(kp).float().squeeze(0), ann['label']


def _collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, 0).unsqueeze(1), torch.tensor(ys, dtype=torch.long)


def make_loader(data: List[dict], batch_size: int,
                num_workers: int, augment: bool) -> DataLoader:
    return DataLoader(
        _G1Dataset(data, CLIP_LEN, augment),
        batch_size=batch_size, shuffle=augment,
        num_workers=num_workers, collate_fn=_collate,
        pin_memory=torch.cuda.is_available(), drop_last=augment,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_action_classifier(num_classes: int) -> nn.Module:
    """Build the action classifier (graph backbone + linear head)."""
    # Backbone implementation lives under code/src/ProtoGCN/; imported lazily
    # so this module loads without requiring it on the path at import time.
    from protogcn.models.gcns.protogcn import ProtoGCN  # type: ignore[import]

    backbone = ProtoGCN(
        graph_cfg=dict(layout='bones_seed_g1', mode='random',
                       num_filter=8, init_off=0.04, init_std=0.02),
        in_channels=3,
        base_channels=96,
        num_person=1,
        num_prototype=400,
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
    )
    head = nn.Linear(384, num_classes)
    pool = nn.AdaptiveAvgPool2d(1)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = head
            self.pool = pool

        def forward(self, x):
            N, M = x.shape[0], x.shape[1]
            feat, _ = self.backbone(x)
            feat = feat.reshape(N * M, feat.shape[2], feat.shape[3], feat.shape[4])
            feat = self.pool(feat).reshape(N, M, -1).mean(dim=1)
            return self.head(feat)

    return _Model()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor,
                  topk: Tuple[int, ...] = (1, 5)) -> Dict[str, float]:
    maxk = max(topk)
    _, pred = logits.topk(min(maxk, logits.shape[1]), dim=1, largest=True, sorted=True)
    pred    = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    return {
        f'top{k}': correct[:min(k, logits.shape[1])].any(dim=0).float().mean().item() * 100
        for k in topk
    }


@torch.no_grad()
def evaluate_full(model, loader, device
                  ) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    model.eval()
    all_logits, all_labels = [], []
    for xs, ys in loader:
        all_logits.append(model(xs.to(device)).cpu())
        all_labels.append(ys)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    overall = topk_accuracy(logits, labels)
    per_class: Dict[int, Dict[str, float]] = {}
    for c in range(logits.shape[1]):
        mask = labels == c
        n    = int(mask.sum())
        if n == 0:
            continue
        m = topk_accuracy(logits[mask], labels[mask])
        per_class[c] = {**m, 'n': n}
    return overall, per_class


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total = 0.0
    for xs, ys in loader:
        xs, ys = xs.to(device), ys.to(device)
        loss = F.cross_entropy(model(xs), ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


def _save_ckpt(path: Path, epoch: int, model: nn.Module,
               optimizer, scheduler) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch':     epoch,
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, path)


def train_and_eval(
    label: str,
    train_data: List[dict],
    test_data: List[dict],
    val_data: List[dict],
    num_classes: int,
    args: argparse.Namespace,
    load_ckpt: Optional[Path],   # explicit path -> load weights and SKIP training
    save_ckpt: Optional[Path],   # auto-save path (used for both saving and resuming)
) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    """
    Build the action classifier, optionally train, then evaluate on test_data.

    Modes (controlled by args.resume and load_ckpt):
      load_ckpt set  -> load weights, skip training, evaluate only
      args.resume    -> if save_ckpt exists: restore model+optimizer+scheduler+epoch
                        and continue training for remaining epochs
      otherwise      -> train from scratch
    """
    print(f'\n{"=" * 62}')
    print(f'  {label}')
    print(f'  train: {len(train_data):,}  |  val: {len(val_data):,}  |  test: {len(test_data):,}')
    print(f'  classes: {num_classes}')
    print(f'{"=" * 62}')

    model = build_action_classifier(num_classes).to(DEVICE)

    # -- Evaluate-only mode (explicit checkpoint path) -----------------------
    if load_ckpt and load_ckpt.exists():
        ckpt = torch.load(load_ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt.get('model', ckpt), strict=False)
        print(f'  Loaded checkpoint (epoch {ckpt.get("epoch", "?")})  '
              f'keys: {list(ckpt.keys() if isinstance(ckpt, dict) else ["weights"])}')
        print('  Skipping training.')

    # -- Train (from scratch or resumed) ------------------------------------
    else:
        train_loader = make_loader(train_data, args.batch_size, args.num_workers, augment=True)
        monitor_data  = val_data if val_data else test_data
        monitor_label = 'val' if val_data else 'test(monitor)'
        monitor_loader = make_loader(monitor_data, args.batch_size,
                                     args.num_workers, augment=False) if monitor_data else None

        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr,
            momentum=0.9, weight_decay=5e-4, nesterov=True,
        )
        # T_max covers the full target epochs so the LR schedule is consistent
        # whether we train in one shot or resume partway through.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-4,
        )

        start_epoch = 1

        # -- Resume from auto-save checkpoint --------------------------------
        if args.resume and save_ckpt and save_ckpt.exists():
            ckpt        = torch.load(save_ckpt, map_location=DEVICE, weights_only=False)
            saved_epoch = ckpt.get('epoch', 0)
            model.load_state_dict(ckpt['model'])

            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            else:
                print('  [warn] Checkpoint has no optimizer state — '
                      'resuming with fresh optimizer.')

            if 'scheduler' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler'])
            else:
                # Fast-forward the LR schedule to the correct position so
                # cosine annealing continues from the right LR, not the peak.
                for _ in range(saved_epoch):
                    scheduler.step()
                print(f'  [warn] Checkpoint has no scheduler state — '
                      f'fast-forwarded {saved_epoch} steps.')

            start_epoch = saved_epoch + 1
            print(f'  Resumed from epoch {saved_epoch} / {args.epochs}  '
                  f'-> continuing from epoch {start_epoch}')

            if start_epoch > args.epochs:
                print(f'  Training already complete ({args.epochs} epochs). '
                      'Increase --epochs to train further.')
                start_epoch = args.epochs + 1   # skip the loop

        if start_epoch == 1:
            print(f'\n  Training from scratch for {args.epochs} epochs...')
        elif start_epoch <= args.epochs:
            remaining = args.epochs - start_epoch + 1
            print(f'\n  Resuming: {remaining} epochs remaining '
                  f'(epochs {start_epoch}-{args.epochs})...')

        best_top1 = 0.0
        for epoch in range(start_epoch, args.epochs + 1):
            loss = train_epoch(model, train_loader, optimizer, DEVICE)
            scheduler.step()

            if monitor_loader and args.eval_every > 0 and epoch % args.eval_every == 0:
                m, _ = evaluate_full(model, monitor_loader, DEVICE)
                best_top1 = max(best_top1, m['top1'])
                print(f'  Epoch {epoch:3d}/{args.epochs}  loss={loss:.4f}  '
                      f'{monitor_label}_top1={m["top1"]:.1f}%  '
                      f'top5={m["top5"]:.1f}%  best={best_top1:.1f}%')
            else:
                print(f'  Epoch {epoch:3d}/{args.epochs}  loss={loss:.4f}')

            # Periodic checkpoint (every --save-every epochs and at the end)
            if save_ckpt and (epoch % args.save_every == 0 or epoch == args.epochs):
                _save_ckpt(save_ckpt, epoch, model, optimizer, scheduler)
                print(f'  Checkpoint saved (epoch {epoch}) -> {save_ckpt}')

    print(f'\n  Evaluating on test ({len(test_data):,} clips)...')
    overall, per_class = evaluate_full(
        model,
        make_loader(test_data, args.batch_size, args.num_workers, augment=False),
        DEVICE,
    )
    return overall, per_class


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_results_table(label: str, n: int,
                        overall: Dict[str, float],
                        per_class: Dict[int, Dict[str, float]],
                        idx2label: List[str]) -> None:
    print(f'\n  {label}  (n={n:,})')
    print(f'    Overall  top-1: {overall["top1"]:6.2f}%   top-5: {overall["top5"]:6.2f}%')
    print(f'    {"Class":<36} {"n":>6}  {"top-1":>7}  {"top-5":>7}')
    print(f'    {"-"*36} {"------":>6}  {"-------":>7}  {"-------":>7}')
    for idx, name in enumerate(idx2label):
        if idx not in per_class:
            continue
        m = per_class[idx]
        print(f'    {name:<36} {m["n"]:>6,}  {m["top1"]:>6.1f}%  {m["top5"]:>6.1f}%')


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_pkl(pkl_dir: Path, split: str, max_clips: int = 0) -> List[dict]:
    path = pkl_dir / f'{split}.pkl'
    if not path.exists():
        return []
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data[:max_clips] if max_clips > 0 else data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    san_choices = [k for k in VARIANT_ROOTS if k != 'original']
    p = argparse.ArgumentParser(
        description='Action classifier: original G1 vs one or more anonymized variants',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--fast', action='store_true',
                   help='Use balanced 30k fast subsets (data/pkl/{variant}_fast/)')
    p.add_argument('--label-type', choices=['category', 'action_type'], default='category',
                   help='category (20 classes) | action_type (fine-grained)')
    p.add_argument('--sanitized-variants', nargs='+', default=['pmr_contrast'],
                   choices=san_choices, metavar='VARIANT',
                   help=f'Sanitized variants to evaluate. Choices: {san_choices}. '
                        'Default: pmr_contrast. '
                        'Example: --sanitized-variants pmr_contrast pmr_random grl')
    p.add_argument('--original-pkl-dir', type=str, default=None,
                   help='Override directory for original G1 pkl files')
    p.add_argument('--epochs', type=int, default=30,
                   help='Training epochs per model (30 quick / 150 full)')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--max-clips', type=int, default=0,
                   help='Cap clips loaded per split (0 = all)')
    p.add_argument('--ckpt-orig', type=str, default=None,
                   help='Explicit checkpoint for Model A (original): load and skip training')
    p.add_argument('--resume', action='store_true',
                   help='Resume all models from their auto-saved checkpoints')
    p.add_argument('--save-every', type=int, default=10,
                   help='Save checkpoint every N epochs (default: 10)')
    p.add_argument('--eval-every', type=int, default=10,
                   help='Evaluate on val/test every N epochs (0 = end only)')
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', type=str, default=None,
                   help='Path to save JSON results')
    return p.parse_args()


def _resolve_pkl_dir(variant: str, suffix: str) -> Path:
    """Return the pkl directory for a given variant name + fast/label suffix."""
    canonical = 'pmr_contrast' if variant == 'sanitized' else variant
    return _PKL / f'{canonical}{suffix}'


def main():
    args = parse_args()

    lt_suffix = '' if args.label_type == 'category' else f'_{args.label_type}'
    suffix    = ('_fast' if args.fast else '') + lt_suffix

    orig_dir = Path(args.original_pkl_dir) if args.original_pkl_dir \
               else _resolve_pkl_dir('original', suffix)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    idx2label, num_classes = load_label_map(orig_dir)

    mode_str = '[FAST -- 30k balanced subset]' if args.fast else '[FULL dataset]'
    print(f'Device             : {DEVICE}')
    print(f'Mode               : {mode_str}')
    print(f'Label type         : {args.label_type}  ({num_classes} classes)')
    print(f'Original pkl       : {orig_dir}')
    print(f'Sanitized variants : {args.sanitized_variants}')
    print(f'Epochs             : {args.epochs}  per model')
    print(f'Resume             : {"yes" if args.resume else "no"}')
    print(f'Save every         : {args.save_every} epochs')

    # -----------------------------------------------------------------------
    # Train Model A  (original)
    # -----------------------------------------------------------------------
    orig_train = load_pkl(orig_dir, 'train', args.max_clips)
    orig_val   = load_pkl(orig_dir, 'val',   args.max_clips)
    orig_test  = load_pkl(orig_dir, 'test',  args.max_clips)

    if not orig_train:
        flag = '--fast' if args.fast else ''
        print('\n[ERROR] Original training data not found.')
        print(f'  Run:  python Action_Recognition/g1_to_action_pkl.py --variant original {flag}')
        sys.exit(1)

    overall_orig, per_class_orig = train_and_eval(
        label       = 'MODEL A  --  ORIGINAL G1',
        train_data  = orig_train,
        test_data   = orig_test,
        val_data    = orig_val,
        num_classes = num_classes,
        args        = args,
        load_ckpt   = Path(args.ckpt_orig) if args.ckpt_orig else None,
        save_ckpt   = orig_dir / 'model.pt',
    )

    # -----------------------------------------------------------------------
    # Train one model per sanitized variant
    # -----------------------------------------------------------------------
    san_results: Dict[str, dict] = {}

    for i, variant in enumerate(args.sanitized_variants, start=2):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        pkl_dir = _resolve_pkl_dir(variant, suffix)
        nice    = VARIANT_LABELS.get(variant, variant.upper())

        san_train = load_pkl(pkl_dir, 'train', args.max_clips)
        san_val   = load_pkl(pkl_dir, 'val',   args.max_clips)
        san_test  = load_pkl(pkl_dir, 'test',  args.max_clips)

        if not san_train:
            flag = '--fast' if args.fast else ''
            print(f'\n[SKIP] No training data for variant "{variant}" at {pkl_dir}')
            print(f'  Run:  python Action_Recognition/g1_to_action_pkl.py '
                  f'--variant {variant} {flag}')
            continue

        overall, per_class = train_and_eval(
            label       = f'MODEL {chr(64 + i)}  --  {nice} G1',
            train_data  = san_train,
            test_data   = san_test,
            val_data    = san_val,
            num_classes = num_classes,
            args        = args,
            load_ckpt   = None,
            save_ckpt   = pkl_dir / 'model.pt',
        )
        san_results[variant] = {
            'label':     nice,
            'n_test':    len(san_test),
            'overall':   overall,
            'per_class': per_class,
        }

    # -----------------------------------------------------------------------
    # Per-model detailed tables
    # -----------------------------------------------------------------------
    print('\n' + '=' * 66)
    print(f'  FINAL COMPARISON  {mode_str}')
    print('=' * 66)

    print_results_table('Model A -- Original G1  (train -> test)',
                        len(orig_test), overall_orig, per_class_orig, idx2label)

    for v, r in san_results.items():
        print_results_table(
            f'Model {chr(65 + list(san_results.keys()).index(v) + 1)} -- {r["label"]} G1  (train -> test)',
            r['n_test'], r['overall'], r['per_class'], idx2label,
        )

    # -----------------------------------------------------------------------
    # Summary comparison table
    # -----------------------------------------------------------------------
    variants_done = [v for v in san_results]
    col_w = 13
    print()
    header = f'  {"Metric":<22}  {"Original":>{col_w}}' + \
             ''.join(f'  {san_results[v]["label"]:>{col_w}}' for v in variants_done)
    print(header)
    print('  ' + '-' * (22 + (col_w + 2) * (1 + len(variants_done))))

    for metric, key in [('top-1 accuracy', 'top1'), ('top-5 accuracy', 'top5')]:
        row = f'  {metric:<22}  {overall_orig[key]:>{col_w}.2f}%'
        for v in variants_done:
            row += f'  {san_results[v]["overall"][key]:>{col_w}.2f}%'
        print(row)

    for metric, key in [('delta top-1', 'top1'), ('delta top-5', 'top5')]:
        row = f'  {metric:<22}  {"baseline":>{col_w}}'
        for v in variants_done:
            d = san_results[v]['overall'][key] - overall_orig[key]
            row += f'  {d:>+{col_w}.2f} pp'
        print(row)

    row = f'  {"utility check":<22}  {"---":>{col_w}}'
    for v in variants_done:
        d1  = san_results[v]['overall']['top1'] - overall_orig['top1']
        tag = 'PASS' if abs(d1) < 5.0 else 'REVIEW'
        row += f'  {tag:>{col_w}}'
    print(row)
    print('=' * 66)

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    def _per_class_named(pc: Dict[int, Dict]) -> Dict:
        return {idx2label[k]: v for k, v in pc.items()}

    save: Dict = {
        'meta': {
            'timestamp':          datetime.now().isoformat(timespec='seconds'),
            'mode':               'fast_30k' if args.fast else 'full',
            'label_type':         args.label_type,
            'sanitized_variants': args.sanitized_variants,
            'epochs':             args.epochs,
            'device':             str(DEVICE),
        },
        'original': {
            'n_test':    len(orig_test),
            'overall':   overall_orig,
            'per_class': _per_class_named(per_class_orig),
        },
    }
    for v, r in san_results.items():
        save[v] = {
            'label':     r['label'],
            'n_test':    r['n_test'],
            'overall':   r['overall'],
            'per_class': _per_class_named(r['per_class']),
            'delta': {
                'top1_pp': round(r['overall']['top1'] - overall_orig['top1'], 4),
                'top5_pp': round(r['overall']['top5'] - overall_orig['top5'], 4),
            },
            'utility_check': 'PASS' if abs(r['overall']['top1'] - overall_orig['top1']) < 5.0
                             else 'REVIEW',
        }

    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode     = 'fast' if args.fast else 'full'
    out_path = Path(args.out) if args.out \
               else _HERE / 'data' / 'results' / f'{mode}_{ts}_results.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(save, f, indent=2)
    print(f'\n  Results saved -> {out_path}')
    print()


if __name__ == '__main__':
    main()
