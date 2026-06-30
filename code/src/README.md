# Who Moved the Robot? Humanoid Datasets Remember Their Operators

**Sihat Afnan, Unnat Jain\*, Habiba Farrukh\*** — University of California, Irvine

---

# UNVEIL

**UNVEIL** is the training and evaluation framework from our paper *Who Moved the Robot? Humanoid Datasets Remember Their Operators*. It recovers operator attributes — gender, age, height, weight, and re-identification — directly from humanoid joint-angle trajectories, with no access to body shape. This page is the supplementary code: everything below is what you need to reproduce the paper's numbers, including the actor-disjoint data splits and the per-task evaluation harness.

---

## Install

```bash
pip install -r requirements.txt
```

Required: `torch >= 2.0`, `numpy`, `pandas`, `scikit-learn`. UNVEIL's default backbone — a prototype-learning network — additionally needs `mmcv`. `torch.compile` (enabled by default) needs `triton`; pass `--no-compile` to skip it if `triton` is unavailable.

---

## Data preparation

Before training, run these two scripts once:

```bash
python Data_split/create_splits.py                      # → artifacts/splits/{train,val,test}_manifest.csv
python Motion_cache_builder/build_g1_motion_cache.py    # → float32 memmap of every G1 trajectory referenced in the manifests
```

`create_splits.py` produces the actor-wise split described under *Train / val / test split* below. `build_g1_motion_cache.py` reads those manifests and packs every referenced G1 CSV into a flat memmap, which the training loop loads in one shot for substantially faster dataloading.

---

## Quick start

```bash
# Quick dry-run (limited data, 1 epoch) — sanity-check that everything is wired up
python src/unveil.py --task reid --max-train 200 --max-test 100 --epochs 1

# UNVEIL on the full dataset, gender classification (uses the default backbone)
python src/unveil.py --format g1 --task gender

# Same setup, all privacy tasks at once
python src/unveil.py --format g1 --task all

# Baseline backbones — same flags, just set --spatial-backbone
python src/unveil.py --spatial-backbone sgn   --format g1 --task reid
python src/unveil.py --spatial-backbone dsgcn --format uniform --task gender
```

---

## Backbones

By default, UNVEIL uses a **prototype-learning network** as its spatial backbone — this is the
configuration reported as UNVEIL in the paper, and you get it without passing any backbone flag.
`sgn` and `dsgcn` are two action-recognition architectures adapted into the same pipeline as
comparison backbones, selected via `--spatial-backbone`.

| `--spatial-backbone` | Description |
|---|---|
| *(default — no flag needed)* | Prototype-learning network. The UNVEIL model reported in the paper. Needs `mmcv`. |
| `sgn` | Semantics-guided network: consumes position + velocity + acceleration as three explicit input streams. |
| `dsgcn` | Dynamic spatial GCN with body-part typing. |

All three share the same kinematic input pipeline, prediction heads, SupCon warmup, and
actor-disjoint evaluation; they differ only in the spatial backbone. Each consumes the raw G1
(or BVH) joint trajectory — body shape is never provided.

### Heads (shared across all backbones)

- Re-ID: `Linear(emb_dim, num_actors)` + CE-with-label-smoothing + SupCon (warmup per backbone)
- Gender: `Linear(emb_dim, 2)` + CE-with-label-smoothing + SupCon
- Age / Height / Weight: `Linear(emb_dim, 1)` + MSE + SupCon on discretized targets
  (5-year / 5 cm / 5 kg bins)

### Defaults

Per-backbone training defaults (learning rate, batch size, SupCon warmup, channel widths, etc.)
live in `SPATIAL_BACKBONE_DEFAULTS` in `src/unveil.py` and are filled in automatically; any of
them can be overridden from the CLI (see below). Shared defaults: AdamW, weight decay `1e-4`,
gradient clip norm `5.0`, dropout `0.5`, early-stopping patience `40` evaluation cycles, and a
max sequence length of `256` frames @ 30 fps (downsampled from 120 fps).

---

## CLI reference

### Training arguments

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 100 | Number of training epochs |
| `--lr` | varies by `--spatial-backbone` | Learning rate |
| `--batch-size` | varies by `--spatial-backbone` | Batch size |
| `--weight-decay` | 1e-4 | AdamW weight decay |
| `--label-smoothing` | 0.05 | Cross-entropy label smoothing |
| `--lambda-supcon` | 0.1 | SupCon loss weight (0 = CE only) |
| `--lambda-proto` | 0.1 | Memory contrastive loss weight (default backbone only) |
| `--supcon-warmup` | varies by `--spatial-backbone` | Epoch to start contrastive losses |
| `--supcon-temp` | 0.07 | SupCon temperature |
| `--early-stop` | 40 | Early stopping patience (eval cycles) |
| `--eval-every` | 1 | Evaluate every N epochs |
| `--seed` | 42 | Random seed |

### Architecture arguments

| Argument | Applies to | Default | Description |
|---|---|---|---|
| `--emb-dim` | all | 256 | Embedding dimension |
| `--dim1` | sgn | 256 | Feature dimension |
| `--seg` | sgn | 64 | Temporal segments |
| `--base-channels` | dsgcn, default | 64 / 96 | Base channel count |
| `--num-stages` | dsgcn, default | 10 | Number of spatiotemporal blocks |
| `--num-prototype` | default | 100 | Number of latent prototypes |
| `--dropout` | dsgcn, default | 0.5 | Dropout rate |
| `--variance-percentile` | all (BVH) | varies by `--spatial-backbone` | BVH channel variance filtering (0 = keep all) |

---

## Checkpoint layout

Each `--spatial-backbone` × `--format` × `--task` combination writes to its own directory to prevent collisions:

```
artifacts/models/unveil/<spatial-backbone>/actor_holdout_split_<format>/<task>/
├── best_model.pt
├── checkpoint_epoch010.pt
├── checkpoint_epoch020.pt
├── final_<format>_<task>.pt
└── final_metrics_<format>_<task>.json
```

---

## Train / val / test split

The split is **actor-level**: every actor's motion sequences land entirely in one of `pure_train`, `seen_val`, or `unseen_test`, so reported test metrics are operator-disjoint from training. Split artifacts are written to `artifacts/splits/`.

### Files

| File | Rows (excl. header) | Description |
|---|---|---|
| `train_manifest.csv` | 111,857 | Training rows (originals + mirrors) |
| `val_manifest.csv` | 15,233 | Held-out demos of *seen* actors (validation signal during training) |
| `test_manifest.csv` | 15,002 | All demos of completely *unseen* actors (final reported result) |
| `split_summary.json` | — | Config, integrity checks, and per-actor row counts |
| `top20_action_types_per_category.csv` | 368 | Per-category action whitelist used for category-level analyses |

### Actor partition

492 of the 522 raw actors are eligible (30 are skipped for having fewer than 20 motions). The eligible actors are partitioned as:

| Group | Actors | Description |
|---|---|---|
| `pure_train` | 294 | Appear only in training |
| `seen_val` | 99 | Same actor appears in both train and val: 80 % of their demos → train, 20 % → val |
| `unseen_test` | 99 | Held out entirely; used only for the final test |

---

## Acknowledgments

We thank the authors of DS-GCN, ProtoGCN, and pyskl, whose code we adapted
for the backbones bundled under `src/`.
