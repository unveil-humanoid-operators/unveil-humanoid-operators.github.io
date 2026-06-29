# UNVEIL

**UNVEIL** is the training and evaluation framework from our paper *Inverting Retargeting: Humanoid Datasets Remember Their Operators*. It recovers operator attributes — gender, age, height, weight, and re-identification — directly from humanoid joint-angle trajectories, with no access to body shape. This page is the supplementary code: everything below is what you need to reproduce the paper's numbers, including the actor-disjoint data splits and the per-task evaluation harness.

---

## Install

```bash
pip install -r requirements.txt
```

Required: `torch >= 2.0`, `numpy`, `pandas`, `scikit-learn`. Passing `--spatial-backbone proto-mem` additionally needs `mmcv`. `torch.compile` (enabled by default) needs `triton`; pass `--no-compile` to skip it if `triton` is unavailable.

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
python src/unveil.py --spatial-backbone unveil-vanilla --task reid --max-train 200 --max-test 100 --epochs 1

# Vanilla UNVEIL on the full dataset, gender classification
python src/unveil.py --spatial-backbone unveil-vanilla --format g1 --task gender

# Same setup, all privacy tasks at once
python src/unveil.py --spatial-backbone unveil-vanilla --format g1 --task all

# Other options — same flags, just swap --spatial-backbone
python src/unveil.py --spatial-backbone stream-attn --format g1 --task reid
python src/unveil.py --spatial-backbone dyn-graph --format uniform --task gender
python src/unveil.py --spatial-backbone proto-mem --format proportional --task age
```

---

## `unveil-vanilla` in detail

A clean hierarchical spatiotemporal GCN baseline. Unlike `stream-attn` (which feeds position
+ velocity + acceleration as three explicit input streams), `unveil-vanilla` consumes only
the raw **position** trajectory and learns the temporal dynamics inside a kinematic encoder.

### Pipeline

1. **Joint reshape** (no padding for BVH; small zero-padding for G1):
   - **G1** : `(B, 35, T)` raw DoFs → `(B, 15, 3, T)` — 35 G1 DoFs grouped into 15 semantic joints, ≤ 3 DoFs per joint, zero-padded to a uniform 3-channel feature.
   - **BVH** : `(B, 72, T)` rotation channels → `(B, 24, 3, T)` — direct reshape (24 joints × 3 rotation channels).

2. **Kinematic encoder** — a `Conv1d(C_in=3 → 64, kernel=9, padding=4)` applied independently
   to each joint, followed by BN + ReLU. This replaces the explicit velocity/acceleration
   computation in `stream-attn`: the network learns motion dynamics directly from position.

3. **Hierarchical spatial GCN** — 9 weight matrices per spatial layer:
   - **Hierarchy 1 (intra-limb, 5 subgraphs)**: left-arm, right-arm, left-leg, right-leg, torso.
   - **Hierarchy 2 (limb-torso, 4 subgraphs)**: each limb plus its connecting torso joint.
   - Adjacencies are symmetric-normalized D⁻½AD⁻½ with self-loops.

4. **Temporal convolution** — `Conv2d(kernel=(1,9))` shared across all joints, applied after
   each spatial layer.

5. **10 spatiotemporal blocks** with residual connections. Channel / stride schedule:

   | Layers | In → Out | Stride |
   |---|---|---|
   | 1–4 | 64 → 64 | 1 |
   | 5 | 64 → 128 | 2 |
   | 6–7 | 128 → 128 | 1 |
   | 8 | 128 → 256 | 2 |
   | 9–10 | 256 → 256 | 1 |

   After the 10 blocks: `(B, J, 256, T/4)`.

6. **Global average pool** over both joints (J) and time (T) → `(B, 256)`.

7. **Heads** (shared across all `--spatial-backbone` choices):
   - Re-ID: `Linear(256, num_actors)` + CE-with-label-smoothing + SupCon (warmup 20 epochs)
   - Gender: `Linear(256, 2)` + CE-with-label-smoothing + SupCon (warmup 20 epochs)
   - Age / Height / Weight: `Linear(256, 1)` + MSE + SupCon on discretized targets
     (5-year / 5 cm / 5 kg bins, warmup 20 epochs)

### G1 joint grouping (15 joints, all 35 DoFs covered)

| Joint | DoF indices | # |
|---|---|---|
| Pelvis | 0–2 | 3 |
| PelvisRot | 3–5 | 3 |
| LeftHip | 6–8 | 3 |
| LeftKnee | 9 | 1 |
| LeftAnkle | 10–11 | 2 |
| RightHip | 12–14 | 3 |
| RightKnee | 15 | 1 |
| RightAnkle | 16–17 | 2 |
| Waist | 18–20 | 3 |
| LeftShoulder | 21–23 | 3 |
| LeftElbow | 24 | 1 |
| LeftWrist | 25–27 | 3 |
| RightShoulder | 28–30 | 3 |
| RightElbow | 31 | 1 |
| RightWrist | 32–34 | 3 |

Joints with fewer than 3 DoFs are zero-padded to a uniform 3-channel feature. Total ~ 4.5M parameters.

### Defaults (per the architecture spec)

| | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 128 |
| Gradient clip norm | 5.0 |
| Dropout | 0.5 (on the embedding before heads) |
| SupCon warmup | 20 epochs |
| Early-stopping patience | 40 evaluation cycles |
| Max sequence length | 256 frames @ 30 fps (downsampled from 120 fps) |

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
| `--lambda-proto` | 0.1 | Memory contrastive loss weight (proto-mem only) |
| `--supcon-warmup` | varies by `--spatial-backbone` | Epoch to start contrastive losses |
| `--supcon-temp` | 0.07 | SupCon temperature |
| `--early-stop` | 40 | Early stopping patience (eval cycles) |
| `--eval-every` | 1 | Evaluate every N epochs |
| `--seed` | 42 | Random seed |

### Architecture arguments

| Argument | Applies to | Default | Description |
|---|---|---|---|
| `--emb-dim` | all | 256 | Embedding dimension |
| `--dim1` | stream-attn | 256 | Feature dimension |
| `--seg` | stream-attn | 64 | Temporal segments |
| `--base-channels` | dyn-graph, proto-mem | 64 / 96 | Base channel count |
| `--num-stages` | dyn-graph, proto-mem | 10 | Number of spatiotemporal blocks |
| `--num-prototype` | proto-mem | 100 | Number of latent prototypes |
| `--dropout` | dyn-graph, proto-mem, unveil-vanilla | 0.5 | Dropout rate |
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
