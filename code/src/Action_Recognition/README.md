# Action Recognition

## Why we evaluate action recognition on anonymized trajectories

The paper's main result is that humanoid retargeting leaks operator biometrics — gender, age, height, weight, identity — through joint-angle dynamics alone. The natural defence is **anonymization**: transform the trajectories so they no longer carry operator-specific signals.

Anonymization is only useful, however, if the transformed data still supports the downstream tasks robotics practitioners actually care about. Action recognition is the canonical such task: given a motion clip, predict what the operator is doing (walking, dancing, throwing, etc.). If anonymization collapses action accuracy, the resulting data is worthless for policy learning, evaluation, or any other utility task.

This folder lets us run that **utility check**:

1. Train an action classifier on the original (non-anonymized) G1 trajectories. Call this **Model A**.
2. Train an independent action classifier — same architecture, same hyperparameters — on each anonymized variant. Call those **Model B**, **C**, ...
3. Compare top-1 / top-5 accuracy. If an anonymized variant lands within a few points of the original, the privacy gain (lower Re-ID, lower attribute leakage) came **without** a meaningful utility loss.

The paper reports this as the "9 pp action-recognition utility loss" headline: Re-ID dropped from 97.2 % to 16.8 % under our operator-aware anonymizer, while action recognition lost only about nine percentage points.

## Running it

The pipeline has two stages: (1) bake the CSVs into pickle files the classifier consumes, then (2) train + compare.

```bash
# 1. Convert G1 CSVs to pickle (run once per variant)
python Action_Recognition/g1_to_action_pkl.py --variant original
python Action_Recognition/g1_to_action_pkl.py --variant sanitized

# 2. Train both classifiers and print the comparison table
python Action_Recognition/eval_action_classifier.py
```

For a quicker first pass (≈1 hour on a single GPU), use a balanced 30 k-clip subset:

```bash
python Action_Recognition/g1_to_action_pkl.py --variant original  --fast
python Action_Recognition/g1_to_action_pkl.py --variant sanitized --fast
python Action_Recognition/eval_action_classifier.py --fast
```

For the full-dataset run reported in the paper:

```bash
python Action_Recognition/eval_action_classifier.py --epochs 150 --batch-size 32
```

To evaluate more than one anonymized variant in the same run:

```bash
python Action_Recognition/eval_action_classifier.py --sanitized-variants pmr_contrast pmr_random grl
```

The eval script trains independent classifiers per variant, prints per-class top-1/top-5 tables, and writes a JSON results file under `Action_Recognition/data/results/`. The headline is the **delta-top-1** row: a small (positive or negative) delta means the anonymized data is still useful; a large negative delta means anonymization destroyed something the classifier needed.

## What's in this folder

| File | Purpose |
|---|---|
| `g1_to_action_pkl.py` | Reads the split manifests, walks every referenced G1 CSV, computes [position, velocity, acceleration] features, writes per-split `.pkl` files plus a `label_map.json`. |
| `eval_action_classifier.py` | Trains the classifier on original and on each anonymized variant; prints overall + per-class accuracy and a summary delta table. |

The classifier backbone is the graph-conv module that ships under `code/src/ProtoGCN/`; this folder is just the wrapper that turns it into the action-recognition utility check described above.
