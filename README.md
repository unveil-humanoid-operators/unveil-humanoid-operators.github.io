# Who Moved the Robot? Humanoid Datasets Remember Their Operators

Project page for **UNVEIL** — *Who Moved the Robot? Humanoid Datasets Remember Their Operators*.

Sihat Afnan, Unnat Jain\*, Habiba Farrukh\* · University of California, Irvine

🌐 **Live site:** https://unveil-humanoid-operators.github.io

## Overview

Human-to-humanoid motion retargeting solves frame-by-frame inverse kinematics: it matches
landmark positions on a shared robot skeleton (Unitree G1) and intentionally discards the
operator's body shape. It is widely assumed that this makes the resulting robot trajectories
anonymous.

We show this assumption is false. Retargeting places no objective over how joints move *across
time*, so operator-specific movement dynamics — joint velocity profiles, ranges of motion, and
coordination rhythms shaped by individual physiology — survive the transform. **UNVEIL inverts
retargeting**, recovering an operator's biometric attributes directly from the robot's joint
trajectories.

On **BONES-SEED** (522 operators, 142K G1-retargeted sequences), UNVEIL reaches:

- **96.0%** gender accuracy and **97.2%** Top-1 re-identification on operators seen during training;
- **83.4%** gender accuracy, and age (±4.2 yr), height (±5.7 cm), and weight (±9.1 kg) inference on
  operators *never seen* during training.

To mitigate this leakage, we also propose an **operator-aware anonymizer** that drops
re-identification from 97.2% to 16.8% while sacrificing only ~9 points of action-recognition utility.

## The page

A static single-page site (Bootstrap 5 + vanilla Three.js viewers). It walks through the finding
with synchronized, side-by-side demos: the G1 robot trajectory, the reconstructed operator body,
and the ground-truth body — shown across locomotion, dance, and idle motions so the leakage is
visible across motion types.

## Local development

No build step. Open `index.html` directly, or serve it:

```bash
python -m http.server 8000
```

Then visit `http://localhost:8000`.

## Deployment

Deployed via GitHub Pages from the `main` branch.
