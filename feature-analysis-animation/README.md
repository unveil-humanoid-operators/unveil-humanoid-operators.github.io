# feature-analysis-animation

Interactive figure for the UNVEIL project page: **operator body weight survives retargeting**. Two identical G1 robots play the same activity, one driven by a heavy operator's motion and one by a light operator's; the heavier-operator robot moves visibly slower.

## What's inside

```
feature-analysis-animation/
├── viewer.html              entry — three-layer page (iframe + scatter + bars)
├── twin_g1.html             standalone twin-robot Three.js scene
├── config.json              defaults consumed by viewer.html
├── data/
│   ├── partials.json        per-category partial r + n + sign + pearson
│   ├── actors.json          per-category list of {weight, root_avg_speed, ...}
│   ├── pairs.json           per-category {heavy: {...}, light: {...}}
│   └── trajectories/*.csv   heavy/light G1 motion CSVs
└── generate_data.py         rebuilds everything in data/ from bones-seed
```

The G1 mesh + model files are not duplicated — `twin_g1.html` loads them from `../g1_engine/`.

## Running locally

```powershell
cd C:\Users\sihat\Downloads\invert
python -m http.server 8000
# then open http://localhost:8000/feature-analysis-animation/viewer.html
```

`twin_g1.html` can also be opened directly: `…/feature-analysis-animation/twin_g1.html?category=jumping`.

## Regenerating the data

```powershell
cd C:\Users\sihat\Downloads\invert\feature-analysis-animation
python generate_data.py --bones-seed C:\Users\sihat\Downloads\bones-seed
```

Inputs (read-only):
- `bones-seed/Correlation/cross-joint-task-results-content-type/<cat>/actor_features_v2.csv` — per-actor aggregates by motion content type.
- `bones-seed/Correlation/cross-joint-task-results/<cat>/actor_features_v2.csv` — coarse 8 categories.
- `bones-seed/metadata/seed_metadata_v003.parquet` — used to map actor → G1 CSV file path.
- `bones-seed/g1/csv/<date>/*.csv` — the actual G1 trajectories that get copied into `data/trajectories/`.

The script recomputes the partial correlation `partial_r(actor_weight_kg, root_avg_speed | actor_age_yr, actor_height_cm, actor_gender)` per category, picks a heavy/light operator pair (top/bottom decile by weight, closest to that decile's mean root speed to avoid outliers), copies the chosen G1 CSVs, and writes all three JSON files plus the trajectories.

Thresholds (tunable in the script): fine subcategories `n ≥ 5`, coarse categories `n ≥ 30`.

## Embedding in `index.html`

```html
<iframe src="feature-analysis-animation/viewer.html"
        loading="lazy"
        style="width:100%; height:1500px; border:0; border-radius:12px;"></iframe>
```

The widget is fully self-contained — no parent CSS or JS dependencies. Sized as a single tall iframe; alternatively you can swap the three layers into the parent page by lifting `viewer.html`'s `<section>` blocks and replacing the iframe in Layer 1 with a direct embed of `twin_g1.html`.

## Visualization blocklist

`../visualization_blocklist.json` lists clips and task labels whose SMPL human is visually distorted (broken pose / fitting failure). Anything in that file must **never** be shown in any visualization built from this directory either — `instance_view.html`, the trajectories under `data/`, or any future widget. If you build a new instance picker here, filter its inputs against the blocklist before rendering. Currently blocked: `turning`, `gesture, dancing` (task-level, every panel). See the repo-root `CLAUDE.md` → "Visualization blocklist" for the schema.
