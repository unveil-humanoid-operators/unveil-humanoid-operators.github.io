"""
find_best_activities.py
-----------------------
For each (attribute, feature) pair planned for the paper boxplots, scan every
activity in the dataset and rank by "how cleanly the two decile boxes separate
when we plot ALL clips of that activity".

Output a CSV + console table. The top-scoring activity per (attr, feat) is the
one we lock into make_boxplots.py.

Ranking score combines:
  * |r|                       — strength of the underlying correlation
  * gap                       — (Q1 of higher-F side) − (Q3 of lower-F side)
                                 divided by the overall IQR. Positive => boxes
                                 are visually separated.
  * sample size               — n clips per side (we want >= 100 each)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


ROOT = Path(__file__).resolve().parent
INVERT = ROOT.parent
BS = Path('C:/Users/sihat/Downloads/bones-seed')

CACHE = INVERT / 'feature-analysis-animation' / 'data' / 'per_clip_features.json'
META  = BS / 'metadata' / 'seed_metadata_v003.parquet'

# (attribute_short, feature_key)
PANELS = [
    ('age',    'waist_rom'),
    ('weight', 'root_translate_peak_vel'),   # user-requested swap from ankle_peak_vel
    ('height', 'root_translate_rom'),
    ('gender', 'ankle_rom'),
]

ATTR_COL = {
    'age':    'actor_age_yr',
    'weight': 'actor_weight_kg',
    'height': 'actor_height_cm',
    'gender': 'actor_gender',
}

DECILE = 0.10
MIN_PER_SIDE = 20          # exclude activities with too few clips per decile


def load():
    meta = pd.read_parquet(META)
    meta = meta[~meta['move_g1_mujoco_path'].astype(str).str.endswith('_M.csv')]
    meta['csv_rel'] = (meta['move_g1_mujoco_path'].astype(str)
                       .str.replace('\\', '/', regex=False)
                       .str.replace(r'^dataset/', '', regex=True))
    meta = meta[meta['move_duration_frames'].fillna(0) >= 200].copy()
    cache = json.loads(CACHE.read_text())
    rows = []
    for r in meta.itertuples():
        per = cache.get(r.csv_rel)
        if not per: continue
        rows.append({
            'task':            r.content_type_of_movement,
            'actor_uid':       r.actor_uid,
            'actor_age_yr':    getattr(r, 'actor_age_yr',    None),
            'actor_weight_kg': getattr(r, 'actor_weight_kg', None),
            'actor_height_cm': getattr(r, 'actor_height_cm', None),
            'actor_gender':    getattr(r, 'actor_gender',    None),
            'per':             per,
        })
    return pd.DataFrame(rows)


def decile_actors(df, attr):
    if attr == 'gender':
        f = set(df[df['actor_gender'].astype(str) == 'F']['actor_uid'].unique())
        m = set(df[df['actor_gender'].astype(str) == 'M']['actor_uid'].unique())
        return f, m
    col = ATTR_COL[attr]
    a = df.groupby('actor_uid')[col].first().dropna()
    lo_thr, hi_thr = a.quantile(DECILE), a.quantile(1 - DECILE)
    return set(a[a <= lo_thr].index), set(a[a >= hi_thr].index)


def score_activity(F_lo, F_hi):
    """Higher is better. Combines |r|-like signal magnitude, box-gap cleanness,
    and a clip-count multiplier (caps once both sides exceed ~200 clips)."""
    n = min(len(F_lo), len(F_hi))
    if n < MIN_PER_SIDE:
        return None
    Q1_lo, Q3_lo = np.percentile(F_lo, [25, 75])
    Q1_hi, Q3_hi = np.percentile(F_hi, [25, 75])
    if np.median(F_hi) > np.median(F_lo):
        higher, lower_Q3 = Q1_hi, Q3_lo
    else:
        higher, lower_Q3 = Q1_lo, Q3_hi
    overall_iqr = max(np.percentile(np.r_[F_lo, F_hi], 75)
                      - np.percentile(np.r_[F_lo, F_hi], 25), 1e-9)
    gap = (higher - lower_Q3) / overall_iqr      # >0 means boxes don't overlap
    # Cohen's d between sides — a clean directional effect-size signal
    pooled_std = np.sqrt(0.5 * (F_lo.var(ddof=1) + F_hi.var(ddof=1)))
    if pooled_std > 0:
        cohens_d = abs(F_lo.mean() - F_hi.mean()) / pooled_std
    else:
        cohens_d = 0.0
    sample_factor = min(1.0, n / 200)
    score = (max(0.0, gap) + 0.1) * (cohens_d + 0.1) * sample_factor
    return {
        'n_per_side':  n,
        'gap_iqr':     float(gap),
        'cohens_d':    float(cohens_d),
        'med_lo':      float(np.median(F_lo)),
        'med_hi':      float(np.median(F_hi)),
        'score':       float(score),
    }


def main():
    df = load()
    print(f'{len(df):,} clips × {df["task"].nunique()} activities')
    out_rows = []
    for attr, feat in PANELS:
        print(f'\n=== {attr.upper():7s} | {feat} ===')
        sub = df.copy()
        sub['F'] = sub['per'].apply(lambda d: d.get(feat))
        sub = sub.dropna(subset=['F'])
        lo_actors, hi_actors = decile_actors(sub, attr)
        if not lo_actors or not hi_actors:
            print('   (no decile split)'); continue
        rows = []
        for task, grp in sub.groupby('task'):
            F_lo = grp[grp['actor_uid'].isin(lo_actors)]['F'].values
            F_hi = grp[grp['actor_uid'].isin(hi_actors)]['F'].values
            res = score_activity(F_lo, F_hi)
            if res is None: continue
            # Per-actor mean r (separate from box-level metrics)
            if attr == 'gender':
                actor_F = grp.groupby('actor_uid')['F'].mean()
                actor_a = grp.groupby('actor_uid')['actor_gender'].first().map(
                    lambda g: 0.0 if str(g) == 'F' else 1.0)
            else:
                actor_F = grp.groupby('actor_uid')['F'].mean()
                actor_a = grp.groupby('actor_uid')[ATTR_COL[attr]].first()
            x = actor_a.dropna(); y = actor_F.reindex(x.index).dropna()
            x = x.reindex(y.index)
            if len(x) >= 3:
                r_val, _ = sp_stats.pearsonr(x.values, y.values)
            else:
                r_val = float('nan')
            res['task'] = task
            res['r']    = float(r_val)
            rows.append(res)
        # Sort by gap/IQR first (visual separation) then by Cohen's d
        # then by sample size — user explicitly wants clear box separation.
        rows.sort(key=lambda x: (-x['gap_iqr'], -x['cohens_d'], -x['n_per_side']))
        for i, rec in enumerate(rows[:10]):
            print(f'   {i+1}. {rec["task"]:28s}  r={rec["r"]:+.3f}  '
                  f'gap/IQR={rec["gap_iqr"]:+.3f}  d={rec["cohens_d"]:.2f}  '
                  f'n={rec["n_per_side"]:4d}  med_lo={rec["med_lo"]:.1f}  med_hi={rec["med_hi"]:.1f}')
        for rec in rows:
            out_rows.append({'attr': attr, 'feature': feat, **rec})
    pd.DataFrame(out_rows).to_csv(ROOT / 'activity_search_results.csv', index=False)
    print(f'\n[out] activity_search_results.csv')


if __name__ == '__main__':
    main()
