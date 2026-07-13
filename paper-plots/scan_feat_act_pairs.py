"""
scan_feat_act_pairs.py
----------------------
For a given attribute, scan EVERY (feature, activity) pair and rank by
"how cleanly the two decile boxes separate". Used when the bar-plot's
default feature doesn't yield a clean enough gap for the paper figure.
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

DECILE = 0.10
MIN_PER_SIDE = 80
MIN_ABS_R    = 0.25

ATTR_COL = {
    'age':    'actor_age_yr',
    'weight': 'actor_weight_kg',
    'height': 'actor_height_cm',
    'gender': 'actor_gender',
}


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
            'task': r.content_type_of_movement, 'actor_uid': r.actor_uid,
            'actor_age_yr':    getattr(r, 'actor_age_yr',    None),
            'actor_weight_kg': getattr(r, 'actor_weight_kg', None),
            'actor_height_cm': getattr(r, 'actor_height_cm', None),
            'actor_gender':    getattr(r, 'actor_gender',    None),
            'per': per,
        })
    return pd.DataFrame(rows)


def decile_actors(df, attr):
    if attr == 'gender':
        f = set(df[df['actor_gender'].astype(str) == 'F']['actor_uid'].unique())
        m = set(df[df['actor_gender'].astype(str) == 'M']['actor_uid'].unique())
        return f, m
    col = ATTR_COL[attr]
    a = df.groupby('actor_uid')[col].first().dropna()
    return (set(a[a <= a.quantile(DECILE)].index),
            set(a[a >= a.quantile(1 - DECILE)].index))


def score(F_lo, F_hi):
    if min(len(F_lo), len(F_hi)) < MIN_PER_SIDE:
        return None
    Q1_lo, Q3_lo = np.percentile(F_lo, [25, 75])
    Q1_hi, Q3_hi = np.percentile(F_hi, [25, 75])
    if np.median(F_hi) > np.median(F_lo):
        higher_Q1, lower_Q3 = Q1_hi, Q3_lo
    else:
        higher_Q1, lower_Q3 = Q1_lo, Q3_hi
    iqr = max(np.percentile(np.r_[F_lo, F_hi], 75)
              - np.percentile(np.r_[F_lo, F_hi], 25), 1e-9)
    gap = (higher_Q1 - lower_Q3) / iqr
    return {
        'n_per_side': min(len(F_lo), len(F_hi)),
        'gap_iqr':    float(gap),
        'med_lo':     float(np.median(F_lo)),
        'med_hi':     float(np.median(F_hi)),
    }


def scan(df, attr, top_n=12):
    print(f'\n=== {attr.upper()} — scanning every (feature, activity) ===')
    if not df.iloc[0]['per']: return
    all_features = sorted(df.iloc[0]['per'].keys())
    rows = []
    for feat in all_features:
        sub = df.copy()
        sub['F'] = sub['per'].apply(lambda d: d.get(feat))
        sub = sub.dropna(subset=['F'])
        lo_actors, hi_actors = decile_actors(sub, attr)
        for task, grp in sub.groupby('task'):
            F_lo = grp[grp['actor_uid'].isin(lo_actors)]['F'].values
            F_hi = grp[grp['actor_uid'].isin(hi_actors)]['F'].values
            res = score(F_lo, F_hi)
            if res is None: continue
            # Per-actor r
            if attr == 'gender':
                aF = grp.groupby('actor_uid')['F'].mean()
                ax = grp.groupby('actor_uid')['actor_gender'].first().map(
                    lambda g: 0.0 if str(g) == 'F' else 1.0)
            else:
                aF = grp.groupby('actor_uid')['F'].mean()
                ax = grp.groupby('actor_uid')[ATTR_COL[attr]].first()
            ax = ax.dropna(); aF = aF.reindex(ax.index).dropna()
            ax = ax.reindex(aF.index)
            if len(ax) >= 3:
                r_val, _ = sp_stats.pearsonr(ax.values, aF.values)
            else:
                r_val = float('nan')
            if not np.isnan(r_val) and abs(r_val) < MIN_ABS_R:
                continue
            res['task'] = task; res['feature'] = feat; res['r'] = float(r_val)
            rows.append(res)
    rows.sort(key=lambda x: (-x['gap_iqr'], -abs(x['r'])))
    print(f'   top {top_n} (gap/IQR > 0 = boxes do not overlap):')
    for i, rec in enumerate(rows[:top_n]):
        marker = ' *' if rec['gap_iqr'] > 0 else '  '
        print(f'  {i+1:2d}.{marker} {rec["feature"]:26s} | {rec["task"]:22s}  '
              f'r={rec["r"]:+.3f}  gap/IQR={rec["gap_iqr"]:+.3f}  '
              f'n={rec["n_per_side"]:4d}  med_lo={rec["med_lo"]:.1f}  med_hi={rec["med_hi"]:.1f}')


def main():
    df = load()
    print(f'{len(df):,} clips × {df["task"].nunique()} activities × '
          f'{len(df.iloc[0]["per"])} features')
    for attr in ['age', 'height']:
        scan(df, attr)


if __name__ == '__main__':
    main()
