"""
Quick scan: best activity for each (attr, feature) v2 spec the user requested.
"""
import json, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

INVERT = Path(__file__).resolve().parent.parent
BS = Path('C:/Users/sihat/Downloads/bones-seed')

cache = json.loads((INVERT / 'feature-analysis-animation' / 'data' / 'per_clip_features.json').read_text())
meta = pd.read_parquet(BS / 'metadata' / 'seed_metadata_v003.parquet')
meta = meta[~meta['move_g1_mujoco_path'].astype(str).str.endswith('_M.csv')]
meta['csv_rel'] = (meta['move_g1_mujoco_path'].astype(str)
                   .str.replace('\\', '/', regex=False)
                   .str.replace(r'^dataset/', '', regex=True))
meta = meta[meta['move_duration_frames'].fillna(0) >= 200].copy()

rows = []
for r in meta.itertuples():
    per = cache.get(r.csv_rel)
    if not per: continue
    rows.append({
        'task': r.content_type_of_movement, 'actor_uid': r.actor_uid,
        'actor_age_yr':    getattr(r, 'actor_age_yr',    None),
        'actor_weight_kg': getattr(r, 'actor_weight_kg', None),
        'actor_height_cm': getattr(r, 'actor_height_cm', None),
        'actor_gender':    getattr(r, 'actor_gender',    None), 'per': per,
    })
df = pd.DataFrame(rows)
print(f'{len(df):,} clips × {df["task"].nunique()} activities')

ATTR_COL = {'age':'actor_age_yr','weight':'actor_weight_kg','height':'actor_height_cm','gender':'actor_gender'}
DECILE = 0.10
MIN_PER_SIDE = 50

def decile(d, attr):
    if attr == 'gender':
        f = set(d[d['actor_gender'].astype(str) == 'F']['actor_uid'].unique())
        m = set(d[d['actor_gender'].astype(str) == 'M']['actor_uid'].unique())
        return f, m
    a = d.groupby('actor_uid')[ATTR_COL[attr]].first().dropna()
    return (set(a[a <= a.quantile(DECILE)].index),
            set(a[a >= a.quantile(1 - DECILE)].index))

def score(F_lo, F_hi):
    if min(len(F_lo), len(F_hi)) < MIN_PER_SIDE:
        return None
    Q1_lo, Q3_lo = np.percentile(F_lo, [25, 75])
    Q1_hi, Q3_hi = np.percentile(F_hi, [25, 75])
    if np.median(F_hi) > np.median(F_lo):
        h, l = Q1_hi, Q3_lo
    else:
        h, l = Q1_lo, Q3_hi
    iqr = max(np.percentile(np.r_[F_lo, F_hi], 75) - np.percentile(np.r_[F_lo, F_hi], 25), 1e-9)
    return {'n': min(len(F_lo), len(F_hi)),
            'gap': float((h - l) / iqr),
            'med_lo': float(np.median(F_lo)),
            'med_hi': float(np.median(F_hi))}

V2 = [
    ('age',    'hip_peak_vel'),
    ('weight', 'waist_rom'),
    ('height', 'root_translate_rom'),
    ('gender', 'wrist_rom'),
]

for attr, feat in V2:
    sub = df.copy()
    sub['F'] = sub['per'].apply(lambda d: d.get(feat))
    sub = sub.dropna(subset=['F'])
    lo_a, hi_a = decile(sub, attr)
    if not lo_a or not hi_a:
        print(f'\n!! {attr}|{feat}: no decile')
        continue
    rs = []
    for task, grp in sub.groupby('task'):
        F_lo = grp[grp['actor_uid'].isin(lo_a)]['F'].values
        F_hi = grp[grp['actor_uid'].isin(hi_a)]['F'].values
        s = score(F_lo, F_hi)
        if s is None: continue
        if attr == 'gender':
            aF = grp.groupby('actor_uid')['F'].mean()
            ax = grp.groupby('actor_uid')['actor_gender'].first().map(
                lambda g: 0.0 if str(g) == 'F' else 1.0)
        else:
            aF = grp.groupby('actor_uid')['F'].mean()
            ax = grp.groupby('actor_uid')[ATTR_COL[attr]].first()
        ax = ax.dropna(); aF = aF.reindex(ax.index).dropna(); ax = ax.reindex(aF.index)
        if len(ax) >= 3:
            r = sp_stats.pearsonr(ax.values, aF.values)[0]
        else:
            r = float('nan')
        s['task'] = task; s['r'] = float(r)
        rs.append(s)
    rs.sort(key=lambda x: (-x['gap'], -abs(x['r'])))
    print(f"\n--- {attr} | {feat} (top 6) ---")
    for i, r in enumerate(rs[:6]):
        mk = '*' if r['gap'] > 0 else ' '
        print(f"  {i+1}.{mk} {r['task']:24s}  r={r['r']:+.3f}  gap/IQR={r['gap']:+.3f}  "
              f"n={r['n']:4d}  med_lo={r['med_lo']:.1f}  med_hi={r['med_hi']:.1f}")
