"""
build_single_activity_pool.py
-----------------------------
For each (attribute, feature) pair the bar plot supports, choose ONE activity
and emit every eligible (actor, clip) record from it. The JS will then randomly
draw 10 per side from this single-activity pool, rather than mixing demos
across many activities as before.

Pick rule: highest combined score of (a) |r| of the feature vs attribute,
(b) clip count per side, (c) F-separation cushion in the predicted direction.
Activities listed in the visualization blocklist are excluded.

Output:
  feature-analysis-animation/data/single_activity_pool.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


ROOT = Path(__file__).parent
BS   = Path('C:/Users/sihat/Downloads/bones-seed')

# All (attr, feature) pairs the bar plot can show. Mirrors
# ATTR_FEATURE_OPTIONS in correlation-bar-plot.html.
ATTR_FEATURES = {
    'actor_age_yr':    ['waist_rom',          'waist_mean_vel',          'ankle_rom'],
    'actor_weight_kg': ['ankle_peak_vel',     'ankle_rom',               'root_translate_peak_vel'],
    'actor_height_cm': ['root_translate_rom', 'root_translate_peak_vel', 'ankle_rom'],
    'gender_numeric':  ['ankle_rom',          'shoulder_rom',            'shoulder_mean_vel'],
}
ATTR_SHORT = {
    'actor_age_yr': 'age', 'actor_weight_kg': 'weight',
    'actor_height_cm': 'height', 'gender_numeric': 'gender',
}
ATTR_UNIT  = {'actor_age_yr':'yr','actor_weight_kg':'kg','actor_height_cm':'cm','gender_numeric':''}

DECILE       = 0.10          # tighter than before — clearer Younger/Older labels
MIN_PER_SIDE = 20            # need at least this many in each decile to qualify
MIN_ACTORS   = 50


def slugify(s):
    return ''.join(c if c.isalnum() else '_' for c in str(s).strip().lower()).strip('_')


def feature_unit(feat):
    if feat.startswith('root_translate'):
        return 'cm/s' if feat.endswith('vel') else 'cm'
    return 'deg/s' if feat.endswith('vel') else 'deg'


def feature_recipe(feat):
    # Minimal recipe shape used by instance_view (we only need .name).
    return { 'name': feat, 'kind': 'joint', 'unit': feature_unit(feat) }


def load_blocklist():
    p = Path('C:/Users/sihat/Downloads/invert/visualization_blocklist.json')
    if not p.exists():
        return {'blocked_tasks': [], 'blocked_per_attr_feature': {}, 'blocked_instance_ids': []}
    b = json.loads(p.read_text())
    return {
        'blocked_tasks': b.get('blocked_tasks') or [],
        'blocked_per_attr_feature': b.get('blocked_per_attr_feature') or {},
        'blocked_instance_ids': b.get('blocked_instance_ids') or [],
    }


def main():
    print('[load] inputs')
    cache = json.loads((ROOT / 'data' / 'per_clip_features.json').read_text())
    meta = pd.read_parquet(BS / 'metadata' / 'seed_metadata_v003.parquet')
    meta = meta[~meta['move_g1_mujoco_path'].astype(str).str.endswith('_M.csv')].copy()
    meta['csv_rel'] = (meta['move_g1_mujoco_path'].astype(str)
                       .str.replace('\\', '/', regex=False)
                       .str.replace(r'^dataset/', '', regex=True))
    meta['nf'] = pd.to_numeric(meta['move_duration_frames'], errors='coerce')
    meta = meta[meta['nf'].fillna(0) >= 200]
    print(f'  -> {len(meta):,} usable clips')
    blocklist = load_blocklist()
    print(f'  -> blocked tasks: {blocklist["blocked_tasks"]}')

    # Attach per-clip F + attributes to each metadata row
    rows = []
    for r in meta.itertuples():
        per = cache.get(r.csv_rel)
        if not per:
            continue
        rows.append({
            'csv_rel':     r.csv_rel,
            'task':        r.content_type_of_movement,
            'actor_uid':   r.actor_uid,
            'actor_age_yr':    getattr(r, 'actor_age_yr', None),
            'actor_weight_kg': getattr(r, 'actor_weight_kg', None),
            'actor_height_cm': getattr(r, 'actor_height_cm', None),
            'actor_gender':    getattr(r, 'actor_gender', None),
            'nf':          int(r.nf),
            'per':         per,
        })
    print(f'  -> {len(rows):,} clips with per-clip F cached')
    df_clips = pd.DataFrame(rows)

    out = {
        'schema': 'single_activity_pool_v1',
        'decile': DECILE,
        'panels': {},
    }
    summary = []
    for attr, feats in ATTR_FEATURES.items():
        attr_short = ATTR_SHORT[attr]
        # Actor-level attribute values
        if attr == 'gender_numeric':
            actor_attr = df_clips.groupby('actor_uid')['actor_gender'].first().map(
                lambda g: 0.0 if str(g) == 'F' else 1.0).dropna()
        else:
            actor_attr = df_clips.groupby('actor_uid')[attr].first().dropna()
        for feat in feats:
            print(f'\n=== {attr_short.upper()} | {feat} ===')
            df_clips_f = df_clips[df_clips['per'].apply(lambda d: feat in d)].copy()
            df_clips_f['F'] = df_clips_f['per'].apply(lambda d: float(d[feat]))

            best = None
            all_task_scores = []
            for task, sub in df_clips_f.groupby('task'):
                if task in blocklist['blocked_tasks']:
                    continue
                tl = blocklist['blocked_per_attr_feature'].get(f'{attr_short}|{feat}', [])
                if task in tl:
                    continue
                # Per-actor mean F (need >= 2 clips per actor)
                grp = sub.groupby('actor_uid')['F'].agg(['mean','count']).rename(columns={'mean':'F_mean','count':'n'})
                grp = grp[grp['n'] >= 2]
                grp = grp.join(actor_attr.rename('attr'), how='inner').dropna()
                if len(grp) < MIN_ACTORS:
                    continue
                r_val, p_val = sp_stats.pearsonr(grp['attr'].values, grp['F_mean'].values)
                sign_r = 1 if r_val > 0 else -1

                lo_thr = grp['attr'].quantile(DECILE)
                hi_thr = grp['attr'].quantile(1 - DECILE)
                lo_actors = set(grp[grp['attr'] <= lo_thr].index)
                hi_actors = set(grp[grp['attr'] >= hi_thr].index)
                lo_clips = sub[sub['actor_uid'].isin(lo_actors)]
                hi_clips = sub[sub['actor_uid'].isin(hi_actors)]
                if len(lo_clips) < MIN_PER_SIDE or len(hi_clips) < MIN_PER_SIDE:
                    continue

                # Direction-checked: which clips end up on top vs bottom bar
                if sign_r > 0:
                    bot_clips, top_clips = lo_clips.sort_values('F'), hi_clips.sort_values('F', ascending=False)
                else:
                    bot_clips, top_clips = hi_clips.sort_values('F'), lo_clips.sort_values('F', ascending=False)
                top10_top_min = top_clips['F'].head(10).min()
                top10_bot_max = bot_clips['F'].head(10).max()
                cushion = (top10_top_min - top10_bot_max) / (top10_top_min or 1)
                if not (top10_top_min > top10_bot_max):
                    continue

                # Score: prioritize DEMO COUNT (user wanted "most-demos set"),
                # then |r|, then cushion. Big-clip-pool wins.
                n_per_side = min(len(lo_clips), len(hi_clips))
                score = (n_per_side) * (1 + abs(r_val)) * (1 + cushion)
                all_task_scores.append({
                    'task': task, 'r': float(r_val), 'n_per_side': int(n_per_side),
                    'cushion': float(cushion), 'score': float(score),
                    'lo_clips': lo_clips, 'hi_clips': hi_clips, 'sign_r': sign_r,
                })
            if not all_task_scores:
                print(f'  ! no qualifying task — skipping')
                continue

            all_task_scores.sort(key=lambda x: -x['score'])
            print('  top 5 candidate tasks:')
            for t in all_task_scores[:5]:
                print(f'    task={t["task"]:25s} r={t["r"]:+.3f} n_per_side={t["n_per_side"]:5d} cushion={t["cushion"]*100:+.1f}%  score={t["score"]:.0f}')
            best = all_task_scores[0]
            print(f'  -> CHOSEN: "{best["task"]}"  (n_per_side={best["n_per_side"]})')

            # Materialize instances: each side's candidate pool is the
            # FRACTION of clips in that decile pointing in the predicted
            # direction by F (FRAC=0.75 → top-75% for high-F bar, bottom-75%
            # for low-F bar). Then sample MAX_PER_SIDE at random from that
            # fraction. The two fractions overlap in the middle 50% of F
            # values, so bars can legitimately overlap in y-range — that
            # reflects real population variance rather than cherry-picked
            # extremes.
            # Fixed seed + exactly DOTS_PER_SIDE picks per side: the bar plot
            # shows the same 20 dots every page load, so we only need to fit
            # SMPL meshes for those 20 (not 200). This is what the JS hooks
            # render — see correlation-bar-plot.html / pickRandom().
            MAX_PER_SIDE = 10
            FRAC = 0.75
            sign_r = best['sign_r']
            rng = np.random.default_rng(42)
            def fraction(df, want_low_F, frac):
                if df.empty: return df
                if want_low_F:
                    thr = df['F'].quantile(frac)
                    return df[df['F'] <= thr]
                else:
                    thr = df['F'].quantile(1 - frac)
                    return df[df['F'] >= thr]
            def sample(df, n):
                if len(df) <= n: return df
                return df.iloc[rng.choice(len(df), size=n, replace=False)]
            if sign_r > 0:
                bot_clips = sample(fraction(best['lo_clips'], True,  FRAC), MAX_PER_SIDE)
                top_clips = sample(fraction(best['hi_clips'], False, FRAC), MAX_PER_SIDE)
                bot_side, top_side = 'low', 'high'
            else:
                bot_clips = sample(fraction(best['hi_clips'], True,  FRAC), MAX_PER_SIDE)
                top_clips = sample(fraction(best['lo_clips'], False, FRAC), MAX_PER_SIDE)
                bot_side, top_side = 'high', 'low'
            instances = []
            for side_label, side_df in [(bot_side, bot_clips), (top_side, top_clips)]:
                for r in side_df.itertuples():
                    actor_uid = r.actor_uid
                    iid = f'{attr_short}_{slugify(feat)}_{slugify(best["task"])}_{actor_uid}_{Path(r.csv_rel).stem}'
                    if attr == 'gender_numeric':
                        attr_value = 0.0 if str(r.actor_gender) == 'F' else 1.0
                    else:
                        attr_value = float(getattr(r, attr, 0.0))
                    # Slim per-instance schema — the bar plot only needs these
                    # fields. Popup loaders can fetch full records on demand
                    # via instance_view + the g1_csv path.
                    instances.append({
                        'instance_id':    iid,
                        'side':           side_label,
                        'task':           best['task'],
                        'feature':        feat,
                        'attribute_short': attr_short,
                        'predicted_sign': sign_r,
                        'r':              round(best['r'], 4),
                        'per_clip_F':     round(float(r.F), 2),
                        'attr_value':     round(attr_value, 1),
                        'actor_uid':      str(actor_uid),
                        'actor_gender':   str(r.actor_gender or ''),
                        'g1_csv':         r.csv_rel,
                    })

            key = f'{attr_short}|{feat}'
            out['panels'][key] = {
                'task':    best['task'],
                'r':       best['r'],
                'sign':    best['sign_r'],
                'n_per_side': best['n_per_side'],
                'instances': instances,
            }
            summary.append({'panel': key, 'task': best['task'],
                            'r': best['r'], 'n_per_side': best['n_per_side'],
                            'n_instances': len(instances)})

    out_path = ROOT / 'data' / 'single_activity_pool.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f'\n[out] wrote {out_path}')
    pd.DataFrame(summary).to_csv(ROOT / 'data' / 'single_activity_pool_summary.csv', index=False)
    print(f'\nSummary:')
    for s in summary:
        print(f'  {s["panel"]:35s} task={s["task"]:20s} r={s["r"]:+.3f} n_per_side={s["n_per_side"]:4d}  ({s["n_instances"]} total instances)')


if __name__ == '__main__':
    main()
