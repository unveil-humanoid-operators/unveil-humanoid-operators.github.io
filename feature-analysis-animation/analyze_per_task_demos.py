"""
analyze_per_task_demos.py
-------------------------
For each (attribute, default-feature) used by the bar plot, scan every task
and report:
  - |r| over all actors with >= 2 clips
  - actor counts in the bottom-20% and top-20% deciles by attribute
  - clip counts (each actor has >= 2 clips, so usually 2x actor count)
  - per-clip F values per side: min/max/mean
  - the "gap": (lo-side max F) vs (hi-side min F) — story-relevant separation
  - direction-checked verdict: does the panel's claim land on this task?

Pick: top 5 tasks per (attribute, feature) ranked by a story-quality score
that combines (a) enough demos per side, (b) clean F separation in the
predicted direction.

Output: feature-analysis-animation/data/per_task_demo_analysis.json + .csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


ROOT = Path(__file__).parent
BS   = Path('C:/Users/sihat/Downloads/bones-seed')

# Match the bar-plot's defaults — these are the (attr, feature) pairs that
# actually drive the visible visualization.
ATTR_FEATURE = {
    'actor_age_yr':    'waist_rom',
    'actor_weight_kg': 'ankle_peak_vel',
    'actor_height_cm': 'root_translate_rom',
    'gender_numeric':  'ankle_rom',
}

DECILE      = 0.20    # bottom/top 20% by attribute
MIN_PER_SIDE = 10     # need at least this many clips per side to plot
MIN_ACTORS  = 50      # task should have enough actors to score correlation


def load_atf():
    atf = pd.read_csv(BS / 'Correlation_V2' / 'simple_interpretability_results' / 'actor_task_features.csv')
    return atf


def load_per_clip_cache():
    return json.loads((ROOT / 'data' / 'per_clip_features.json').read_text())


def main():
    print('[load] actor_task_features.csv')
    atf = load_atf()
    print(f'  -> {len(atf):,} actor-task rows, {atf["actor_uid"].nunique()} actors')

    print('[load] per_clip_features cache (~50 MB)')
    cache = load_per_clip_cache()
    # cache key = forward-slash CSV path under bones-seed; value = feature dict
    print(f'  -> {len(cache):,} cached per-clip feature blobs')

    print('[load] metadata parquet (for csv->actor map)')
    meta = pd.read_parquet(BS / 'metadata' / 'seed_metadata_v003.parquet')
    meta_cand = meta[~meta['move_g1_mujoco_path'].astype(str).str.endswith('_M.csv')].copy()
    meta_cand['csv_rel'] = (meta_cand['move_g1_mujoco_path'].astype(str)
                            .str.replace('\\', '/', regex=False)
                            .str.replace(r'^dataset/', '', regex=True))
    meta_cand['nf'] = pd.to_numeric(meta_cand['move_duration_frames'], errors='coerce')
    meta_cand = meta_cand[meta_cand['nf'].fillna(0) >= 200]
    print(f'  -> {len(meta_cand):,} clips with metadata')

    # Build per-clip F lookup: (task, actor) -> [F values across that actor's clips]
    clip_F_by_actor_task = {}
    for row in meta_cand.itertuples():
        per_clip = cache.get(row.csv_rel)
        if not per_clip:
            continue
        key = (row.content_type_of_movement, row.actor_uid)
        clip_F_by_actor_task.setdefault(key, {}).setdefault('F', [])

    # Now actually populate
    for row in meta_cand.itertuples():
        per_clip = cache.get(row.csv_rel)
        if not per_clip:
            continue
        key = (row.content_type_of_movement, row.actor_uid)
        for feat in {f for f in ATTR_FEATURE.values()}:
            if feat not in per_clip:
                continue
            d = clip_F_by_actor_task.setdefault(key, {})
            d.setdefault(feat, []).append(float(per_clip[feat]))
    print(f'  -> populated {len(clip_F_by_actor_task):,} (task, actor) buckets')

    summary_rows = []
    by_attr_feat = {}

    for attr, feat in ATTR_FEATURE.items():
        attr_short = attr.replace('actor_', '').replace('_yr','').replace('_kg','').replace('_cm','').replace('_numeric','')
        print(f'\n=== {attr_short.upper()} | {feat} ===')

        # Actor-level attribute values
        if attr == 'gender_numeric':
            actor_attr = meta_cand.groupby('actor_uid')['actor_gender'].first().map(
                lambda g: 0.0 if str(g) == 'F' else 1.0)
        else:
            actor_attr = meta_cand.groupby('actor_uid')[attr].first()
        actor_attr = actor_attr.dropna()

        # All tasks
        all_tasks = sorted(set(k[0] for k in clip_F_by_actor_task.keys()))
        task_rows = []
        for task in all_tasks:
            # Gather (actor, F_mean, n_clips) for this task+feature
            recs = []
            for (t, actor), d in clip_F_by_actor_task.items():
                if t != task or feat not in d:
                    continue
                if actor not in actor_attr.index:
                    continue
                fs = d[feat]
                if len(fs) < 2:
                    continue
                recs.append({'actor': actor, 'attr': actor_attr[actor],
                             'Fs': fs, 'F_mean': float(np.mean(fs))})
            if len(recs) < MIN_ACTORS:
                continue

            # Pearson r over actor means
            x = np.array([r['attr']  for r in recs])
            y = np.array([r['F_mean'] for r in recs])
            r_val, p_val = sp_stats.pearsonr(x, y)
            sign_r = 1 if r_val > 0 else -1

            # Decile split
            lo_thr = np.quantile(x, DECILE)
            hi_thr = np.quantile(x, 1 - DECILE)
            lo_recs = [r for r in recs if r['attr'] <= lo_thr]
            hi_recs = [r for r in recs if r['attr'] >= hi_thr]

            # Each side's clip-level F-values (one per clip, not per actor)
            lo_clips = [f for r in lo_recs for f in r['Fs']]
            hi_clips = [f for r in hi_recs for f in r['Fs']]

            # Direction-checked separation:
            # sign>0 → low attr ↔ low F, high attr ↔ high F
            # sign<0 → low attr ↔ high F, high attr ↔ low F
            if sign_r > 0:
                # We want hi-side bar ABOVE lo-side bar
                low_bar_F  = lo_clips     # should land low on y
                high_bar_F = hi_clips     # should land high on y
            else:
                low_bar_F  = hi_clips     # high-attr actors → SMALL F → bottom bar
                high_bar_F = lo_clips     # low-attr actors  → LARGE F → top bar

            if not low_bar_F or not high_bar_F:
                continue

            low_bar_F  = sorted(low_bar_F)
            high_bar_F = sorted(high_bar_F, reverse=True)

            # Score: enough clips per side AND clean separation
            # "clean": top-bar's bottom > bottom-bar's top, with cushion
            top10_hi = high_bar_F[:10]    # 10 most-extreme high
            top10_lo = low_bar_F[:10]     # 10 smallest-F low
            min_top10_hi  = min(top10_hi) if top10_hi else 0
            max_top10_lo  = max(top10_lo) if top10_lo else 0
            cushion_pct   = (min_top10_hi - max_top10_lo) / (min_top10_hi or 1) * 100

            task_rows.append({
                'task':       task,
                'r':          float(r_val),
                'p':          float(p_val),
                'abs_r':      abs(float(r_val)),
                'n_actors':   len(recs),
                'n_lo_actors': len(lo_recs),
                'n_hi_actors': len(hi_recs),
                'n_lo_clips':  len(low_bar_F),
                'n_hi_clips':  len(high_bar_F),
                'low_bar_F_range':  f'{low_bar_F[0]:.1f}..{low_bar_F[-1]:.1f}',
                'high_bar_F_range': f'{high_bar_F[-1]:.1f}..{high_bar_F[0]:.1f}',
                'top10_hi_min':  float(min_top10_hi),
                'top10_lo_max':  float(max_top10_lo),
                'cushion_pct':   float(cushion_pct),
                'enough_demos':  len(low_bar_F) >= MIN_PER_SIDE and len(high_bar_F) >= MIN_PER_SIDE,
                'separated':     min_top10_hi > max_top10_lo,
            })

        # Story-quality score: |r| * (enough_demos) * (separated) * (cushion / 100, clipped)
        for tr in task_rows:
            score = tr['abs_r']
            if not tr['enough_demos']: score *= 0.3
            if not tr['separated']:    score *= 0.0
            score *= min(1.0, max(0.0, tr['cushion_pct'] / 50))   # cushion 50% = full credit
            tr['score'] = score

        task_rows.sort(key=lambda x: -x['score'])
        kept_for_attr = []
        print(f'  top 5 tasks for {attr_short}|{feat}:')
        for tr in task_rows[:5]:
            ok = 'OK' if (tr['enough_demos'] and tr['separated']) else 'NO'
            print(f'    {ok} {tr["task"]:30s} r={tr["r"]:+.3f} '
                  f'lo_clips={tr["n_lo_clips"]:3d} hi_clips={tr["n_hi_clips"]:3d} '
                  f'cushion={tr["cushion_pct"]:+.1f}%  score={tr["score"]:.3f}')
            kept_for_attr.append(tr)
        by_attr_feat[f'{attr_short}|{feat}'] = task_rows
        for tr in task_rows[:8]:
            summary_rows.append({'attr': attr_short, 'feature': feat, **tr})

    # Save
    out_json = ROOT / 'data' / 'per_task_demo_analysis.json'
    out_json.write_text(json.dumps(by_attr_feat, indent=2))
    print(f'\n[out] wrote {out_json}')

    out_csv = ROOT / 'data' / 'per_task_demo_analysis.csv'
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f'[out] wrote {out_csv}')


if __name__ == '__main__':
    main()
