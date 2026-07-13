"""
build_pool_matrix_config.py
---------------------------
Adapter: single_activity_pool.json -> feature_matrix_v1 config that
generate_smpl_matrix.py can consume.

Each instance from the pool becomes one row. We:
  1. Join with the metadata parquet to recover source_bvh + actor biometrics
     (the slim pool only stores g1_csv + actor_uid + actor_gender).
  2. Copy each instance's G1 CSV into data/feature_matrix_instances/<rid>.csv
     so instance_view.html can fetch it via a same-origin relative URL.
  3. MERGE the new rows with the existing feature_matrix_config_instances.json
     so the popup viewer keeps working for both old and new instance_ids.

Output:
  data/feature_matrix_config_pool.json          (pool rows only, for the fitter)
  data/feature_matrix_config_instances.json     (merged: old + new, for popups)
  data/feature_matrix_instances/<rid>.csv       (copied G1 CSVs)
  pool_chunk_{0,1,2}.txt                        (3 even chunks of row_ids)
"""

import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent
BS   = Path('C:/Users/sihat/Downloads/bones-seed')
GENDER_MAP = {'M': 'male', 'F': 'female'}


def main():
    pool = json.loads((ROOT / 'data' / 'single_activity_pool.json').read_text())
    print(f'[load] single_activity_pool.json: {len(pool["panels"])} panels')

    meta = pd.read_parquet(BS / 'metadata' / 'seed_metadata_v003.parquet')
    meta['csv_rel'] = (meta['move_g1_mujoco_path'].astype(str)
                       .str.replace('\\', '/', regex=False)
                       .str.replace(r'^dataset/', '', regex=True))
    # Keep the 'dataset/' prefix on BVH paths — the actual files live under
    # bones-seed/dataset/soma_uniform/bvh/..., not bones-seed/soma_uniform/bvh.
    meta['bvh_rel'] = meta['move_soma_uniform_path'].astype(str).str.replace('\\', '/', regex=False)
    meta = meta[['csv_rel', 'bvh_rel', 'actor_age_yr', 'actor_height_cm',
                 'actor_weight_kg', 'actor_gender', 'move_duration_frames']]
    by_csv = meta.set_index('csv_rel').to_dict('index')
    print(f'[load] metadata indexed: {len(by_csv):,} csv-keyed rows')

    # Copy G1 CSVs into the local mirror dir so instance_view can serve them
    # over the same origin without going back to bones-seed.
    csv_mirror = ROOT / 'data' / 'feature_matrix_instances'
    csv_mirror.mkdir(parents=True, exist_ok=True)

    rows = []
    missing_bvh = 0
    missing_meta = 0
    missing_csv = 0
    copied = 0
    for key, panel in pool['panels'].items():
        for inst in panel['instances']:
            csv_rel = inst['g1_csv']
            m = by_csv.get(csv_rel)
            if not m:
                missing_meta += 1
                continue
            bvh_rel = m.get('bvh_rel')
            if not bvh_rel or bvh_rel in (None, 'nan', 'None'):
                missing_bvh += 1
                continue
            # Copy the G1 CSV into the local mirror.
            src_csv = BS / 'dataset' / csv_rel       # dataset/g1/csv/...
            if not src_csv.exists():
                # Try without the dataset prefix in case metadata changes.
                src_csv = BS / csv_rel
            if not src_csv.exists():
                missing_csv += 1
                continue
            rid = inst['instance_id']
            dst_csv = csv_mirror / f'{rid}.csv'
            if not dst_csv.exists():
                shutil.copyfile(src_csv, dst_csv); copied += 1
            shape = {
                'gender':    GENDER_MAP.get(str(m.get('actor_gender', 'M')), 'male'),
                'height_cm': int(round(m.get('actor_height_cm') or 170)),
                'weight_kg': int(round(m.get('actor_weight_kg') or 70)),
            }
            # Build a biometric label so the popup reads e.g.
            # "25 yr old human operator", "45 kg human operator". The
            # `instance_view.html` template appends "old human operator" for
            # the age panel and "human operator" for the others.
            a = inst['attribute_short']
            if a == 'age':
                label = f'{int(round(m.get("actor_age_yr") or 0))} yr'
            elif a == 'weight':
                label = f'{shape["weight_kg"]} kg'
            elif a == 'height':
                label = f'{shape["height_cm"]} cm'
            elif a == 'gender':
                label = shape['gender']
            else:
                label = ''
            rows.append({
                'row_id':          rid,
                'attribute_short': inst['attribute_short'],
                'task':            inst['task'],
                'feature':         inst['feature'],
                'side':            inst['side'],
                'r':               inst['r'],
                'predicted_sign':  inst['predicted_sign'],
                # The fitter writes <rid>_high.bin; we always populate the
                # "high" block (and leave "low" empty so it skips).
                'high': {
                    'label':      label,
                    'actor_uid':  inst['actor_uid'],
                    'source_bvh': bvh_rel,
                    'g1_csv':     f'./data/feature_matrix_instances/{rid}.csv',
                    'shape':      shape,
                    'per_clip_F': inst['per_clip_F'],
                    'n_frames':   int(m.get('move_duration_frames') or 600),
                },
                'low': {},
            })
    print(f'  -> {len(rows)} rows ready; skipped {missing_meta} no-meta, {missing_bvh} no-bvh, {missing_csv} no-csv')
    print(f'  -> copied {copied} G1 CSVs into {csv_mirror}')

    cfg = {
        'version':  1,
        'schema':   'feature_matrix_v1',
        'chrome':   'minimal',
        'smpl_dir': './data/smpl_feature_matrix_instances',  # same dir as before
        'rows':     rows,
    }
    out = ROOT / 'data' / 'feature_matrix_config_pool.json'
    out.write_text(json.dumps(cfg, indent=2))
    print(f'[out] wrote {out}  ({len(rows)} rows)')

    # Merged config so instance_view.html keeps resolving both old and new
    # row_ids out of one file. The new pool rows OVERWRITE any existing row
    # with the same id (so label / shape edits in this script land cleanly).
    inst_cfg_path = ROOT / 'data' / 'feature_matrix_config_instances.json'
    if inst_cfg_path.exists():
        existing = json.loads(inst_cfg_path.read_text())
        old_rows = existing.get('rows', [])
    else:
        existing, old_rows = cfg, []
    new_ids = {r['row_id'] for r in rows}
    kept_old = [r for r in old_rows if r['row_id'] not in new_ids]
    merged = kept_old + rows
    merged_cfg = dict(existing)
    merged_cfg['rows'] = merged
    inst_cfg_path.write_text(json.dumps(merged_cfg, indent=2))
    print(f'[out] merged {inst_cfg_path}  ({len(merged)} rows total: '
          f'{len(kept_old)} preserved old + {len(rows)} new/updated pool)')

    # Also emit clip-ids file split into N chunks for parallel workers
    n_workers = 3
    chunks = [[] for _ in range(n_workers)]
    for i, r in enumerate(rows):
        chunks[i % n_workers].append(r['row_id'])
    for i, chunk in enumerate(chunks):
        p = ROOT / f'pool_chunk_{i}.txt'
        p.write_text('\n'.join(chunk))
        print(f'[out] {p.name} -> {len(chunk)} ids')


if __name__ == '__main__':
    main()
