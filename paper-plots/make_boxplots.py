"""
make_boxplots.py
----------------
Publication-grade boxplots of "feature of movement dynamics" distributions
split by an attribute's bottom vs top decile, for each of the four (attribute,
feature, activity) pairs that drive the project page's boxplot panels.

Style choices follow NeurIPS / ICML / ICLR / CVPR conventions:
  * serif typography (Computer Modern via matplotlib's `cm` mathtext font set)
  * 8-10 pt body labels, 7-9 pt ticks
  * vector PDF output sized for a two-column page (~3.4" wide each)
  * Tukey whiskers (1.5 * IQR), median as a solid horizontal line
  * box fill at low alpha + darker edge, sampled from seaborn 'colorblind'
  * strip plot of individual observations overlaid (semi-transparent dots)
  * top + right spines removed (Tufte-style)
  * y-axis label carries units; x-tick labels are semantic
  * r-value + n annotated in the top corner of each axis

Outputs:
  paper-plots/boxplots_all.pdf            -- 1 x 4 grid of all four attributes
  paper-plots/boxplot_<attr>_<feat>.pdf   -- individual panels

Data source: same upstream as the project page — per-clip cached features
plus the BONES-SEED metadata parquet.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats


# ── Paths ────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent
INVERT  = ROOT.parent
BS      = Path('C:/Users/sihat/Downloads/bones-seed')
CACHE   = INVERT / 'feature-analysis-animation' / 'data' / 'per_clip_features.json'
META    = BS / 'metadata' / 'seed_metadata_v003.parquet'

OUT_DIR = ROOT
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Panel spec ────────────────────────────────────────────────────────────
# Matches the four panels rendered by correlation-bar-plot.html. Each entry
# is (attr_short, feature_key, locked_activity, axis_unit, low_label, high_label).
PANELS = [
    # v1: (feature, activity) per panel chosen by scan_feat_act_pairs.py —
    # picked so the LEFT box (low-attribute decile) sits cleanly above the
    # RIGHT box (high-attribute decile) when ALL clips of that activity are
    # shown. gap/IQR > 0 ⇒ the box quartiles don't overlap.
    ('age',    'waist_mean_vel',          'dancing',          'deg/s', 'Younger', 'Older'),   # gap/IQR +0.21
    ('weight', 'root_translate_peak_vel', 'walking, turning', 'cm/s',  'Lighter', 'Heavier'), # gap/IQR +0.34
    ('height', 'root_translate_rom',      'climbing box',     'cm',    'Shorter', 'Taller'),  # gap/IQR +0.07
    ('gender', 'ankle_rom',               'jogging, turning', 'deg',   'Female',  'Male'),    # gap/IQR −0.27 (best available)
]

# v2: alternate feature picks (one per panel) requested for comparison.
# Activities selected by the same gap/IQR search (scan_v2.py output).
PANELS_V2 = [
    ('age',    'hip_peak_vel',            'jogging, turning', 'deg/s', 'Younger', 'Older'),   # gap/IQR -0.26
    ('weight', 'waist_rom',               'jogging',          'deg',   'Lighter', 'Heavier'), # gap/IQR -0.47
    ('height', 'root_translate_rom',      'climbing box',     'cm',    'Shorter', 'Taller'),  # gap/IQR +0.07
    ('gender', 'wrist_rom',               'dancing',          'deg',   'Female',  'Male'),    # gap/IQR -0.58
]

# Attribute -> column name in the metadata parquet (or special token for gender)
ATTR_COL = {
    'age':    'actor_age_yr',
    'weight': 'actor_weight_kg',
    'height': 'actor_height_cm',
    'gender': 'actor_gender',
}

# Pretty y-axis label for each feature
FEATURE_LABEL = {
    'waist_rom':               'Waist range of motion',
    'waist_mean_vel':          'Waist mean velocity',
    'ankle_peak_vel':          'Ankle peak velocity',
    'ankle_rom':               'Ankle range of motion',
    'hip_peak_vel':            'Hip peak velocity',
    'wrist_rom':               'Wrist range of motion',
    'root_translate_rom':      'Root range of motion',
    'root_translate_peak_vel': 'Root peak velocity',
}

DECILE = 0.10            # bottom-10% vs top-10% by attribute
MIN_CLIPS_PER_ACTOR = 2


# ── Matplotlib / seaborn style ────────────────────────────────────────────
def configure_style():
    sns.set_theme(style='ticks', context='paper')
    mpl.rcParams.update({
        'font.family':       'serif',
        'font.serif':        ['Times New Roman', 'Times', 'DejaVu Serif', 'CMU Serif'],
        'mathtext.fontset':  'cm',
        'axes.titlesize':    16,
        'axes.labelsize':    16,    # y-axis label
        'xtick.labelsize':   14,    # 'Younger' / 'Older' tick labels
        'ytick.labelsize':   12,
        'legend.fontsize':   12,
        'axes.linewidth':    1.0,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size':  3.5,
        'ytick.major.size':  3.5,
        'pdf.fonttype':      42,    # embed TrueType so reviewers' previews keep glyphs
        'ps.fonttype':       42,
        'savefig.bbox':      'tight',
        'savefig.pad_inches': 0.06,
    })


# ── Data loading ──────────────────────────────────────────────────────────
def load_clip_table():
    """Return one row per (clip_csv_path) with task, actor_uid, attributes,
    and a dict of per-clip features."""
    print('[load] metadata parquet')
    meta = pd.read_parquet(META)
    meta = meta[~meta['move_g1_mujoco_path'].astype(str).str.endswith('_M.csv')]
    meta['csv_rel'] = (meta['move_g1_mujoco_path'].astype(str)
                       .str.replace('\\', '/', regex=False)
                       .str.replace(r'^dataset/', '', regex=True))
    meta = meta[meta['move_duration_frames'].fillna(0) >= 200].copy()

    print(f'[load] per_clip_features cache ({CACHE.stat().st_size/1e6:.0f} MB)')
    cache = json.loads(CACHE.read_text())

    print('[merge] joining metadata × cache')
    rows = []
    for r in meta.itertuples():
        per = cache.get(r.csv_rel)
        if not per:
            continue
        rows.append({
            'csv_rel':         r.csv_rel,
            'task':            r.content_type_of_movement,
            'actor_uid':       r.actor_uid,
            'actor_age_yr':    getattr(r, 'actor_age_yr',    None),
            'actor_weight_kg': getattr(r, 'actor_weight_kg', None),
            'actor_height_cm': getattr(r, 'actor_height_cm', None),
            'actor_gender':    getattr(r, 'actor_gender',    None),
            'per':             per,
        })
    df = pd.DataFrame(rows)
    print(f'  -> {len(df):,} clips with per-clip features')
    return df


def decile_split(df, attr, lo_override=None, hi_override=None):
    """Return (low_actors, high_actors, lo_thr, hi_thr).

    If lo_override / hi_override are provided, they replace the percentile
    cutoffs — used to render variants with intuitive round-number bins
    (e.g. age <= 30 vs >= 60 instead of the auto 10th/90th decile).

    lo_thr / hi_thr are the cutoffs actually applied (so the plot tick
    label reflects the user-facing definition).
    """
    if attr == 'gender':
        f = df[df['actor_gender'].astype(str) == 'F']['actor_uid'].unique()
        m = df[df['actor_gender'].astype(str) == 'M']['actor_uid'].unique()
        return set(f), set(m), None, None
    col = ATTR_COL[attr]
    actor_attr = df.groupby('actor_uid')[col].first().dropna()
    lo_thr = lo_override if lo_override is not None else actor_attr.quantile(DECILE)
    hi_thr = hi_override if hi_override is not None else actor_attr.quantile(1 - DECILE)
    lo = set(actor_attr[actor_attr <= lo_thr].index)
    hi = set(actor_attr[actor_attr >= hi_thr].index)
    return lo, hi, float(lo_thr), float(hi_thr)


# Per-attribute units for the threshold annotations (used under tick labels).
ATTR_UNIT = {'age': 'yr', 'weight': 'kg', 'height': 'cm', 'gender': ''}


def _pretty_unit(u):
    """Replace 'deg' with the degree symbol for nicer axis labels."""
    return u.replace('deg', '°')


# ── Plotting ──────────────────────────────────────────────────────────────
# Seaborn 'deep' palette — designed for publication contrast + colour-blind safe.
COLOR_LOW  = '#4878d0'   # deep blue  (low-attribute decile, plotted left)
COLOR_HIGH = '#ee854a'   # deep orange (high-attribute decile, plotted right)
# Slightly darker dot tints for the strip overlay so individual observations
# read distinct from the box fill.
DOT_LOW    = '#2d5496'
DOT_HIGH   = '#c66423'


def render_panel(ax, df_clips, panel):
    """Draw one box+strip plot on `ax` for one (attr, feat, activity) spec.

    Panel tuple is (attr, feat, activity, unit, low_lbl, high_lbl), optionally
    extended with (lo_cutoff, hi_cutoff) to override the default deciles.
    """
    if len(panel) == 6:
        attr, feat, activity, unit, low_lbl, hi_lbl = panel
        lo_override = hi_override = None
    else:
        attr, feat, activity, unit, low_lbl, hi_lbl, lo_override, hi_override = panel

    sub = df_clips[df_clips['task'] == activity].copy()
    sub['F'] = sub['per'].apply(lambda d: d.get(feat))
    sub = sub.dropna(subset=['F'])

    lo_actors, hi_actors, lo_thr, hi_thr = decile_split(sub, attr, lo_override, hi_override)
    lo = sub[sub['actor_uid'].isin(lo_actors)]
    hi = sub[sub['actor_uid'].isin(hi_actors)]

    n_lo, n_hi = len(lo), len(hi)
    if n_lo < 5 or n_hi < 5:
        print(f'  ! {attr}|{feat}|{activity}: tiny pool (lo={n_lo}, hi={n_hi})')

    # Pearson r over actor means (the same statistic the bar-plot picks
    # activities by). Computed on per-actor means to mirror the page.
    if attr != 'gender':
        actor_F = sub.groupby('actor_uid')['F'].mean()
        actor_a = sub.groupby('actor_uid')[ATTR_COL[attr]].first()
        x = actor_a.reindex(actor_F.index).dropna()
        y = actor_F.reindex(x.index)
        r_val, _ = sp_stats.pearsonr(x.values, y.values)
        r_str = f'$r = {r_val:+.2f}$'
    else:
        actor_F = sub.groupby('actor_uid')['F'].mean()
        actor_g = sub.groupby('actor_uid')['actor_gender'].first()
        x = actor_g.reindex(actor_F.index).dropna().map(lambda g: 0.0 if g == 'F' else 1.0)
        y = actor_F.reindex(x.index)
        r_val, _ = sp_stats.pearsonr(x.values, y.values)
        r_str = f'$r = {r_val:+.2f}$'

    plot_df = pd.DataFrame({
        'F':    list(lo['F']) + list(hi['F']),
        'side': [low_lbl] * n_lo + [hi_lbl] * n_hi,
    })

    palette     = {low_lbl: COLOR_LOW, hi_lbl: COLOR_HIGH}
    dot_palette = {low_lbl: DOT_LOW,   hi_lbl: DOT_HIGH}

    # Box
    sns.boxplot(
        data=plot_df, x='side', y='F', order=[low_lbl, hi_lbl],
        ax=ax, palette=palette,
        width=0.55,
        fliersize=0,                        # outliers drawn by stripplot below
        linewidth=1.2,
        boxprops={'alpha': 0.60, 'edgecolor': '#1c232e'},
        whiskerprops={'color': '#1c232e', 'linewidth': 1.0},
        capprops={'color': '#1c232e', 'linewidth': 1.0},
        medianprops={'color': '#1c232e', 'linewidth': 1.9},
    )
    # Strip
    sns.stripplot(
        data=plot_df, x='side', y='F', order=[low_lbl, hi_lbl],
        ax=ax, palette=dot_palette,
        size=2.4, alpha=0.35, jitter=0.20,
        edgecolor='none',
    )

    # Cosmetics — attribute name goes BELOW the plot (as x-axis label).
    ax.set_xlabel(attr.capitalize(), labelpad=10, fontweight='bold')
    ax.set_ylabel(f'{FEATURE_LABEL[feat]} ({_pretty_unit(unit)})')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', length=0, pad=4)
    ax.yaxis.grid(True, linewidth=0.5, color='#cfd5e0', alpha=0.7)
    ax.set_axisbelow(True)

    # X-tick labels: two-line, with the actual numeric cutoff for that decile
    # in parens beneath the semantic word. Gender uses a blank second line so
    # its x-axis "Gender" label sits at the same height as the other panels'
    # ("Age", "Weight", "Height") below their cutoff lines.
    if attr != 'gender' and lo_thr is not None:
        u = ATTR_UNIT[attr]
        tick_lo = f'{low_lbl}\n($\\leq${int(round(lo_thr))} {u})'
        tick_hi = f'{hi_lbl}\n($\\geq${int(round(hi_thr))} {u})'
    else:
        tick_lo = f'{low_lbl}\n '
        tick_hi = f'{hi_lbl}\n '
    ax.set_xticklabels([tick_lo, tick_hi])

    # Activity label only (no "Activity:" prefix), top-right. Collapse
    # multi-word activity names (e.g. "walking, turning") to the first word.
    activity_short = activity.split(',')[0].split()[0].capitalize()
    ax.text(0.97, 0.97,
            activity_short,
            transform=ax.transAxes,
            ha='right', va='top', fontsize=12, fontweight='600',
            color='#1c232e',
            bbox=dict(boxstyle='round,pad=0.36', facecolor='#ffffff',
                      edgecolor='#9aa3b3', linewidth=0.8, alpha=0.94))


def render_combined(df_clips, panels, out_name):
    # Wider panels + a bit more height so the bigger fonts have room.
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.0))
    for ax, panel in zip(axes, panels):
        render_panel(ax, df_clips, panel)
    fig.tight_layout()
    out = OUT_DIR / out_name
    fig.savefig(out)
    plt.close(fig)
    print(f'[out] wrote {out}')


# ── Variants: different intuitive cutoff sets ────────────────────────────
# Each variant overrides (lo_cutoff, hi_cutoff) per non-gender attribute.
# The activity + feature are fixed (the winners from scan_feat_act_pairs.py).
def _variant(lo_age, hi_age, lo_w, hi_w, lo_h, hi_h):
    return [
        ('age',    'waist_mean_vel',          'dancing',          '°/s', 'Younger', 'Older',   lo_age, hi_age),
        ('weight', 'root_translate_peak_vel', 'walking, turning', 'cm/s','Lighter', 'Heavier', lo_w,   hi_w),
        ('height', 'root_translate_rom',      'climbing box',     'cm',  'Shorter', 'Taller',  lo_h,   hi_h),
        ('gender', 'ankle_rom',               'jogging, turning', '°',   'Female',  'Male'),     # binary
    ]

VARIANTS = [
    # name,                     age cutoffs,  weight,    height
    ('boxplots_all',           (None, None,  None, None, None, None)),  # decile (current)
    ('boxplots_round25_45',    ( 25,  45,    55,   85,   165,  185)),   # cleaner round numbers, moderate
    ('boxplots_extreme30_55',  ( 30,  55,    60,   90,   170,  185)),   # broader bins, more clips
    ('boxplots_tight25_50',    ( 25,  50,    55,   90,   165,  190)),   # tight on one side
    ('boxplots_extreme_30_60', ( 30,  60,    60,   95,   170,  190)),   # very intuitive labels
    ('boxplots_under20_over50',( 20,  50,    55,   90,   165,  190)),   # strict young + clean older
    ('boxplots_under20_over60',( 20,  60,    50,   90,   165,  190)),   # tightest age + weight
    ('boxplots_under22_over60',( 22,  60,    50,   90,   165,  190)),
    ('boxplots_under22_over55',( 22,  55,    50,   90,   165,  190)),
]


def main():
    configure_style()
    df_clips = load_clip_table()
    # v1 / v2 (used earlier) — keep producing them
    render_combined(df_clips, PANELS,    'boxplots_all.pdf')
    render_combined(df_clips, PANELS_V2, 'boxplots_all_v2.pdf')
    # New cutoff variants
    for name, cutoffs in VARIANTS[1:]:    # skip [0] which == PANELS already
        panels = _variant(*cutoffs)
        render_combined(df_clips, panels, f'{name}.pdf')
    print('\nDone.')


if __name__ == '__main__':
    main()
