"""Error-analysis figures for the report, from the v2 out-of-fold predictions (all 707 training images).

  fig_overlap_scores.*   IoU and Dice of every GT/prediction pair that overlaps (self-evaluation notebook definition)
  fig_overlap_degrees.*  1:n and n:1 overlap counts (how often filaments are split or merged)
  fig_fn_causes.*        why GT filaments are missed (diagnose.py buckets)
  fig_examples.*         image crops with GT and predicted outlines, three good and three poor images

usage: SOLAR_DATA=<data dir> python analysis/report_figs.py <oof runs dir> <table.pkl> <out dir>
<oof runs dir> holds v2/probs_f0-1.npz and v2b/probs_f2-4.npz; <table.pkl> comes from build_table.py.
"""
import pickle
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

OOF, TABLE, OUT = (Path(a) for a in sys.argv[1:4])
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PQ, ROOT, ann_mask, load_coco  # noqa: E402
from diagnose import BUCKETS, diagnose  # noqa: E402
from fastsweep import components  # noqa: E402
from postproc import upsample  # noqa: E402

# reference palette, light mode (validated: scripts/validate_palette.js "#2a78d6,#eb6834" --mode light)
SURFACE, INK, INK2, GRID, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df', '#a3a29d'
BLUE, ORANGE = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'axes.titlesize': 8.5, 'axes.titleweight': 'bold',
                     'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                     'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def grid_y(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=.6)
    ax.set_axisbelow(True)


def kept_overlaps(T):
    """IoU>0 pairs and overlap degrees under the v2 rule."""
    A, M = T['names'].index('area'), T['names'].index('mean')
    pairs, g_deg, p_deg = [], [], []
    for r in T['rows']:
        keep = (r['feats'][:, A] >= 300) & (r['feats'][:, M] >= .85)
        for iou in r['ious']:
            iou = iou[:, keep]
            pairs.append(iou[iou > 0])
            g_deg += (iou > 0).sum(1).tolist()
            p_deg += (iou > 0).sum(0).tolist()
    return np.concatenate(pairs), np.array(g_deg), np.array(p_deg)


def fig_scores(pairs):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.2))
    for ax, vals, name in [(axes[0], pairs, 'IoU'), (axes[1], 2 * pairs / (1 + pairs), 'Dice')]:
        h, _, _ = ax.hist(vals, bins=np.linspace(0, 1, 41), color=BLUE, edgecolor=SURFACE, linewidth=.8,
                          weights=np.full(len(vals), 1 / len(vals)))
        top = h.max() * 1.22  # headroom so labels sit above the bars
        ax.set_ylim(0, top)
        ax.axvline(vals.mean(), color=INK2, linestyle='--', linewidth=1)
        ax.text(vals.mean() - .015, top * .93, f'mean {vals.mean():.3f}', ha='right', va='top', color=INK2)
        if name == 'IoU':
            ax.axvline(.5, color=MUTED, linewidth=.8)
            ax.text(.485, top * .55, 'match\nthreshold\n(IoU > 0.5)', ha='right', color=INK2, fontsize=7)
        ax.set(xlabel=name, xlim=(0, 1))
        ax.set_title(f'{name} of overlapping pairs (n = {len(vals):,})', loc='left')
        grid_y(ax)
    axes[0].set_ylabel('share of pairs')
    save(fig, 'fig_overlap_scores')


def fig_degrees(g_deg, p_deg):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.2))
    for ax, deg, title, lab, zero in [(axes[0], g_deg, 'GT filaments by overlapping predictions', '1:{}', 'missed'),
                                      (axes[1], p_deg, 'Predictions by overlapping GT filaments', '{}:1', 'spurious')]:
        counts = [int((deg == k).sum()) for k in (0, 1, 2)] + [int((deg >= 3).sum())]
        names = [lab.format(k) for k in ('0', '1', '2', '3+')]
        names[0] += f'\n({zero})'
        names[2] += '\n(split)' if lab == '1:{}' else '\n(merged)'
        bars = ax.bar(names, counts, color=[MUTED, BLUE, BLUE, BLUE], width=.6)
        for b, c in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2, c, f'{c:,}', ha='center', va='bottom', color=INK2, fontsize=7)
        ax.set_ylim(0, max(counts) * 1.12)
        ax.set_title(title, loc='left')
        grid_y(ax)
    axes[0].set_ylabel('count')
    save(fig, 'fig_overlap_degrees')


def fig_causes(buckets):
    label = {'split': 'split into pieces', 'merged': 'merged with neighbours', 'too big': 'predicted too large',
             'too small': 'predicted too small', 'other': 'other shape error',
             'dropped': 'removed by area/confidence filter', 'no signal': 'not detected at all'}
    order = sorted(BUCKETS, key=lambda b: buckets[b])
    total = sum(buckets.values())
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    colors = [ORANGE if b in ('dropped', 'no signal') else BLUE for b in order]
    ax.barh([label[b] for b in order], [buckets[b] for b in order], color=colors, height=.62)
    for i, b in enumerate(order):
        ax.text(buckets[b] + 12, i, f'{buckets[b]:,}  ({buckets[b] / total:.0%})', va='center', color=INK2, fontsize=7)
    ax.set_xlim(0, max(buckets.values()) * 1.25)
    ax.xaxis.grid(True, color=GRID, linewidth=.6)
    ax.set_axisbelow(True)
    ax.set_xlabel('missed ground-truth filaments (expert records, n = {:,})'.format(total))
    ax.set_title('Why ground-truth filaments are missed', loc='left')
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    ax.legend(handles, ['overlapped, but IoU <= 0.5', 'no kept prediction overlaps'], frameon=False,
              loc='lower right', fontsize=7)
    save(fig, 'fig_fn_causes')


def outline(mask, width=3):
    m = mask.astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((width, width), np.uint8)).astype(bool)


def fig_examples(T):
    A, M = T['names'].index('area'), T['names'].index('mean')
    per = []
    for row in T['rows']:  # per-image PQ under the v2 rule
        keep = (row['feats'][:, A] >= 300) & (row['feats'][:, M] >= .85)
        pq = PQ()
        for iou in row['ious']:
            pq.add(iou[:, keep])
        per.append(dict(file=row['file'], fold=row['fold'], pq=pq.value, n_gt=pq.tp + pq.fn, records=len(row['ious'])))
    per = [r for r in per if r['n_gt'] >= 4 * r['records']]  # at least ~4 GT filaments per expert
    per.sort(key=lambda r: r['pq'])
    picks = [(r, 'good') for r in per[-3:][::-1]] + [(r, 'poor') for r in per[len(per) // 20:len(per) // 20 + 3]]
    _, anns, by_file = load_coco()
    probs = {}
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.6))
    for ax, (r, kind) in zip(axes.ravel(), picks):
        f, k = r['file'], r['fold']
        if k not in probs:
            z = np.load(OOF / ('v2' if k < 2 else 'v2b') / f'probs_f{k}.npz')
            probs[k] = {str(n): p for n, p in zip(z['val_files'], z['val_prob'])}
        lab, feat = components(upsample(probs[k][f]), .4, 8)
        keep = np.flatnonzero((feat[:, 0] >= 300) & (feat[:, 1] >= .85)) + 1
        pred = np.isin(lab, keep)
        gts = [ann_mask(a) for a in anns[by_file[f][0]]]
        gt = np.any(gts, 0)
        big = max(gts, key=lambda m: m.sum())
        cy, cx = (int(v.mean()) for v in np.nonzero(big))
        h = 400
        y0, x0 = np.clip(cy - h, 0, 2048 - 2 * h), np.clip(cx - h, 0, 2048 - 2 * h)
        img = cv2.imread(str(ROOT / 'train/train_images' / f), cv2.IMREAD_GRAYSCALE)
        rgb = np.repeat(img[..., None], 3, 2).astype(np.float32) / 255
        for m, c in [(outline(gt), BLUE), (outline(pred), ORANGE)]:
            rgb[m] = np.array(matplotlib.colors.to_rgb(c))
        ax.imshow(rgb[y0:y0 + 2 * h, x0:x0 + 2 * h])
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f'{kind.capitalize()} · {f[:4]}-{f[4:6]}-{f[6:8]} · PQ {r["pq"]:.2f}', loc='left',
                     fontsize=7, fontweight='normal')
    handles = [plt.Line2D([], [], color=BLUE, linewidth=2), plt.Line2D([], [], color=ORANGE, linewidth=2)]
    fig.legend(handles, ['ground truth (first expert)', 'v2 prediction (out-of-fold)'], frameon=False, ncol=2,
               loc='lower center', bbox_to_anchor=(.5, -.01), fontsize=7.5)
    fig.subplots_adjust(wspace=.04, hspace=.14, bottom=.06)
    save(fig, 'fig_examples')
    return [(r['file'], kind, round(r['pq'], 3)) for r, kind in picks]


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    T = pickle.loads(TABLE.read_bytes())
    pairs, g_deg, p_deg = kept_overlaps(T)
    fig_scores(pairs)
    fig_degrees(g_deg, p_deg)
    fig_causes(diagnose(T)[0])
    print('examples:', fig_examples(T))
    print(f'pairs {len(pairs)} mean IoU {pairs.mean():.4f} mean Dice {(2 * pairs / (1 + pairs)).mean():.4f}')
