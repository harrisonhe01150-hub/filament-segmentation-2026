"""Why do ground-truth filaments end up as false negatives under the v2 rule (area >= 300, mean >= .85)?

Each (expert record, GT filament) with best IoU <= .5 is put in the first bucket that applies:
  split     - >= 2 kept predictions overlap it and their union would reach IoU > .5
  merged    - its best prediction also covers other GT filaments; cutting those parts out would reach IoU > .5
  too big   - best prediction is larger than the GT and covers >= half of it
  too small - best prediction is smaller than the GT and lies >= half inside it
  other     - overlaps, but none of the above (offset / wrong shape)
  dropped   - no kept prediction overlaps, but a component removed by the area/mean filter does
  no signal - nothing above threshold .4 overlaps it at all
"expert disagreement" flags near misses whose best prediction is a true positive for another expert of the same image.

usage: python analysis/diagnose.py <table.pkl>
"""
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np

BUCKETS = ['split', 'merged', 'too big', 'too small', 'other', 'dropped', 'no signal']


def diagnose(T):
    """Returns (bucket counts, pred/GT area ratios, GT sizes, best IoUs, expert-disagreement counts)."""
    A, M = T['names'].index('area'), T['names'].index('mean')
    buckets, ratio, gt_size, best_iou = Counter(), defaultdict(list), defaultdict(list), defaultdict(list)
    disagree = Counter()
    for r in T['rows']:
        f = r['feats']
        keep = r['keep'] if 'keep' in r else (f[:, A] >= 300) & (f[:, M] >= .85)  # 'keep': final pipeline
        P = f[:, A]
        tp_somewhere = np.zeros(len(f), bool)  # component is a TP for at least one expert of this image
        for iou in r['ious']:
            tp_somewhere |= (iou > .5).any(0)
        for iou_all, inter_all, G in zip(r['ious'], r['inters'], r['gt_areas']):
            iou, inter = iou_all[:, keep], inter_all[:, keep]
            Pk, kept_idx = P[keep], np.flatnonzero(keep)
            for g in range(len(G)):
                best = iou[g].max(initial=0)
                if best > .5:
                    continue
                if best == 0:
                    b = 'dropped' if inter_all[g].any() else 'no signal'
                else:
                    ov = inter[g] > 0
                    j = int(iou[g].argmax())
                    I, Pj = inter[g, j], Pk[j]
                    union_iou = inter[g, ov].sum() / (G[g] + Pk[ov].sum() - inter[g, ov].sum())
                    others = inter[:, j].sum() - I
                    cut_iou = I / (G[g] + Pj - others - I)
                    if ov.sum() >= 2 and union_iou > .5:
                        b = 'split'
                    elif others > 0 and cut_iou > .5:
                        b = 'merged'
                    elif Pj > G[g] and I / G[g] >= .5:
                        b = 'too big'
                    elif Pj < G[g] and I / Pj >= .5:
                        b = 'too small'
                    else:
                        b = 'other'
                    ratio[b].append(Pj / G[g])
                    best_iou[b].append(best)
                    if tp_somewhere[kept_idx[j]]:
                        disagree[b] += 1
                buckets[b] += 1
                gt_size[b].append(G[g])
    return buckets, ratio, gt_size, best_iou, disagree


if __name__ == '__main__':
    buckets, ratio, gt_size, best_iou, disagree = diagnose(pickle.loads(open(sys.argv[1], 'rb').read()))
    total = sum(buckets.values())
    print(f'false negatives: {total}')
    print(f'{"bucket":12s} {"count":>6s} {"share":>6s} {"med GT px":>9s} {"med pred/GT":>11s} {"med best IoU":>12s} '
          f'{"IoU .4-.5":>9s} {"expert disagr.":>14s}')
    for b in BUCKETS:
        n = buckets[b]
        rr = f'{np.median(ratio[b]):.2f}' if ratio[b] else '-'
        bi = f'{np.median(best_iou[b]):.2f}' if best_iou[b] else '-'
        near = sum(x > .4 for x in best_iou[b])
        print(f'{b:12s} {n:6d} {n / total:6.1%} {np.median(gt_size[b]):9.0f} {rr:>11s} {bi:>12s} {near:9d} {disagree[b]:14d}')
