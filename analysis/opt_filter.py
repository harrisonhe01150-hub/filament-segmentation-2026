"""Rubric statistics and a learned keep/drop filter, scored on the OOF component table.

Baseline keeps components with area >= 300 and mean prob >= .85 (the v2 rule). The candidate
replaces that rule with a gradient-boosted regressor that predicts each component's PQ value
(mean over expert records of its matched IoU, 0 if unmatched) from shape/confidence features and
keeps it above a threshold tau. Everything is nested by the date-grouped folds: for outer fold k,
the model and tau only ever see the other four folds.

usage: python analysis/opt_filter.py <table.pkl>
"""
import pickle
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

T = pickle.loads(open(sys.argv[1], 'rb').read())
ROWS, NAMES = T['rows'], T['names']
IMG_FOLD = np.array([r['fold'] for r in ROWS])
SIZES = [len(r['feats']) for r in ROWS]
X = np.concatenate([r['feats'] for r in ROWS])
CFOLD = np.repeat(IMG_FOLD, SIZES)
AREA, MEAN = X[:, NAMES.index('area')], X[:, NAMES.index('mean')]
VALID = AREA > 0
# a component's PQ value: matched IoU averaged over the image's expert records
Y = np.concatenate([np.mean([np.where(i.max(0, initial=0) > .5, i.max(0, initial=0), 0) for i in r['ious']], 0)
                    for r in ROWS])
TAUS = np.round(np.arange(0, .61, .01), 2)


def split(mask):
    return np.split(mask, np.cumsum(SIZES)[:-1])


def parts(keep, folds=range(5)):
    """Official rule (IoU > .5 is TP). Returns [iou_sum, tp, fp, fn]."""
    out = np.zeros(4)
    for r, k in zip(ROWS, split(keep)):
        if r['fold'] not in folds:
            continue
        for iou in r['ious']:
            iou = iou[:, k]
            if iou.shape[0] == 0:
                out[2] += iou.shape[1]; continue
            if iou.shape[1] == 0:
                out[3] += iou.shape[0]; continue
            hit = iou > .5
            out += iou[hit].sum(), hit.sum(), (~hit.any(0)).sum(), (~hit.any(1)).sum()
    return out


def pq(p):
    return p[0] / (p[1] + .5 * p[2] + .5 * p[3])


def rubric(keep):
    """Self-evaluation notebook statistics: all pairs with IoU > 0, and 1:n / n:1 overlap degrees."""
    ious, g_deg, p_deg = [], [], []
    for r, k in zip(ROWS, split(keep)):
        for iou in r['ious']:
            iou = iou[:, k]
            ious.append(iou[iou > 0])
            g_deg += (iou > 0).sum(1).tolist()
            p_deg += (iou > 0).sum(0).tolist()
    ious, g_deg, p_deg = np.concatenate(ious), np.array(g_deg), np.array(p_deg)
    return dict(mean_iou=ious.mean(), mean_dice=(2 * ious / (1 + ious)).mean(), pairs=len(ious),
                gt_missed=int((g_deg == 0).sum()), gt_1to1=int((g_deg == 1).sum()), gt_split=int((g_deg >= 2).sum()),
                pred_missed=int((p_deg == 0).sum()), pred_1to1=int((p_deg == 1).sum()), pred_merged=int((p_deg >= 2).sum()))


def rule(area, mean):
    return VALID & (AREA >= area) & (MEAN >= mean)


def fit_predict(train, target):
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, random_state=0)
    m.fit(X[train & VALID], Y[train & VALID])
    out = np.full(len(X), -1.)
    out[target] = m.predict(X[target])
    return out


def best_tau(score, folds):
    return max(TAUS, key=lambda t: pq(parts(VALID & (score > t), folds)))


def best_rule(folds):
    grid = [(a, m) for a in (0, 100, 200, 300, 400, 500, 700) for m in np.round(np.arange(.6, .96, .05), 2)]
    return max(grid, key=lambda g: pq(parts(rule(*g), folds)))


if __name__ == '__main__':
    base = rule(300, .85)
    nested_base, cand = np.zeros(len(X), bool), np.zeros(len(X), bool)
    print('fold  v2-rule  v2-rule(nested)  learned(nested)  tau  rule')
    for k in range(5):
        rest = [j for j in range(5) if j != k]
        inner = np.full(len(X), -1.)
        for j in rest:
            inner[CFOLD == j] = fit_predict((CFOLD != k) & (CFOLD != j), CFOLD == j)[CFOLD == j]
        tau = best_tau(inner, rest)
        cand[CFOLD == k] = (VALID & (fit_predict(CFOLD != k, CFOLD == k) > tau))[CFOLD == k]
        g = best_rule(rest)
        nested_base[CFOLD == k] = rule(*g)[CFOLD == k]
        print(f'{k}     {pq(parts(base, [k])):.4f}   {pq(parts(nested_base, [k])):.4f}           '
              f'{pq(parts(cand, [k])):.4f}           {tau:.2f} {g}', flush=True)
    for name, keep in [('v2 rule (tuned on all folds)', base), ('v2 rule, nested', nested_base), ('learned, nested', cand)]:
        p = parts(keep)
        print(f'{name:30s} PQ {pq(p):.4f}  TP {p[1]:.0f} FP {p[2]:.0f} FN {p[3]:.0f}  SQ {p[0]/p[1]:.3f}')
        print('   ', {k: round(v, 4) if isinstance(v, float) else v for k, v in rubric(keep).items()})
    gains = [pq(parts(cand, [k])) - pq(parts(nested_base, [k])) for k in range(5)]
    total = pq(parts(cand)) - pq(parts(nested_base))
    print('gain vs nested v2 rule:', round(total, 4), 'per fold', np.round(gains, 4).tolist(),
          '| gate (>= .005 overall, better in >= 4/5 folds):', total >= .005 and sum(g > 0 for g in gains) >= 4)
