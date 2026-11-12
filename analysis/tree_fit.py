"""Cross-fit the node scorer of the tree decision layer and evaluate it out of fold.

  python analysis/tree_fit.py --bank scratch/bank/v2x2 [--objective official|pool1] [--scorer direct|hurdle]
                              [--save scratch/bank/v2x2/fit.pkl]

Every node's score for an image of fold k comes from a scorer trained only on the other folds' images (whose maps
are out-of-fold too), so the reported PQ is honest. tau is chosen per fold on the other four folds ("nested").
Two scoring rules are reported for every selection:
  official  organisers' self-evaluation notebook: each annotator record scored separately, counts pooled
  pool1     all records of an image pooled into one ground-truth set, one prediction matches at most one filament
Scorers:
  direct    one regressor of the expected matched IoU
  hurdle    P(the node matches a record) x E[IoU | it matches], two models (the match is a yes/no event,
            the IoU of a match is a narrow distribution around 0.68; modelling them separately is easier)
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'src'))
from treeselect import COL, FEATURES, select

TAUS = [round(x, 3) for x in np.arange(0.14, 0.361, 0.02)]


def load_bank(folder):
    items = []
    for p in sorted(Path(folder).glob('bank_f*_s*of*.pkl')):
        items += pickle.loads(p.read_bytes())['items']
    items.sort(key=lambda it: it['file'])
    return items


def targets(it):
    """per node: official target u (mean over records of matched IoU) and pool1 target v (best record)"""
    iou = it['iou'][:, :len(it['records'])]
    m = iou * (iou > .5)
    return m.mean(1), m.max(1)


def parts(it, chosen):
    """exact counts of a selection under both rules: (tp, fp, fn, s) official, (tp, fp, fn, s) pool1"""
    R = len(it['records'])
    iou = it['iou'][chosen][:, :R] if len(chosen) else np.zeros((0, R), np.float32)
    hit = iou > .5
    tp = hit.sum(0)
    o = np.array([tp.sum(), (len(chosen) - tp).sum(), (np.array(it['n_gt']) - tp).sum(), (iou * hit).sum()], np.float64)
    b = iou.max(1) if len(chosen) and R else np.zeros(0)
    bh = b > .5
    p = np.array([bh.sum(), len(chosen) - bh.sum(), sum(it['n_gt']) - bh.sum(), (b * bh).sum()], np.float64)
    return o, p


def pq(t):
    d = t[0] + .5 * t[1] + .5 * t[2]
    return float(t[3] / d) if d else 0.0


def fixed_rule(it):
    n = it['nodes']
    return np.flatnonzero((np.abs(n[:, COL['level']] - .4) < 1e-6) & (n[:, COL['area']] >= 300) & (n[:, COL['p_mean']] >= .85))


def gbm(kind, a):
    cls = HistGradientBoostingClassifier if kind == 'clf' else HistGradientBoostingRegressor
    return cls(max_iter=a.iters, learning_rate=a.lr, max_leaf_nodes=a.leaves, max_depth=a.depth or None,
               min_samples_leaf=a.min_leaf, l2_regularization=1.0, random_state=2026)


def fit_predict(a, Xtr, items_tr, Xte):
    """scores for the test nodes from models trained on the training images' nodes"""
    official = a.objective == 'official'
    if a.scorer == 'direct':
        y = np.concatenate([targets(it)[0 if official else 1] for it in items_tr])
        w = np.concatenate([np.full(len(it['level']), len(it['records']) if official else 1.0) for it in items_tr])
        return gbm('reg', a).fit(Xtr, y, sample_weight=w).predict(Xte)
    # hurdle: one row per (node, record) for the official rule, one row per node (best record) for pool1
    rows, hit, val = [], [], []
    start = 0
    for it in items_tr:
        n = len(it['level'])
        iou = it['iou'][:, :len(it['records'])]
        if not official:
            iou = iou.max(1, keepdims=True)
        for r in range(iou.shape[1]):
            rows.append(np.arange(start, start + n)); hit.append(iou[:, r] > .5); val.append(iou[:, r])
        start += n
    rows, hit, val = np.concatenate(rows), np.concatenate(hit), np.concatenate(val)
    h = gbm('clf', a).fit(Xtr[rows], hit).predict_proba(Xte)[:, 1]
    m = gbm('reg', a).fit(Xtr[rows[hit]], val[hit]).predict(Xte)
    return h * m


def evaluate(items, scores, tau):
    O, P = np.zeros(4), np.zeros(4)
    per = []
    for it, sc in zip(items, scores):
        o, p = parts(it, select(sc, it['parent'], tau))
        O += o; P += p; per.append((o, p))
    return O, P, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', type=Path, required=True)
    ap.add_argument('--objective', default='official', choices=['official', 'pool1'])
    ap.add_argument('--scorer', default='hurdle', choices=['direct', 'hurdle'])
    ap.add_argument('--drop', nargs='*', default=[], help='feature names to leave out')
    ap.add_argument('--iters', type=int, default=300)
    ap.add_argument('--lr', type=float, default=.05)
    ap.add_argument('--leaves', type=int, default=31)
    ap.add_argument('--depth', type=int, default=0)
    ap.add_argument('--min-leaf', type=int, default=40)
    ap.add_argument('--save', type=Path)
    a = ap.parse_args()
    t0 = time.time()
    items = load_bank(a.bank)
    folds = np.array([it['fold'] for it in items])
    recs = np.array([len(it['records']) for it in items])
    cols = [i for i, f in enumerate(FEATURES) if f not in a.drop]
    j = 0 if a.objective == 'official' else 1
    print(json.dumps(dict(images=len(items), nodes=int(sum(len(it['level']) for it in items)), features=len(cols),
                          objective=a.objective, scorer=a.scorer, iters=a.iters, depth=a.depth, leaves=a.leaves)), flush=True)
    base = [parts(it, fixed_rule(it)) for it in items]
    O = sum(b[0] for b in base); P = sum(b[1] for b in base)
    print('fixed rule: official', round(pq(O), 4), [int(x) for x in O[:3]], '| pool1', round(pq(P), 4), flush=True)

    X = [it['nodes'][:, cols] for it in items]
    scores = [None] * len(items)
    for k in range(5):
        tr = np.flatnonzero(folds != k); te = np.flatnonzero(folds == k)
        pred = fit_predict(a, np.concatenate([X[i] for i in tr]), [items[i] for i in tr], np.concatenate([X[i] for i in te]))
        pos = 0
        for i in te:
            scores[i] = pred[pos:pos + len(X[i])]; pos += len(X[i])
        print(json.dumps(dict(fold=k, fitted=True, secs=round(time.time() - t0))), flush=True)
    yy = np.concatenate([targets(it)[j] for it in items]); ss = np.concatenate(scores)
    print('scorer: correlation with target', round(float(np.corrcoef(yy, ss)[0, 1]), 4), flush=True)

    per_tau = {}
    for tau in TAUS:
        Oo, Pp, per = evaluate(items, scores, tau)
        per_tau[tau] = per
        print(f'tau {tau:.2f}: official {pq(Oo):.4f} (TP {int(Oo[0])} FP {int(Oo[1])} FN {int(Oo[2])} SQ {Oo[3] / max(Oo[0], 1):.4f})  '
              f'pool1 {pq(Pp):.4f}  per image {(Pp[0] + Pp[1]) / len(items):.2f}', flush=True)

    def total(tau, idx, jj):
        return sum(per_tau[tau][i][jj] for i in idx)

    result = {}
    rng = np.random.default_rng(0)
    for name, jj in (('official', 0), ('pool1', 1)):
        chosen = []
        nested = [None] * len(items)
        for k in range(5):
            other = np.flatnonzero(folds != k)
            best = max(TAUS, key=lambda t: pq(total(t, other, jj)))
            chosen.append(best)
            for i in np.flatnonzero(folds == k):
                nested[i] = per_tau[best][i][jj]
        nested = np.array(nested); basej = np.array([b[jj] for b in base])
        per_fold = [round(pq(nested[folds == k].sum(0)), 4) for k in range(5)]
        base_fold = [round(pq(basej[folds == k].sum(0)), 4) for k in range(5)]
        by_rec = {int(r): (round(pq(nested[recs == r].sum(0)), 4), round(pq(basej[recs == r].sum(0)), 4)) for r in (1, 2, 3)}
        draws = rng.integers(0, len(items), (2000, len(items)))
        diff = np.array([pq(nested[d].sum(0)) - pq(basej[d].sum(0)) for d in draws])
        result[name] = dict(nested=round(pq(nested.sum(0)), 4), baseline=round(pq(basej.sum(0)), 4), taus=[float(t) for t in chosen],
                            per_fold=per_fold, baseline_per_fold=base_fold, folds_better=int(sum(a_ > b_ for a_, b_ in zip(per_fold, base_fold))),
                            by_records_nested_vs_base=by_rec, boot_mean=round(float(diff.mean()), 4),
                            boot_5_95=[round(float(np.percentile(diff, 5)), 4), round(float(np.percentile(diff, 95)), 4)],
                            boot_share_positive=round(float((diff > 0).mean()), 3),
                            counts=[int(x) for x in nested.sum(0)[:3]])
        print(name, json.dumps(result[name]), flush=True)
    if a.save:
        a.save.write_bytes(pickle.dumps(dict(files=[it['file'] for it in items], scores=scores, result=result,
                                             features=[FEATURES[i] for i in cols], args=vars(a) | {'bank': str(a.bank), 'save': str(a.save)})))
    print(json.dumps(dict(done=True, secs=round(time.time() - t0))), flush=True)


if __name__ == '__main__':
    main()
