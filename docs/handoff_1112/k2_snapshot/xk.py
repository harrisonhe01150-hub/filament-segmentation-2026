"""Explorer "stronger keep model": shared harness.

Everything is evaluated with the stored-IoU formula of analysis/keep_fit.py (stage-1 masks): a kept seed adds its
matched IoU per annotator record. Strict nesting as there: the seeds of fold k are scored by a model trained on the
other four folds, and the threshold for fold k is the best one on an inner cross-fit among those four folds.

A learner is a function  learner(D, tr, te) -> scores of the seeds te  (index arrays into the 6788 seeds); it may use
any field of D for the rows tr and only the descriptors for the rows te.

Cross-fitted scores of a learner on a split are cached in cache/<learner>__<split>.npz:
  outer (n,)     score of every seed from the model trained without the seed's fold
  inner (5, n)   inner[k, i] = score of seed i (fold j != k) from the model trained without folds k and j
The model "without folds k and j" serves both inner[k, fold j] and inner[j, fold k], so a split costs 15 fits.

Splits: 'orig' = the original folds; 'rs<s>' = audit_lib.day_folds(images, s) (random, grouped by observation day);
the confirmation set of the task is rs1..rs10, the development set used for choosing is orig + rs101..rs105.
"""
import hashlib
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
sys.path.insert(0, str(WORK / 'src'))
TABLES = WORK / 'scratch/keep/oof'
BANK = WORK / 'scratch/ws/main/rbank/kc_bridge'
GT_INDEX = WORK / 'scratch/bank/gt_index.pkl'
CACHE = HERE / 'cache'
TAUS = np.round(np.arange(.16, .331, .01), 2)
DEV = ['orig'] + [f'rs{s}' for s in range(101, 106)]
CONFIRM = [f'rs{s}' for s in range(1, 11)]


def pq(t):
    return float(t[0] / (t[1] + .5 * t[2] + .5 * t[3]))


def target(iou):
    """as keep.target: u, R, s, tp from best_iou (n, 3; -1 = no such record)"""
    iou = np.asarray(iou, np.float64)
    R = (iou >= 0).sum(1)
    hit = iou > .5
    s = (iou * hit).sum(1)
    return s / np.maximum(R, 1), R, s, hit.sum(1)


class Data:
    def __init__(self):
        path = HERE / 'data.pkl'
        if path.exists():
            self.__dict__.update(pickle.loads(path.read_bytes()))
        else:
            import refine
            X, fold, fname, label, kept, iou = [], [], [], [], [], []
            for k in range(5):
                with np.load(TABLES / f'keep_f{k}.npz', allow_pickle=False) as z:
                    X.append(z['X']); fname.append(z['fname']); label.append(z['label']); kept.append(z['kept']); iou.append(z['best_iou'])
                    fold += [k] * len(z['X'])
            self.X, self.fold0 = np.concatenate(X), np.array(fold)
            self.fname, self.label = np.concatenate(fname).astype(str), np.concatenate(label)
            self.kept, self.iou = np.concatenate(kept), np.concatenate(iou).astype(np.float64)
            gt = pickle.loads(GT_INDEX.read_bytes())
            image_fold = {f: k for k in range(5) for f in refine.load_images(BANK, k)}
            self.images = sorted(image_fold)
            index = {f: i for i, f in enumerate(self.images)}
            self.img = np.array([index[f] for f in self.fname])
            self.ifold0 = np.array([image_fold[f] for f in self.images])
            assert np.array_equal(self.ifold0[self.img], self.fold0)
            self.n_gt = np.array([sum(len(rec) for _, rec in gt[f]) for f in self.images], np.float64)
            self.n_rec = np.array([len(gt[f]) for f in self.images])
            path.write_bytes(pickle.dumps(self.__dict__))
        self.u, self.R, self.s_sum, self.tp = target(self.iou)
        self.w = self.R.astype(np.float64)
        self.day = np.array([f[:8] for f in self.images])
        self.n = len(self.X)
        self.extra = {}

    def per_image(self, keep_mask, s_sum=None, tp=None):
        s_sum = self.s_sum if s_sum is None else s_sum
        tp = self.tp if tp is None else tp
        m = len(self.images)
        S = np.bincount(self.img[keep_mask], weights=s_sum[keep_mask], minlength=m)
        TP = np.bincount(self.img[keep_mask], weights=tp[keep_mask], minlength=m)
        FP = np.bincount(self.img[keep_mask], weights=(self.R - tp)[keep_mask], minlength=m)
        return np.stack([S, TP, FP, self.n_gt - TP], 1)

    def ifold(self, split):
        """image fold of a split name ('orig' or 'rs<seed>')"""
        if split == 'orig':
            return self.ifold0
        f = day_folds(self.images, int(split[2:]))
        days = defaultdict(set)
        for d, k in zip(self.day, f):
            days[d].add(int(k))
        assert all(len(v) == 1 for v in days.values()), 'an observation day is split'
        return f


def day_folds(images, seed, n_folds=5):
    """Random split of the images grouped by observation day (first 8 characters); the same function as
    scratch/ws/k1_audit/audit_lib.day_folds, so rs1..rs10 are the audit's ten re-splits."""
    groups = defaultdict(list)
    for f in sorted(images):
        groups[f[:8]].append(f)
    keys = sorted(groups, key=lambda k: hashlib.md5(f'{seed}-{k}'.encode()).hexdigest())
    sizes = [0] * n_folds
    out = {}
    for k in sorted(keys, key=lambda k: -len(groups[k])):
        i = min(range(n_folds), key=lambda i: (sizes[i], i))
        for f in groups[k]:
            out[f] = i
        sizes[i] += len(groups[k])
    return np.array([out[f] for f in images])


def crossfit(D, learner, split, name=None, verbose=True):
    """outer (n,) and inner (5, n) scores of a learner on a split; cached when name is given."""
    path = CACHE / f'{name}__{split}.npz' if name else None
    if path is not None and path.exists():
        with np.load(path) as z:
            return z['outer'], z['inner']
    ifold = D.ifold(split)
    fold = ifold[D.img]
    outer, inner = np.full(D.n, np.nan), np.full((5, D.n), np.nan)
    t0 = time.time()
    for k in range(5):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        assert not set(D.img[tr]) & set(D.img[te])
        outer[te] = learner(D, tr, te)
    for k in range(5):
        for j in range(k + 1, 5):
            tr = np.flatnonzero((fold != k) & (fold != j))
            te = np.flatnonzero((fold == k) | (fold == j))
            s = learner(D, tr, te)
            fj = fold[te] == j
            inner[k, te[fj]] = s[fj]
            inner[j, te[~fj]] = s[~fj]
    if verbose:
        print(f'  crossfit {name} {split}: {time.time() - t0:.0f}s', flush=True)
    if path is not None:
        CACHE.mkdir(exist_ok=True)
        np.savez_compressed(path, outer=outer, inner=inner)
    return outer, inner


def evaluate(D, outer, inner, split, taus=TAUS, s_sum=None, tp=None):
    """Strictly nested per-image parts and a few other read-outs of a set of cross-fitted scores."""
    ifold = D.ifold(split)
    fold = ifold[D.img]
    kw = dict(s_sum=s_sum, tp=tp)
    per_tau = [D.per_image(outer > t, **kw) for t in taus]
    strict, strict_taus = np.zeros((len(D.images), 4)), []
    loose, loose_taus = np.zeros((len(D.images), 4)), []
    for k in range(5):
        others = ifold != k
        rest = fold != k
        j = int(np.argmax([pq(D.per_image(rest & (np.nan_to_num(inner[k], nan=-1e9) > t), **kw)[others].sum(0)) for t in taus]))
        strict_taus.append(float(taus[j]))
        strict[~others] = per_tau[j][~others]
        j = int(np.argmax([pq(p[others].sum(0)) for p in per_tau]))
        loose_taus.append(float(taus[j]))
        loose[~others] = per_tau[j][~others]
    curve = np.array([pq(p.sum(0)) for p in per_tau])
    out = dict(strict=pq(strict.sum(0)), strict_taus=strict_taus, loose=pq(loose.sum(0)), loose_taus=loose_taus,
               strict_fold=[pq(strict[ifold == k].sum(0)) for k in range(5)],
               strict_rec=[pq(strict[D.n_rec == r].sum(0)) for r in (1, 2, 3)],
               curve=curve, parts=strict, ifold=ifold)
    for t in (0.2245, 0.25):
        out[f'at_{t}'] = pq(D.per_image(outer > t, **kw).sum(0))
    plateau = (taus >= 0.195) & (taus <= 0.265)
    out['plateau'] = float(curve[plateau].mean()) if plateau.any() else float('nan')
    out['best_tau'] = float(curve.max())
    out['corr'] = float(np.corrcoef(outer, D.u)[0, 1])
    return out


def boot(a, b, n=2000, seed=0):
    """paired bootstrap over images of pq(a) - pq(b); a, b: per-image parts. Returns the n differences."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(a), (n, len(a)))
    A, B = a[draws].sum(1), b[draws].sum(1)
    return A[:, 0] / (A[:, 1] + .5 * A[:, 2] + .5 * A[:, 3]) - B[:, 0] / (B[:, 1] + .5 * B[:, 2] + .5 * B[:, 3])


def blend(parts):
    """mean of several (outer, inner) pairs"""
    return np.mean([p[0] for p in parts], 0), np.mean([p[1] for p in parts], 0)
