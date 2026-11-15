"""Cross-fit the learned keep rule (src/keep.py) and evaluate it out of fold. CPU only, about a minute.

  python analysis/keep_fit.py --tables scratch/keep/oof --bank scratch/ws/main/rbank/kc_bridge
                              [--scores scratch/keep/oof/scores.npz] [--table weights/keep_table.npz]

The seeds of fold k are scored by a model fitted on the seeds of the other four folds (whose probability maps are
out of fold too). The threshold tau is chosen in two ways, both reported:
  nested         per fold, on the other four folds, from the same cross-fitted scores. Those scores come from
                 models that saw fold k's labels, so this is slightly optimistic (about 0.001 here).
  strict nested  per fold, from an inner cross-fit among the other four folds that never touches fold k.
PQ is computed from the stored IoU of every seed with every record (stage-1 masks): a kept seed adds its matched
IoU per record, and the seeds are disjoint, so they can be summed independently. This agrees with scoring real
label maps except when one seed matches two overlapping filaments of the same record (1 seed in 6788: one TP).
analysis/keep_eval.py repeats the comparison on real label maps, with the final mask shapes.

--scores writes the cross-fitted score of every seed (bank order) for keep_eval.py.
--table writes the training table of the final model (all five folds), which src/predict_final.py refits.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'src'))
sys.path.insert(0, str(WORK / 'analysis'))
from tree_bank import gt_index
import keep
import refine

GT_INDEX = WORK / 'scratch/bank/gt_index.pkl'      # TRAIN annotations as pixel lists; built on first use (about 2 minutes)
TAUS = np.round(np.arange(.16, .331, .01), 2)


def pq(t):
    return float(t[0] / (t[1] + .5 * t[2] + .5 * t[3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tables', type=Path, required=True, help='folder with keep_f0..4.npz (analysis/keep_bank.py)')
    ap.add_argument('--bank', type=Path, required=True, help='the seed bank the tables were built from (for the image list)')
    ap.add_argument('--scores', type=Path)
    ap.add_argument('--table', type=Path)
    a = ap.parse_args()
    X, fold, fname, label, kept, iou = [], [], [], [], [], []
    members = None
    for k in range(5):
        with np.load(a.tables / f'keep_f{k}.npz', allow_pickle=False) as z:
            members = int(z['members'])
            X.append(z['X']); fname.append(z['fname']); label.append(z['label']); kept.append(z['kept']); iou.append(z['best_iou'])
            fold += [k] * len(z['X'])
    X, fold, fname, label = np.concatenate(X), np.array(fold), np.concatenate(fname).astype(str), np.concatenate(label)
    kept, iou = np.concatenate(kept), np.concatenate(iou)
    u, R, s_sum, tp = keep.target(iou)
    gt = gt_index(GT_INDEX)
    image_fold = {f: k for k in range(5) for f in refine.load_images(a.bank, k)}     # includes images without seeds
    images = sorted(image_fold)
    index = {f: i for i, f in enumerate(images)}
    img = np.array([index[f] for f in fname])
    ifold = np.array([image_fold[f] for f in images])
    assert np.array_equal(ifold[img], fold), 'a seed is listed under another fold than its image'
    n_gt = np.array([sum(len(rec) for _, rec in gt[f]) for f in images], np.float64)
    n_rec = np.array([len(gt[f]) for f in images])

    def per_image(keep_mask):
        S = np.bincount(img[keep_mask], weights=s_sum[keep_mask], minlength=len(images))
        TP = np.bincount(img[keep_mask], weights=tp[keep_mask], minlength=len(images))
        FP = np.bincount(img[keep_mask], weights=(R - tp)[keep_mask], minlength=len(images))
        return np.stack([S, TP, FP, n_gt - TP], 1)

    score = np.zeros(len(X))
    for k in range(5):
        tr, te = fold != k, fold == k
        assert not set(fname[tr]) & set(fname[te])
        assert not {f[:8] for f in fname[tr]} & {f[:8] for f in fname[te]}, 'an observation day is in two folds'
        score[te] = keep.fit(X[tr], u[tr], R[tr]).predict(X[te])
    base = per_image(kept)
    per_tau = [per_image(score > t) for t in TAUS]
    nested, chosen = base.copy(), []
    for k in range(5):
        j = int(np.argmax([pq(p[ifold != k].sum(0)) for p in per_tau]))
        chosen.append(float(TAUS[j]))
        nested[ifold == k] = per_tau[j][ifold == k]
    # strict nesting: the threshold for fold k comes from scores that models without any fold-k seed produced
    strict, strict_taus = base.copy(), []
    for k in range(5):
        inner = np.zeros(len(X))
        for j in range(5):
            if j != k:
                tr = (fold != k) & (fold != j)
                inner[fold == j] = keep.fit(X[tr], u[tr], R[tr]).predict(X[fold == j])
        others = ifold != k
        best = max(TAUS, key=lambda t: pq(per_image((inner > t) & (fold != k))[others].sum(0)))
        strict_taus.append(float(best))
        strict[ifold == k] = per_image(score > best)[ifold == k]
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(images), (2000, len(images)))
    diff = np.array([pq(nested[d].sum(0)) - pq(base[d].sum(0)) for d in draws])
    fixed = per_tau[int(np.argmin(np.abs(TAUS - keep.TAU)))]
    result = dict(
        seeds=len(X), descriptors=X.shape[1], fixed_rule=round(pq(base.sum(0)), 4), nested=round(pq(nested.sum(0)), 4),
        gain=round(pq(nested.sum(0)) - pq(base.sum(0)), 4), taus=chosen,
        strict_nested=round(pq(strict.sum(0)), 4), strict_gain=round(pq(strict.sum(0)) - pq(base.sum(0)), 4), strict_taus=strict_taus,
        strict_per_fold=[round(pq(strict[ifold == k].sum(0)), 4) for k in range(5)],
        per_fold=[round(pq(nested[ifold == k].sum(0)), 4) for k in range(5)],
        fixed_rule_per_fold=[round(pq(base[ifold == k].sum(0)), 4) for k in range(5)],
        by_records={int(r): [round(pq(nested[n_rec == r].sum(0)), 4), round(pq(base[n_rec == r].sum(0)), 4)] for r in (1, 2, 3)},
        boot_5_95=[round(float(np.percentile(diff, 5)), 4), round(float(np.percentile(diff, 95)), 4)],
        boot_share_positive=round(float((diff > 0).mean()), 3),
        at_keep_tau={'tau': keep.TAU, 'pq': round(pq(fixed.sum(0)), 4), 'kept': int((score > keep.TAU).sum()), 'fixed_rule_kept': int(kept.sum())},
        curve={f'{t:.2f}': round(pq(p.sum(0)), 4) for t, p in zip(TAUS, per_tau)},
        corr_with_target=round(float(np.corrcoef(score, u)[0, 1]), 4))
    result['folds_better'] = int(sum(x > y for x, y in zip(result['per_fold'], result['fixed_rule_per_fold'])))
    print(json.dumps(result, indent=1))
    if a.scores:
        np.savez_compressed(a.scores, fname=fname, label=label, fold=fold, score=score, kept=kept)
    if a.table:
        a.table.parent.mkdir(parents=True, exist_ok=True)
        keep.save_table(a.table, X, u, R, members)


if __name__ == '__main__':
    main()
