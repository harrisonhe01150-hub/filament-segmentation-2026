"""Out-of-fold check of the learned keep rule on real label maps, with the mask shapes of the final pipeline.

  python analysis/keep_eval.py --scores scratch/keep/oof/scores.npz --bank scratch/ws/main/rbank/kc_bridge --shape stage1
  python analysis/keep_eval.py ... --shape draw --members '[{"0": "v2", ...}, {"0": "v2_5070ti", ...}]'
  python analysis/keep_eval.py ... --shape refine --pred scratch/ws/main/kcb/pred

For every image the label map is built and scored with postproc.score_file (the organisers' rule), once with the
fixed keep rule and once per threshold tau of the learned rule (cross-fitted scores from analysis/keep_fit.py).
tau is chosen per fold on the other four folds. Shapes:
  stage1  every kept seed keeps its stage-1 mask
  draw    the stage-1 mask where the probability is at least 0.48 (predict.POST_STACK; needs the member maps)
  refine  refiner prediction of the seed's own fold (pred_f<k>.npz of analysis/refine_train.py), blended 50/50
          with the dense probability, thresholded at 0.5, joined with absorb 0.3 (src/predict_final.FINAL)
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'src'))
import refine
from common import PQ
from postproc import score_file, upsample
from predict_final import FINAL

GT_INDEX = WORK / 'scratch/bank/gt_index.pkl'      # written by analysis/tree_bank.py from the TRAIN annotations
TAUS = (0.20, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28)


def pq_of(a):
    return float(a[3] / (a[0] + .5 * a[1] + .5 * a[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', type=Path, required=True)
    ap.add_argument('--bank', type=Path, required=True)
    ap.add_argument('--shape', choices=['stage1', 'draw', 'refine'], default='stage1')
    ap.add_argument('--pred', type=Path, help='refiner predictions pred_f<k>.npz (shape refine)')
    ap.add_argument('--members', help='JSON list of two {fold: run folder} maps (shape draw)')
    ap.add_argument('--out', type=Path, help='write the per-image parts here (json)')
    a = ap.parse_args()
    with np.load(a.scores, allow_pickle=False) as z:
        sfile, slabel, sfold, sscore = z['fname'].astype(str), z['label'], z['fold'], z['score']
    gt = pickle.loads(GT_INDEX.read_bytes())
    n_pix = refine.IMAGE_SHAPE[0] * refine.IMAGE_SHAPE[1]
    members = json.loads(a.members) if a.members else None
    rows, t0 = [], time.time()
    keys = ['rule'] + [f'{t:.2f}' for t in TAUS]
    for k in range(5):
        bank = refine.load_bank(a.bank, ['fname', 'label', 'kept', 'M', 'pix', 'seed'] + (['prob'] if a.shape == 'refine' else []), k)
        names = bank['fname'].astype(str)
        assert np.array_equal(sfile[sfold == k], names) and np.array_equal(slabel[sfold == k], bank['label']), 'scores and bank list different seeds'
        score = sscore[sfold == k]
        crops = maps = None
        if a.shape == 'refine':
            with np.load(a.pred / f'pred_f{k}.npz', allow_pickle=False) as z:
                assert np.array_equal(z['fname'], bank['fname']) and np.array_equal(z['label'], bank['label'])
                w = FINAL['blend']
                crops = np.round(w * z['pred'].astype(np.float32) + (1 - w) * bank['prob'].astype(np.float32)).astype(np.uint8)
        if a.shape == 'draw':
            maps, mfiles = [], None
            for m in members:
                with np.load(WORK / 'runs' / m[str(k)] / f'probs_f{k}.npz', allow_pickle=False) as z:
                    mfiles = [str(x) for x in z['val_files']]
                    maps.append(z['val_prob'])
            mrow = {f: i for i, f in enumerate(mfiles)}
        for f in refine.load_images(a.bank, k):
            sel = np.flatnonzero(names == f)
            recs = [rec for _, rec in gt[f]]
            prob = None
            if a.shape == 'draw':
                prob = np.mean([upsample(m[mrow[f]]).astype(np.float32) for m in maps], 0)
            parts = {}
            for key, mask in zip(keys, [bank['kept'][sel]] + [score[sel] > t for t in TAUS]):
                idx = sel[mask]
                if a.shape == 'refine':
                    items = [dict(pix=bank['pix'][i], M=bank['M'][i], seed=bank['seed'][i], refined=crops[i]) for i in idx]
                    lab, _, _ = refine.paste_image(refine.IMAGE_SHAPE, items, FINAL['t_r'], absorb=FINAL['absorb'])
                else:
                    lab = np.zeros(n_pix, np.int32)
                    for n, i in enumerate(idx, 1):
                        lab[bank['pix'][i]] = n
                    lab = lab.reshape(refine.IMAGE_SHAPE)
                    if prob is not None:
                        lab = np.where(prob >= FINAL['draw'] * 255, lab, 0)
                pq = score_file(lab, recs, PQ())
                parts[key] = [pq.tp, pq.fp, pq.fn, pq.s]
            rows.append(dict(file=f, fold=k, records=len(recs), parts=parts))
        print(json.dumps(dict(fold=k, secs=round(time.time() - t0))), flush=True)
    if a.out:
        a.out.write_text(json.dumps(rows))
    P = {key: np.array([r['parts'][key] for r in rows], float) for key in keys}
    fold = np.array([r['fold'] for r in rows])
    recs = np.array([r['records'] for r in rows])
    for key in keys:
        t = P[key].sum(0)
        print(f'{key:>5s}: PQ {pq_of(t):.4f}  TP {int(t[0])} FP {int(t[1])} FN {int(t[2])}', flush=True)
    nested, chosen = np.zeros_like(P['rule']), []
    for k in range(5):
        best = max(keys[1:], key=lambda key: pq_of(P[key][fold != k].sum(0)))
        chosen.append(best)
        nested[fold == k] = P[best][fold == k]
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(rows), (2000, len(rows)))
    diff = np.array([pq_of(nested[d].sum(0)) - pq_of(P['rule'][d].sum(0)) for d in draws])
    per_fold = [round(pq_of(nested[fold == k].sum(0)), 4) for k in range(5)]
    rule_fold = [round(pq_of(P['rule'][fold == k].sum(0)), 4) for k in range(5)]
    print(json.dumps(dict(
        shape=a.shape, nested=round(pq_of(nested.sum(0)), 4), fixed_rule=round(pq_of(P['rule'].sum(0)), 4),
        gain=round(pq_of(nested.sum(0)) - pq_of(P['rule'].sum(0)), 4), taus=chosen, per_fold=per_fold, fixed_rule_per_fold=rule_fold,
        folds_better=int(sum(x > y for x, y in zip(per_fold, rule_fold))),
        by_records={int(r): [round(pq_of(nested[recs == r].sum(0)), 4), round(pq_of(P['rule'][recs == r].sum(0)), 4)] for r in (1, 2, 3)},
        boot_5_95=[round(float(np.percentile(diff, 5)), 4), round(float(np.percentile(diff, 95)), 4)],
        boot_share_positive=round(float((diff > 0).mean()), 3)), indent=1))


if __name__ == '__main__':
    main()
