"""Out-of-fold predictions of the final pipeline on all 707 training images, as small files for the report.

Writes to --out:
  oof_full.csv        full pipeline (learned keep rule + refiner outlines), submission format
                      (filament_id = <image stem>_<n>, segmentation_rle = COCO RLE counts of the 2048 x 2048 mask)
  oof_light.csv       light variant (learned keep rule, stage-1 mask drawn where the probability is at least 0.48)
  oof_images.csv      one row per image: fname, fold, records (annotator records), tau, candidates, kept, and
                      TP / FP / FN / IoU sum of both variants under the organisers' rule (postproc.score_file)
  oof_candidates.csv  one row per candidate (stage-1 seed of at least 100 px, 6788 in all): fname, fold, label, area,
                      mean (mean probability), u (cross-fitted keep score), tau, keep (u > tau), fixed_rule (area >= 300
                      and mean >= 0.85, the v2 rule), iou_r1..iou_r3 (best IoU of its stage-1 mask with a filament of
                      annotator record 1..3; empty when the image has fewer records), rle (its stage-1 mask)
  SUMMARY.json        pooled and per-fold PQ of both variants and the settings

Everything is out of fold: the keep model of fold k is trained on the other four folds, its threshold tau is the
one an inner cross-fit picked without fold k (keep_fit.py "strict_taus"), the refiner of fold k never saw fold k.
The pooled PQ must equal EXPERIMENTS.md (K1 strict nesting): full 0.4566, light 0.4537.

  python analysis/export_oof_final.py --scores scratch/keep/oof/scores.npz --bank scratch/ws/main/rbank/kc_bridge
      --pred scratch/ws/main/kcb/pred --fit scratch/keep/oof/fit_result.json --out results/oof_final
      --members <JSON list of two {fold: run folder} maps, as for analysis/keep_bank.py>
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'src'))
sys.path.insert(0, str(WORK / 'analysis'))
from tree_bank import gt_index
import refine
from common import PQ
from postproc import score_file, upsample
from predict_final import FINAL

GT_INDEX = WORK / 'scratch/bank/gt_index.pkl'      # TRAIN annotations as pixel lists; built on first use
VARIANTS = ('full', 'light')


def rle(mask):
    return mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))['counts'].decode()


def pq_of(t):
    return float(t[3] / (t[0] + .5 * t[1] + .5 * t[2]))


def instance_rows(lab, n_max, stem):
    rows, n = [], 0
    for j in range(1, n_max + 1):
        m = lab == j
        if m.any():
            n += 1
            rows.append((f'{stem}_{n}', rle(m)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', type=Path, required=True)
    ap.add_argument('--bank', type=Path, required=True)
    ap.add_argument('--pred', type=Path, required=True)
    ap.add_argument('--fit', type=Path, required=True)
    ap.add_argument('--members', required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    taus = json.loads(a.fit.read_text())['strict_taus']
    members = json.loads(a.members)
    with np.load(a.scores, allow_pickle=False) as z:
        sfile, slabel, sfold, sscore = z['fname'].astype(str), z['label'], z['fold'], z['score']
    gt = gt_index(GT_INDEX)
    shape = refine.IMAGE_SHAPE
    n_pix = shape[0] * shape[1]
    a.out.mkdir(parents=True, exist_ok=True)
    rows = {v: [] for v in VARIANTS}
    images, cands = [], []
    t0 = time.time()
    for k in range(5):
        tau = float(taus[k])
        bank = refine.load_bank(a.bank, ['fname', 'label', 'kept', 'area', 'mean', 'best_iou', 'M', 'pix', 'seed', 'prob'], k)
        names = bank['fname'].astype(str)
        assert np.array_equal(sfile[sfold == k], names) and np.array_equal(slabel[sfold == k], bank['label']), 'scores and bank list different seeds'
        score = sscore[sfold == k]
        with np.load(a.pred / f'pred_f{k}.npz', allow_pickle=False) as z:
            assert np.array_equal(z['fname'], bank['fname']) and np.array_equal(z['label'], bank['label'])
            w = FINAL['blend']
            crops = np.round(w * z['pred'].astype(np.float32) + (1 - w) * bank['prob'].astype(np.float32)).astype(np.uint8)
        maps = []
        for m in members:
            with np.load(WORK / 'runs' / m[str(k)] / f'probs_f{k}.npz', allow_pickle=False) as z:
                maps.append(({str(x): i for i, x in enumerate(z['val_files'])}, z['val_prob']))
        for f in refine.load_images(a.bank, k):
            sel = np.flatnonzero(names == f)
            recs = [rec for _, rec in gt[f]]
            idx = sel[score[sel] > tau]
            stem = f.rsplit('.', 1)[0]
            parts = {}
            # full: refined outline of every kept candidate (src/predict_final.py, FINAL settings)
            items = [dict(pix=bank['pix'][i], M=bank['M'][i], seed=bank['seed'][i], refined=crops[i]) for i in idx]
            lab, areas, _ = refine.paste_image(shape, items, FINAL['t_r'], absorb=FINAL['absorb'])
            pq = score_file(lab, recs, PQ())
            parts['full'] = [pq.tp, pq.fp, pq.fn, pq.s]
            rows['full'] += instance_rows(lab, len(areas), stem)
            # light: stage-1 mask of every kept candidate, drawn where the mean member probability >= 0.48
            prob = np.mean([upsample(p[r[f]]).astype(np.float32) for r, p in maps], 0)
            flat = np.zeros(n_pix, np.int32)
            for n, i in enumerate(idx, 1):
                flat[bank['pix'][i]] = n
            lab = np.where(prob >= FINAL['draw'] * 255, flat.reshape(shape), 0)
            pq = score_file(lab, recs, PQ())
            parts['light'] = [pq.tp, pq.fp, pq.fn, pq.s]
            rows['light'] += instance_rows(lab, len(idx), stem)
            row = dict(fname=f, fold=k, records=len(recs), tau=tau, candidates=len(sel), kept=len(idx))
            for v in VARIANTS:
                row.update({f'{v}_tp': int(parts[v][0]), f'{v}_fp': int(parts[v][1]), f'{v}_fn': int(parts[v][2]),
                            f'{v}_iou_sum': round(float(parts[v][3]), 6)})
            images.append(row)
            for i in sel:
                m = np.zeros(n_pix, np.uint8)
                m[bank['pix'][i]] = 1
                c = dict(fname=f, fold=k, label=int(bank['label'][i]), area=int(bank['area'][i]),
                         mean=round(float(bank['mean'][i]), 4), u=round(float(score[i]), 4), tau=tau,
                         keep=int(score[i] > tau), fixed_rule=int(bank['kept'][i]))
                for r in range(3):
                    c[f'iou_r{r + 1}'] = '' if r >= len(recs) else round(float(bank['best_iou'][i][r]), 4)
                c['rle'] = rle(m.reshape(shape))
                cands.append(c)
        print(json.dumps(dict(fold=k, images=len(images), secs=round(time.time() - t0))), flush=True)
    for v in VARIANTS:
        with open(a.out / f'oof_{v}.csv', 'w', newline='') as fh:
            fh.write('filament_id,segmentation_rle\n')
            for fid, counts in rows[v]:
                fh.write(f'{fid},{counts}\n')
    for name, table in (('oof_images.csv', images), ('oof_candidates.csv', cands)):
        with open(a.out / name, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(table[0]))
            wr.writeheader()
            wr.writerows(table)
    fold = np.array([r['fold'] for r in images])
    summary = dict(images=len(images), candidates=len(cands), taus=taus, settings=FINAL)
    for v in VARIANTS:
        P = np.array([[r[f'{v}_tp'], r[f'{v}_fp'], r[f'{v}_fn'], r[f'{v}_iou_sum']] for r in images], float)
        t = P.sum(0)
        summary[v] = dict(pq=round(pq_of(t), 4), tp=int(t[0]), fp=int(t[1]), fn=int(t[2]), instances=len(rows[v]),
                          per_fold=[round(pq_of(P[fold == k].sum(0)), 4) for k in range(5)])
    (a.out / 'SUMMARY.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
