"""Score one held-out fold the way the organizers' self-evaluation notebook does, and compare with ours.

Builds the ground-truth table (`<annotator>-<image>_<n>`, COCO RLE) and our prediction table
(`<image>_<n>`, same export as the submission) for the fold, then follows the official notebook step by step:
per annotator-image IoU/Dice matrices from RLE layers, pairs with IoU > 0.5 are TP, FP = predictions with no
hit, FN = filaments with no hit. Also reports the notebook's other rubric numbers (IoU/Dice of overlapping
pairs, 1:n and n:1 counts).

  python analysis/official_pq_check.py --run runs/v2_5070ti --fold 0
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mu

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'src'))
from common import ANN, PQ, ann_mask, load_coco
from make_submission import build_rows
from postproc import score_file
from predict import POST


def layers(rles):
    if not rles:
        return torch.zeros((0, 2048, 2048))
    m = mu.decode([{'size': [2048, 2048], 'counts': r.encode()} for r in rles])
    return torch.from_numpy(m.transpose(2, 0, 1).astype(np.float32))


def overlap(gt, pred):
    n, m = len(gt), len(pred)
    g, p = gt.reshape(n, -1), pred.reshape(m, -1)
    inter = g @ p.t()
    ga, pa = g.sum(1).view(-1, 1), p.sum(1).view(1, -1)
    union = ga + pa - inter
    iou = torch.where(union == 0, torch.tensor(0.), inter / union)
    dice = torch.where(union == 0, torch.tensor(0.), 2 * inter / (ga + pa))
    return iou, dice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--annotations', type=Path, default=WORK / ANN)
    a = ap.parse_args()
    images, anns, by_file = load_coco(a.annotations)
    with np.load(a.run / f'probs_f{a.fold}.npz', allow_pickle=False) as z:
        files, prob = [str(x) for x in z['val_files']], z['val_prob']
    pred_rows, _ = build_rows(files, [prob], POST)
    pred_by_image = {}
    for fid, counts in pred_rows:
        pred_by_image.setdefault(fid.split('_', 1)[0], []).append(counts)

    tp_iou, fp, fn = [], 0, 0
    pair_iou, pair_dice, gt_deg, pred_deg = [], [], Counter(), Counter()
    ours = PQ()
    for f in files:
        stem = f.rsplit('.', 1)[0]
        pr = layers(pred_by_image.get(stem, []))
        for rec in by_file[f]:
            gt_rles = [mu.encode(np.asfortranarray(ann_mask(an).astype(np.uint8)))['counts'].decode() for an in anns[rec]]
            gt = layers(gt_rles)
            n_gt, n_pred = len(gt), len(pr)
            if n_gt == 0:
                fp += n_pred; pred_deg[0] += n_pred; continue
            if n_pred == 0:
                fn += n_gt; gt_deg[0] += n_gt; continue
            iou, dice = overlap(gt, pr)
            hit = iou > .5
            tp_iou += iou[hit].tolist()
            fp += int((hit.sum(0) == 0).sum()); fn += int((hit.sum(1) == 0).sum())
            pair_iou += iou[iou > 0].tolist(); pair_dice += dice[dice > 0].tolist()
            touch = iou > 0
            gt_deg.update(touch.sum(1).tolist()); pred_deg.update(touch.sum(0).tolist())
        # our scorer on the same predictions (label map rebuilt from the same RLEs)
        labels = np.zeros(2048 * 2048, np.int32)
        for k, layer in enumerate(pr, 1):
            labels[layer.numpy().ravel() > 0] = k
        recs = [[np.flatnonzero(ann_mask(an)).astype(np.int32) for an in anns[r]] for r in by_file[f]]
        score_file(labels.reshape(2048, 2048), recs, ours)

    official = sum(tp_iou) / (len(tp_iou) + .5 * fp + .5 * fn)
    out = dict(run=str(a.run), fold=a.fold, images=len(files), official_pq=round(official, 6), our_pq=round(ours.value, 6),
               identical=abs(official - ours.value) < 1e-6, tp=len(tp_iou), fp=fp, fn=fn,
               ours_tp_fp_fn=[ours.tp, ours.fp, ours.fn],
               mean_iou_overlapping_pairs=round(float(np.mean(pair_iou)), 4), mean_dice_overlapping_pairs=round(float(np.mean(pair_dice)), 4),
               gt_split_into={str(k): v for k, v in sorted(gt_deg.items())},
               pred_covers_gt={str(k): v for k, v in sorted(pred_deg.items())})
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
