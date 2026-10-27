"""Experiment C scoring: does a predicted boundary channel help separate filaments?

For the boundary run, scores the mask channel with the plain v2 post-processing, then with boundary cuts:
the v2 grouping region (prob >= .4, closing) minus pixels whose boundary prob >= t_b is labelled, and every
grouping pixel within MAX_D px of a core takes its nearest core's label, so a cut line splits touching
filaments without shaving their edges. Final masks are threshold pixels only, filtered like v2
(area >= 300, mean >= .85). The cut setting is picked on one fold and applied to the other (nested).
An optional control run (same seed/machine, no boundary head) is scored with plain v2 post-processing.

usage: SOLAR_DATA=<data dir> python analysis/eval_boundary.py <boundary run> [<control run>]
"""
import itertools
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

RUN = Path(sys.argv[1])
CONTROL = Path(sys.argv[2]) if len(sys.argv) > 2 else None
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import ANN, PQ, ann_mask, load_coco  # noqa: E402
from postproc import score_file, upsample  # noqa: E402

T_LO, MIN_AREA, MIN_MEAN, MAX_D = .4, 300, .85, 10
CUTS = [None] + list(itertools.product([0, 8], [.2, .3, .4, .5, .6]))  # None = plain v2; else (closing, t_b)


def instances(prob, bnd, cut):
    lo = prob >= T_LO * 255
    close, t_b = (8, None) if cut is None else cut
    grp = lo
    if close:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close + 1, 2 * close + 1))
        grp = cv2.morphologyEx(lo.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    core = grp if t_b is None else grp & (bnd < t_b * 255)
    lab, n = ndi.label(core, np.ones((3, 3)))
    if n and t_b is not None:
        d, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
        lab = np.where(grp & (d <= MAX_D), lab[iy, ix], 0)
    lab = lab * lo
    if n == 0:
        return lab
    idx = np.arange(1, n + 1)
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    mean = np.nan_to_num(ndi.mean(prob, lab, idx)) / 255
    keep = (area >= MIN_AREA) & (mean >= MIN_MEAN)
    remap = np.zeros(n + 1, np.int32)
    remap[1:][keep] = np.arange(1, keep.sum() + 1)
    return remap[lab]


def score_image(args):
    prob, bnd, recs, cuts = args
    prob = upsample(prob)
    bnd = upsample(bnd) if bnd is not None else None
    out = []
    for cut in cuts:
        pq = score_file(instances(prob, bnd, cut), recs, PQ())
        out.append((pq.s, pq.tp, pq.fp, pq.fn))
    return out


def pq_of(rows):
    s, tp, fp, fn = np.sum(rows, 0)
    return s / (tp + .5 * fp + .5 * fn), int(tp), int(fp), int(fn)


def run_scores(run, cuts, with_boundary):
    """{fold: array (images, len(cuts), 4)} for folds present in the run."""
    _, anns, by_file = load_coco(ANN)
    out = {}
    for p in sorted(run.glob('probs_f*.npz')):
        k = int(p.stem[-1])
        z = np.load(p)
        files = [str(f) for f in z['val_files']]
        bnds = z['val_boundary'] if with_boundary else [None] * len(files)
        tasks = [(z['val_prob'][i], bnds[i], [[np.flatnonzero(ann_mask(a)).astype(np.int32) for a in anns[r]]
                                              for r in by_file[f]], cuts) for i, f in enumerate(files)]
        with get_context('fork').Pool(max(1, os.cpu_count() - 4)) as pool:
            out[k] = np.array(pool.map(score_image, tasks, chunksize=2))
    return out


if __name__ == '__main__':
    if CONTROL:
        ctrl = run_scores(CONTROL, [None], False)
        for k in sorted(ctrl):
            print(f'control      fold {k}: PQ {pq_of(ctrl[k][:, 0])[0]:.4f}')
        print(f'control      all   : PQ {pq_of(np.concatenate([ctrl[k][:, 0] for k in ctrl]))[0]:.4f}  TP/FP/FN '
              f'{pq_of(np.concatenate([ctrl[k][:, 0] for k in ctrl]))[1:]}')
    sc = run_scores(RUN, CUTS, True)
    folds = sorted(sc)
    for k in folds:
        best = max(range(len(CUTS)), key=lambda c: pq_of(sc[k][:, c])[0])
        print(f'boundary run fold {k}: mask only (v2 pp) {pq_of(sc[k][:, 0])[0]:.4f} | best cut {CUTS[best]} '
              f'{pq_of(sc[k][:, best])[0]:.4f}')
    print('boundary run all   : mask only (v2 pp) '
          f'{pq_of(np.concatenate([sc[k][:, 0] for k in folds]))[0]:.4f}')
    if len(folds) > 1:  # pick the cut on the other folds, apply to this one
        nested = []
        for k in folds:
            rest = np.concatenate([sc[j] for j in folds if j != k])
            c = max(range(len(CUTS)), key=lambda c: pq_of(rest[:, c])[0])
            nested.append(sc[k][:, c])
            print(f'nested fold {k}: cut {CUTS[c]} chosen on other folds -> PQ {pq_of(sc[k][:, c])[0]:.4f}')
        print(f'nested all   : PQ {pq_of(np.concatenate(nested))[0]:.4f}  TP/FP/FN {pq_of(np.concatenate(nested))[1:]}')
