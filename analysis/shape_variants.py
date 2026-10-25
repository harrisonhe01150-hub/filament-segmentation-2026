"""Five-fold OOF PQ of mask-shape post-processing variants on the handoff's v2 probabilities (runs/v2, runs/v2b).

Instances come from the v2 post-processing; each variant then reshapes every instance inside its
bounding box (growing only into background, so masks stay disjoint).
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / 'src'))
from common import PQ, load_coco, ann_mask
from postproc import upsample, score_file
from fastsweep import components


def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def reshape(lab, op, pad=6):
    out = np.zeros_like(lab)
    for n, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, lab.shape[0])
        x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, lab.shape[1])
        m = (lab[y0:y1, x0:x1] == n).astype(np.uint8)
        m2 = op(m).astype(bool)
        region = out[y0:y1, x0:x1]
        others = (lab[y0:y1, x0:x1] > 0) & (lab[y0:y1, x0:x1] != n)
        region[m2 & ~others & (region == 0)] = n
    return out


VARIANTS = {
    'baseline': None,
    'fill_holes': lambda m: ndi.binary_fill_holes(m),
    'close2_fill': lambda m: ndi.binary_fill_holes(cv2.morphologyEx(m, cv2.MORPH_CLOSE, disk(2))),
    'close4_fill': lambda m: ndi.binary_fill_holes(cv2.morphologyEx(m, cv2.MORPH_CLOSE, disk(4))),
    'dilate1': lambda m: cv2.dilate(m, disk(1)),
    'erode1': lambda m: cv2.erode(m, disk(1)),
    'open1': lambda m: cv2.morphologyEx(m, cv2.MORPH_OPEN, disk(1)),
}

images, anns, by_file = load_coco()
pqs = {k: PQ() for k in VARIANTS}
per_fold = {k: {} for k in VARIANTS}
t0 = time.time()
for fold in range(5):
    path = repo / 'runs' / ('v2' if fold < 2 else 'v2b') / f'probs_f{fold}.npz'
    with np.load(path) as z:
        files, prob = [str(x) for x in z['val_files']], z['val_prob']
    fold_pq = {k: PQ() for k in VARIANTS}
    for f, p in zip(files, prob):
        lab, feat = components(upsample(p), .4, 8)
        keep = (feat[:, 0] >= 300) & (feat[:, 1] >= .85)
        remap = np.zeros(len(feat) + 1, np.int32); remap[1:][keep] = np.arange(1, int(keep.sum()) + 1)
        base = remap[lab]
        recs = [[np.flatnonzero(ann_mask(a)).astype(np.int32) for a in anns[r]] for r in by_file[f]]
        for k, op in VARIANTS.items():
            l2 = base if op is None else reshape(base, op)
            s = score_file(l2, recs, PQ())
            fold_pq[k].merge(s); pqs[k].merge(s)
    for k in VARIANTS:
        per_fold[k][fold] = round(fold_pq[k].value, 4)
    print(json.dumps(dict(fold=fold, secs=round(time.time() - t0), **{k: per_fold[k][fold] for k in VARIANTS})), flush=True)
base = pqs['baseline'].value
print(json.dumps({k: dict(pq=round(v.value, 4), delta=round(v.value - base, 4), tp=v.tp, fp=v.fp, fn=v.fn,
                          mean_iou=round(v.s / v.tp, 4),
                          folds_better=sum(per_fold[k][f] > per_fold['baseline'][f] for f in range(5)))
                  for k, v in pqs.items()}, indent=1))
