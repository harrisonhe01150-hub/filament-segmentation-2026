"""Recall (IoU > 0.5) by annotated filament size for runs/<name>/probs_f0,f1.npz, with and without the v2 filters.

  python analysis/recall_by_size.py v2_5070ti B1_5070ti
"""
import json
import sys
from pathlib import Path

import numpy as np

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / 'src'))
from common import load_coco, ann_mask
from postproc import upsample
from fastsweep import components

BINS = [0, 300, 600, 1000, 2000, 10 ** 9]
images, anns, by_file = load_coco()
result = {}
for run in sys.argv[1:]:
    hits = {k: np.zeros(len(BINS) - 1, int) for k in ('filtered', 'unfiltered')}
    total = np.zeros(len(BINS) - 1, int)
    preds = dict(filtered=0, unfiltered=0, small_kept=0)
    for fold in (0, 1):
        with np.load(repo / 'runs' / run / f'probs_f{fold}.npz') as z:
            files, prob = [str(x) for x in z['val_files']], z['val_prob']
        for f, p in zip(files, prob):
            lab, feat = components(upsample(p), .4, 8)
            keep = (feat[:, 0] >= 300) & (feat[:, 1] >= .85)
            preds['unfiltered'] += len(feat); preds['filtered'] += int(keep.sum())
            flat = lab.ravel(); k = len(feat)
            areas = np.bincount(flat, minlength=k + 1)
            for r in by_file[f]:
                for a in anns[r]:
                    ix = np.flatnonzero(ann_mask(a))
                    b = np.searchsorted(BINS, len(ix), side='right') - 1
                    total[b] += 1
                    inter = np.bincount(flat[ix], minlength=k + 1)[1:]
                    iou = inter / np.maximum(areas[1:] + len(ix) - inter, 1)
                    hits['unfiltered'][b] += int((iou > .5).any())
                    hits['filtered'][b] += int((iou[keep] > .5).any()) if k else 0
    result[run] = dict(gt_per_bin=total.tolist(), predictions=preds,
                       recall_filtered=[round(h / t, 3) for h, t in zip(hits['filtered'], total)],
                       recall_unfiltered=[round(h / t, 3) for h, t in zip(hits['unfiltered'], total)])
print(json.dumps(dict(bins_px=['<300', '300-600', '600-1000', '1000-2000', '>2000'], **result), indent=1))
