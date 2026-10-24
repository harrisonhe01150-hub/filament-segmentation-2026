"""Shared data loading and official-style PQ metric for the Solar filament competition."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

import os
ROOT = Path(os.environ.get('SOLAR_DATA', 'data')) / 'MAGFiLO_1.0_Kaggle_2026'
ANN = ROOT / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
H = W = 2048


def _parse(v):
    return json.loads(v) if isinstance(v, str) else v


def load_coco(path=ANN):
    d = json.loads(Path(path).read_text())
    images = d['images']
    anns = defaultdict(list)
    for a in d['annotations']:
        a = dict(a)
        a['segmentation'] = _parse(a['segmentation'])
        a['bbox'] = _parse(a['bbox'])
        a['category_id'] = int(a['category_id'])
        anns[a['image_id']].append(a)
    by_file = defaultdict(list)
    for im in images:
        by_file[im['file_name']].append(im['id'])
    return images, anns, by_file


def ann_mask(a, h=H, w=W):
    seg = a['segmentation']
    if isinstance(seg, list) and seg and not isinstance(seg[0], list):
        seg = [seg]
    rle = mask_util.merge(mask_util.frPyObjects(seg, h, w))
    return mask_util.decode(rle).astype(bool)


def record_masks(anns_of_record):
    if not anns_of_record:
        return np.zeros((0, H, W), bool)
    return np.stack([ann_mask(a) for a in anns_of_record])


def iou_matrix(gt, pred):
    """gt: (n,H,W) bool, pred: (m,H,W) bool -> (n,m) IoU, computed on bbox-cropped union for speed."""
    n, m = len(gt), len(pred)
    out = np.zeros((n, m), np.float32)
    if n == 0 or m == 0:
        return out
    g = gt.reshape(n, -1)
    p = pred.reshape(m, -1)
    # sparse-ish: use index lists
    gi = [np.flatnonzero(x) for x in g]
    pi = [np.flatnonzero(x) for x in p]
    for i in range(n):
        if len(gi[i]) == 0:
            continue
        for j in range(m):
            if len(pi[j]) == 0:
                continue
            inter = np.intersect1d(gi[i], pi[j], assume_unique=True).size
            if inter:
                out[i, j] = inter / (len(gi[i]) + len(pi[j]) - inter)
    return out


def iou_from_labels(gt_masks, labels):
    """IoU between GT stack and a prediction label map (0=bg, 1..K instances)."""
    k = int(labels.max())
    out = np.zeros((len(gt_masks), k), np.float32)
    if k == 0:
        return out
    areas = np.bincount(labels.ravel(), minlength=k + 1)
    for i, g in enumerate(gt_masks):
        inter = np.bincount(labels[g], minlength=k + 1)
        union = areas + g.sum() - inter
        out[i] = (inter / np.maximum(union, 1))[1:]
        out[i][inter[1:] == 0] = 0
    return out


class PQ:
    """Official self-evaluation notebook rules: all pairs with IoU>0.5 are TP."""

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.s = 0.0

    def add(self, iou, n_gt=None, n_pred=None):
        n_gt = iou.shape[0] if n_gt is None else n_gt
        n_pred = iou.shape[1] if n_pred is None else n_pred
        if n_gt == 0:
            self.fp += n_pred
            return
        if n_pred == 0:
            self.fn += n_gt
            return
        hit = iou > .5
        self.tp += int(hit.sum())
        self.s += float(iou[hit].sum())
        self.fp += int((hit.sum(0) == 0).sum())
        self.fn += int((hit.sum(1) == 0).sum())

    def merge(self, other):
        self.tp += other.tp; self.fp += other.fp; self.fn += other.fn; self.s += other.s
        return self

    @property
    def value(self):
        d = self.tp + .5 * self.fp + .5 * self.fn
        return self.s / d if d else 0.0

    def __repr__(self):
        mi = self.s / self.tp if self.tp else 0
        return f'PQ={self.value:.4f} TP={self.tp} FP={self.fp} FN={self.fn} meanIoU={mi:.3f}'
