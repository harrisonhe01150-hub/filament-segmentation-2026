"""Instance extraction from probability maps + official-rule PQ on OOF predictions."""
import sys
import pickle
import hashlib
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage as ndi
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from common import load_coco, ann_mask, PQ, ANN

GT_CACHE = Path(__import__('os').environ.get('SOLAR_CACHE', '.')) / 'gt_cache.pkl'


def gt_index():
    """file -> list of records -> list of flat index arrays (2048 grid)."""
    annotation_sha = hashlib.sha256(ANN.read_bytes()).hexdigest()
    # Separate caches by exact annotation bytes and cache schema; preserve old files.
    cache = GT_CACHE.with_name(f'{GT_CACHE.stem}_{annotation_sha}_v1.pkl')
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    images, anns, by_file = load_coco()
    out = {}
    for f, recs in by_file.items():
        out[f] = [[np.flatnonzero(ann_mask(a)).astype(np.int32) for a in anns[r]] for r in recs]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(out))
    return out


def upsample(prob_u8):
    return cv2.resize(prob_u8, (2048, 2048), interpolation=cv2.INTER_LINEAR)


def instances(prob, t_hi=.5, t_lo=None, min_area=0, close=0, min_peak=0., min_mean=0.):
    """prob: float/uint8 2048x2048 in [0,255]. Returns int32 label map.

    t_hi: seed threshold; t_lo: growth threshold (hysteresis; None=t_hi);
    close: radius of morphological closing used only to *group* nearby fragments;
    min_area / min_peak / min_mean: component filters.
    """
    hi = prob >= t_hi * 255
    lo = prob >= (t_lo if t_lo is not None else t_hi) * 255
    grp = lo
    if close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close + 1, 2 * close + 1))
        grp = cv2.morphologyEx(lo.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    lab, n = ndi.label(grp, np.ones((3, 3)))
    lab = lab * lo
    if n == 0:
        return lab.astype(np.int32)
    idx = np.arange(1, n + 1)
    area = ndi.sum(np.ones_like(lab), lab, idx)
    has_seed = ndi.maximum(hi, lab, idx) > 0
    keep = (area >= max(min_area, 1)) & has_seed
    if min_peak > 0:
        keep &= ndi.maximum(prob, lab, idx) >= min_peak * 255
    if min_mean > 0:
        keep &= ndi.mean(prob, lab, idx) >= min_mean * 255
    remap = np.zeros(n + 1, np.int32)
    remap[idx[keep]] = np.arange(1, keep.sum() + 1)
    return remap[lab]


def score_file(labels, recs, pq):
    flat = labels.ravel()
    k = int(flat.max())
    areas = np.bincount(flat, minlength=k + 1)
    for rec in recs:
        n = len(rec)
        if n == 0:
            pq.add(np.zeros((0, k)), 0, k); continue
        iou = np.zeros((n, k), np.float32)
        for i, ix in enumerate(rec):
            inter = np.bincount(flat[ix], minlength=k + 1)
            union = areas + len(ix) - inter
            iou[i] = (inter / np.maximum(union, 1))[1:]
            iou[i][inter[1:] == 0] = 0
        pq.add(iou, n, k)
    return pq


def evaluate(probs, files, gt, **kw):
    pq = PQ()
    for p, f in zip(probs, files):
        score_file(instances(upsample(p), **kw), gt[f], pq)
    return pq
