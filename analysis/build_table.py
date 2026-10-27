"""Per-component OOF table for the v2 model.

For each of the 707 training images: connected components of the cached OOF probability map
(t=.4, closing 8, no filters), shape/confidence features per component, and the full IoU matrix
against every expert record. Any keep/drop rule can then be scored exactly without recomputing masks.

usage: SOLAR_DATA=<data dir> python analysis/build_table.py <oof runs dir> <out.pkl>
<oof runs dir> holds v2/probs_f0-1.npz and v2b/probs_f2-4.npz (the handoff package's runs/).
"""
import os
import pickle
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import ANN, ann_mask, load_coco  # noqa: E402
from fastsweep import components  # noqa: E402
from postproc import upsample  # noqa: E402

T_LO, CLOSE = .4, 8
NAMES = ['area', 'mean', 'peak', 'std', 'radius', 'height', 'width', 'fill', 'elong', 'pieces', 'n_img']


def table(args):
    file, fold, prob_u8, recs = args
    prob = upsample(prob_u8).astype(np.float32)
    lab, base = components(prob, T_LO, CLOSE)
    k = len(base)
    idx = np.arange(1, k + 1)
    feats = np.zeros((k, len(NAMES)), np.float32)
    if k:
        feats[:, :3] = base
        feats[:, 3] = ndi.standard_deviation(prob, lab, idx) / 255
        cy, cx = np.array(ndi.center_of_mass(np.ones_like(lab), lab, idx)).T
        feats[:, 4] = np.hypot(cy - 1023.5, cx - 1023.5) / 1024
        for j, sl in enumerate(ndi.find_objects(lab)):
            if sl is None:  # component emptied by `lab * lo`
                continue
            yx = np.argwhere(lab[sl] == j + 1)
            h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            ev = np.linalg.eigvalsh(np.cov(yx.T)) if len(yx) > 2 else np.ones(2)
            feats[j, 5:9] = h, w, len(yx) / (h * w), np.sqrt(max(ev[-1], 1e-6) / max(ev[0], 1e-6))
        # how many separate threshold blobs the closing step glued into this component
        lo_lab, n_lo = ndi.label(lab > 0, np.ones((3, 3)))
        on = lab > 0
        pairs = np.unique(lab[on].astype(np.int64) * (n_lo + 1) + lo_lab[on])
        feats[:, 9] = np.bincount(pairs // (n_lo + 1), minlength=k + 1)[1:]
        feats[:, 10] = k
    flat = lab.ravel()
    areas = np.bincount(flat, minlength=k + 1)
    ious, inters = [], []
    for rec in recs:
        inter = np.zeros((len(rec), k), np.int32)
        for g, ix in enumerate(rec):
            inter[g] = np.bincount(flat[ix], minlength=k + 1)[1:]
        gt_area = np.array([len(ix) for ix in rec], np.int32)[:, None]
        ious.append(np.where(inter > 0, inter / np.maximum(areas[1:] + gt_area - inter, 1), 0).astype(np.float32))
        inters.append(inter)
    return dict(file=file, fold=fold, feats=feats, ious=ious, inters=inters,
                gt_areas=[np.array([len(ix) for ix in rec], np.int32) for rec in recs])


def tasks(oof_runs):
    _, anns, by_file = load_coco(ANN)
    for fold in range(5):
        run = Path(oof_runs) / ('v2' if fold < 2 else 'v2b') / f'probs_f{fold}.npz'
        with np.load(run, allow_pickle=False) as z:
            for f, p in zip(z['val_files'], z['val_prob']):
                f = str(f)
                yield f, fold, p, [[np.flatnonzero(ann_mask(a)).astype(np.int32) for a in anns[r]] for r in by_file[f]]


if __name__ == '__main__':
    OOF, OUT = Path(sys.argv[1]), Path(sys.argv[2])
    with get_context('fork').Pool(max(1, os.cpu_count() - 4)) as pool:
        rows = pool.map(table, tasks(OOF), chunksize=4)
    assert len(rows) == 707 and len({r['file'] for r in rows}) == 707
    OUT.write_bytes(pickle.dumps(dict(names=NAMES, t_lo=T_LO, close=CLOSE, rows=rows)))
    print('images', len(rows), 'components', sum(len(r['feats']) for r in rows))
