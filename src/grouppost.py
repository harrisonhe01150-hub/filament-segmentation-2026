"""Instance extraction guided by the learned grouping channel (v7), with exact cached search.

Instances = connected components of (group_prob >= t_g); every mask pixel (prob >= t_lo) takes
the label of the nearest group component within max_d px, otherwise its own connected component.
"""
import sys, json, itertools, time
from multiprocessing import Pool, get_start_method
import numpy as np
import cv2
from scipy import ndimage as ndi
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from postproc import gt_index, upsample
from common import PQ

PROBS = GROUPS = FILES = GT = None


def load(paths):
    p, g, fl = [], [], []
    for path in paths:
        z = np.load(path)
        p.append(z['val_prob']); g.append(z['val_group']); fl += list(z['val_files'])
    return np.concatenate(p), np.concatenate(g), fl


def grouped_labels(prob, group, t_lo, t_g, max_d):
    mask = prob >= t_lo * 255
    glab, ng = ndi.label(group >= t_g * 255, np.ones((3, 3)))
    if ng:
        dist, (iy, ix) = ndi.distance_transform_edt(glab == 0, return_indices=True)
        near = glab[iy, ix]
        near[dist > max_d] = 0
    else:
        near = np.zeros_like(glab)
    lab = np.where(mask, near, 0).astype(np.int32)
    rest, nr = ndi.label(mask & (lab == 0), np.ones((3, 3)))
    lab[rest > 0] = rest[rest > 0] + ng
    n = ng + nr
    # contiguous ids
    u = np.unique(lab); u = u[u > 0]
    remap = np.zeros(n + 1, np.int32); remap[u] = np.arange(1, len(u) + 1)
    lab = remap[lab]
    k = len(u)
    idx = np.arange(1, k + 1)
    area = np.bincount(lab.ravel(), minlength=k + 1)[1:]
    mean = np.nan_to_num(ndi.mean(prob, lab, idx)) / 255 if k else np.zeros(0)
    return lab, np.stack([area, mean], 1).astype(np.float32) if k else np.zeros((0, 2), np.float32)


def table(args):
    i, t_lo, t_g, max_d = args
    lab, feats = grouped_labels(upsample(PROBS[i]), upsample(GROUPS[i]), t_lo, t_g, max_d)
    flat = lab.ravel(); k = len(feats)
    areas = np.bincount(flat, minlength=k + 1)
    recs = []
    for rec in GT[FILES[i]]:
        iou = np.zeros((len(rec), k), np.float32)
        for gi, ix in enumerate(rec):
            inter = np.bincount(flat[ix], minlength=k + 1)
            union = areas + len(ix) - inter
            iou[gi] = np.where(inter[1:] > 0, inter[1:] / np.maximum(union[1:], 1), 0)
        recs.append(iou)
    return feats, recs


if __name__ == '__main__':
    spec = json.loads(sys.argv[1])
    PROBS, GROUPS, FILES = load(spec['npz'])
    GT = gt_index()
    res = []
    for t_lo, t_g, max_d in itertools.product(spec['t_lo'], spec['t_g'], spec['max_d']):
        t = time.time()
        tasks = [(i, t_lo, t_g, max_d) for i in range(len(FILES))]
        if get_start_method() != 'fork':
            tabs = [table(task) for task in tasks]
        else:
            with Pool(2) as pool:
                tabs = pool.map(table, tasks, chunksize=4)
        for ma, mm in itertools.product(spec['min_area'], spec['min_mean']):
            pq = PQ()
            for feats, recs in tabs:
                keep = (feats[:, 0] >= ma) & (feats[:, 1] >= mm)
                for r in recs:
                    pq.add(r[:, keep])
            res.append((dict(t_lo=t_lo, t_g=t_g, max_d=max_d, min_area=ma, min_mean=mm), pq))
        print('built', t_lo, t_g, max_d, round(time.time() - t), flush=True)
    res.sort(key=lambda r: -r[1].value)
    print('TOP')
    for c, pq in res[:10]:
        print(json.dumps(c), pq)
