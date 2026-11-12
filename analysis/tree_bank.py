"""Build the candidate bank for the tree decision layer (src/treeselect.py).

For every image: average the member models' probability maps, build the component tree, and store each node's
descriptors. For training images (out-of-fold maps) also store each node's best IoU against every annotator record,
which is the training target of the scorer and lets any selection be scored exactly without re-rendering.

Out-of-fold bank (members = one {fold: run folder} map per model):
  python analysis/tree_bank.py '{"tag": "v2x2", "members": [{"0": "v2", "1": "v2", "2": "v2b", "3": "v2b", "4": "v2b"},
      {"0": "v2_5070ti", "1": "v2_5070ti", "2": "v2b_5070ti", "3": "v2b_5070ti", "4": "v2b_5070ti"}]}'
Test bank (members = probs npz files that hold test_prob):
  python analysis/tree_bank.py '{"tag": "full2_test", "test": ["runs/v2_full_5070ti/probs_f-1.npz", "runs/v2_full_s2027_5070ti/probs_f-1.npz"]}'
Optional keys: shard / nshard (independent processes), out (default scratch/bank), folds.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'src'))
from common import ann_mask, load_coco
from postproc import upsample
from treeselect import LEVELS, build_tree

DATA = WORK / 'data/MAGFiLO_1.0_Kaggle_2026'


def gt_index(cache):
    """file -> [(record id, [flat pixel indices of each filament])]"""
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    images, anns, by_file = load_coco(DATA / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    out = {f: [(r, [np.flatnonzero(ann_mask(a)).astype(np.int32) for a in anns[r]]) for r in recs] for f, recs in by_file.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(out))
    return out


def node_iou(tree, recs):
    """(N, 3) best IoU of every node against each record's filaments (-1 where the image has no such record)."""
    n = len(tree['level'])
    out = np.full((n, 3), -1, np.float32)
    area = tree['nodes'][:, 1].astype(np.float64)
    for k, lab in enumerate(tree['labels']):
        m = np.flatnonzero(tree['level'] == k)
        if len(m) == 0:
            continue
        flat = lab.ravel()
        top = int(flat.max()) + 1
        for r, (_, rec) in enumerate(recs):
            best = np.zeros(len(m))
            for ix in rec:
                inter = np.bincount(flat[ix], minlength=top)[tree['comp'][m]]
                best = np.maximum(best, inter / (area[m] + len(ix) - inter))
            out[m, r] = best
    return out


def member_maps(arrs, i):
    """per-member maps at 2048 on the 0-255 scale, upsampled exactly like the fixed-rule pipeline"""
    return [upsample(a[i]).astype(np.float32) for a in arrs]


def main():
    spec = json.loads(sys.argv[1])
    out = WORK / spec.get('out', 'scratch/bank') / spec['tag']
    out.mkdir(parents=True, exist_ok=True)
    shard, nshard = spec.get('shard', 0), spec.get('nshard', 1)
    t0 = time.time()
    jobs = []  # (fold or -1, [npz paths], key holding the maps, key holding the file names, image folder)
    if 'test' in spec:
        jobs.append((-1, [WORK / p for p in spec['test']], 'test_prob', 'test_files', DATA / 'test/test_images'))
    else:
        for fold in spec.get('folds', range(5)):
            jobs.append((fold, [WORK / 'runs' / m[str(fold)] / f'probs_f{fold}.npz' for m in spec['members']],
                         'val_prob', 'val_files', DATA / 'train/train_images'))
    gt = None if 'test' in spec else gt_index(WORK / 'scratch/bank/gt_index.pkl')
    done = 0
    for fold, paths, pk, fk, folder in jobs:
        target = out / f'bank_f{fold}_s{shard}of{nshard}.pkl'
        if target.exists():
            continue
        arrs, files = [], None
        for path in paths:
            with np.load(path, allow_pickle=False) as z:
                fl = [str(x) for x in z[fk]]
                assert files is None or fl == files
                files = fl
                arrs.append(z[pk])
        items = []
        for i in range(shard, len(files), nshard):
            f = files[i]
            maps = member_maps(arrs, i)
            prob = maps[0] if len(maps) == 1 else np.mean(maps, 0)
            image = cv2.imread(str(folder / f), cv2.IMREAD_GRAYSCALE)
            tree = build_tree(prob, image=image, members=maps if len(maps) > 1 else None, keep_labels=gt is not None)
            item = dict(file=f, fold=fold, nodes=tree['nodes'], level=tree['level'].astype(np.int16),
                        comp=tree['comp'].astype(np.int32), parent=tree['parent'].astype(np.int32))
            if gt is not None:
                recs = gt[f]
                item['iou'] = node_iou(tree, recs)
                item['records'] = [r for r, _ in recs]
                item['n_gt'] = [len(rec) for _, rec in recs]
            items.append(item)
            done += 1
            if done % 20 == 0:
                print(json.dumps(dict(shard=shard, fold=fold, done=done, secs=round(time.time() - t0))), flush=True)
        target.write_bytes(pickle.dumps(dict(levels=LEVELS, items=items)))
        del arrs
    print(json.dumps(dict(shard=shard, finished=True, images=done, secs=round(time.time() - t0))), flush=True)


if __name__ == '__main__':
    main()
