"""Descriptor tables for the learned keep rule (src/keep.py). CPU only.

Reads a seed bank (analysis/refine_bank.py, built with "bridge": 0.1) and the member models' probability maps the
bank was built from, and writes one table per fold: keep_f<fold>.npz with X (descriptors, keep.NAMES order), fname,
label, kept (the fixed rule) and, for training folds, best_iou (per record; the labels come from it).

Out-of-fold tables (members = one {fold: run folder} map per stage-1 model, as for refine_bank.py):
  python analysis/keep_bank.py '{"bank": "scratch/ws/main/rbank/kc_bridge", "out": "scratch/keep/oof", "members":
      [{"0": "v2", "1": "v2", "2": "v2b", "3": "v2b", "4": "v2b"},
       {"0": "v2_5070ti", "1": "v2_5070ti", "2": "v2b_5070ti", "3": "v2b_5070ti", "4": "v2b_5070ti"}]}'
Test table:
  python analysis/keep_bank.py '{"bank": "scratch/ws/main/rbank/test_full2_bridge", "out": "scratch/keep/test",
      "test": ["runs/v2_full_5070ti/probs_f-1.npz", "runs/v2_full_s2027_5070ti/probs_f-1.npz"]}'
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'src'))
import keep
import refine
from postproc import upsample

SEED_KEYS = ['area', 'mean', 'long', 'short', 'scale', 'M', 'pix', 'image', 'prob', 'seed', 'others']


def fold_table(bank_dir, fold, map_paths, map_key, file_key):
    keys = ['fname', 'label', 'kept'] + SEED_KEYS + (['best_iou'] if fold >= 0 else [])
    b = refine.load_bank(bank_dir, keys, fold)
    maps, files = [], None
    for path in map_paths:
        with np.load(path, allow_pickle=False) as z:
            names = [str(x) for x in z[file_key]]
            if files is not None and names != files:
                raise ValueError(f'{path} lists other images than {map_paths[0]}')
            files = names
            maps.append(z[map_key])
    if len(maps) != 2:
        raise ValueError('the descriptors compare exactly two member models')
    row = {f: i for i, f in enumerate(files)}
    fname = b['fname'].astype(str)
    X = np.zeros((len(fname), len(keep.NAMES)), np.float32)
    start = 0
    while start < len(fname):                           # the bank is sorted by image
        end = start
        while end < len(fname) and fname[end] == fname[start]:
            end += 1
        seeds = {k: (b[k][start:end] if k != 'pix' else b[k][start:end]) for k in SEED_KEYS}
        X[start:end] = keep.image_descriptors(seeds, [upsample(m[row[fname[start]]]) for m in maps])
        start = end
    out = dict(X=X, fname=fname, label=b['label'], kept=b['kept'])
    if fold >= 0:
        out['best_iou'] = b['best_iou']
    return out


def main():
    spec = json.loads(sys.argv[1])
    bank, out = WORK / spec['bank'], WORK / spec['out']
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if 'test' in spec:
        jobs = [(-1, [WORK / p for p in spec['test']], 'test_prob', 'test_files')]
    else:
        jobs = [(k, [WORK / 'runs' / m[str(k)] / f'probs_f{k}.npz' for m in spec['members']], 'val_prob', 'val_files')
                for k in spec.get('folds', range(5))]
    for fold, paths, map_key, file_key in jobs:
        table = fold_table(bank, fold, paths, map_key, file_key)
        np.savez_compressed(out / f'keep_f{fold}.npz', **table)
        print(json.dumps(dict(fold=fold, seeds=len(table['fname']), descriptors=len(keep.NAMES), secs=round(time.time() - t0))), flush=True)


if __name__ == '__main__':
    main()
