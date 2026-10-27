"""Score the average of one or more runs with post-processing re-tuned per fold (nested).

Averaging models smooths the probability maps, so the v2 rule (t=.4, area >= 300, mean >= .85) tuned
for a single model may no longer fit. For each fold, the (threshold, min area, min mean) setting is
chosen on the other folds and applied to this one; the fixed v2 rule is reported alongside.

usage: SOLAR_DATA=<data dir> python analysis/retune_nested.py <run dir> [<run dir> ...]
"""
import itertools
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

RUNS = [Path(p) for p in sys.argv[1:]]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import fastsweep as fs  # noqa: E402
from common import ANN, ann_mask, load_coco  # noqa: E402

T_LO, AREA, MEAN = (.35, .4, .45), (200, 300, 400), (.75, .8, .85, .9)
V2 = (.4, 300, .85)


def tables(t_lo):
    with get_context('fork').Pool(max(1, os.cpu_count() - 4)) as pool:
        return pool.map(fs.image_table, [(i, t_lo, 8) for i in range(len(fs.FILES))], chunksize=2)


if __name__ == '__main__':
    _, anns, by_file = load_coco(ANN)
    folds = sorted(set.intersection(*[{int(p.stem[-1]) for p in r.glob('probs_f*.npz')} for r in RUNS]))
    probs, files, fold_of = [], [], []
    for k in folds:
        zs = [np.load(r / f'probs_f{k}.npz') for r in RUNS]
        fl = [str(f) for f in zs[0]['val_files']]
        assert all([str(f) for f in z['val_files']] == fl for z in zs)
        probs.append(np.round(np.mean([z['val_prob'].astype(np.float32) for z in zs], 0)).astype(np.uint8))
        files += fl
        fold_of += [k] * len(fl)
    fs.PROBS, fs.FILES = np.concatenate(probs), files
    fs.GT = {f: [[np.flatnonzero(ann_mask(a)).astype(np.int32) for a in anns[r]] for r in by_file[f]] for f in files}
    fold_of = np.array(fold_of)
    tab = {t: tables(t) for t in T_LO}  # per image: component features + IoU vs every record

    def pq(setting, mask):
        t, a, m = setting
        return fs.score([x for x, keep in zip(tab[t], mask) if keep], a, m, 0)

    grid = list(itertools.product(T_LO, AREA, MEAN))
    print('runs:', ' + '.join(f'{r.parent.name}/{r.name}' for r in RUNS))
    fixed, nested = fs.PQ(), fs.PQ()
    for k in folds:
        here, rest = fold_of == k, fold_of != k
        best = max(grid, key=lambda g: pq(g, rest).value)
        a, b = pq(V2, here), pq(best, here)
        fixed.merge(a)
        nested.merge(b)
        print(f'fold {k}: v2 rule {a.value:.4f} | re-tuned on other folds {best} -> {b.value:.4f}')
    print(f'all folds: v2 rule {fixed}  |  nested re-tuned {nested}')
