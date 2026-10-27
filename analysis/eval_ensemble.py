"""Score single runs and their probability average on the OOF folds they share (v2 post-processing).

Each run dir holds probs_f{k}.npz from train_seg.py; folds present in every run are scored. Averaging is
done at native resolution and rounded to uint8, as the handoff's fastsweep.load does.

usage: SOLAR_DATA=<data dir> python analysis/eval_ensemble.py <run dir> [<run dir> ...]
"""
import sys
from pathlib import Path

import numpy as np

RUNS = [Path(p) for p in sys.argv[1:]]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import ANN, PQ, ann_mask, load_coco  # noqa: E402
from fastsweep import components  # noqa: E402
from postproc import score_file, upsample  # noqa: E402


def score(prob_u8, recs):
    labels, feat = components(upsample(prob_u8), .4, 8)
    keep = (feat[:, 0] >= 300) & (feat[:, 1] >= .85)
    remap = np.zeros(len(feat) + 1, np.int32)
    remap[1:][keep] = np.arange(1, keep.sum() + 1)
    return score_file(remap[labels], recs, PQ())


_, anns, by_file = load_coco(ANN)
folds = sorted(set.intersection(*[{int(p.stem[-1]) for p in r.glob('probs_f*.npz')} for r in RUNS]))
names = [f'{r.parent.name}/{r.name}' for r in RUNS] + (['average'] if len(RUNS) > 1 else [])
total = {n: PQ() for n in names}
for k in folds:
    zs = [np.load(r / f'probs_f{k}.npz') for r in RUNS]
    files = [str(f) for f in zs[0]['val_files']]
    assert all([str(f) for f in z['val_files']] == files for z in zs), f'fold {k}: runs disagree on validation files'
    per = {n: PQ() for n in names}
    for i, f in enumerate(files):
        recs = [[np.flatnonzero(ann_mask(a)).astype(np.int32) for a in anns[r]] for r in by_file[f]]
        probs = [z['val_prob'][i] for z in zs]
        if len(probs) > 1:
            probs.append(np.round(np.mean([p.astype(np.float32) for p in probs], 0)).astype(np.uint8))
        for n, p in zip(names, probs):
            per[n].merge(score(p, recs))
    for n in names:
        total[n].merge(per[n])
        print(f'fold {k}  {n:32s} {per[n]}', flush=True)
for n in names:
    print(f'folds {folds}  {n:32s} {total[n]}')
