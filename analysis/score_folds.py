"""Score one training run's held-out folds with the v2 post-processing (official PQ rules).

Used for the per-machine control and for experiments (TEAM_ROLES.md section 4):
  python analysis/score_folds.py --run runs/v2_5070ti --folds 0 1
Reads probs_f<k>.npz (validation maps only) and the train annotations; never reads test data.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'src'))
from common import PQ, ANN, load_coco, ann_mask
from postproc import upsample, score_file
from fastsweep import components
from train_seg import fold_of_files

POST = dict(t_lo=.4, close=8, min_area=300, min_mean=.85)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True, help='folder with probs_f<k>.npz')
    p.add_argument('--folds', type=int, nargs='+', default=[0, 1])
    p.add_argument('--annotations', type=Path, default=WORK / ANN)
    p.add_argument('--post', help='JSON overriding the post-processing, e.g. {"bridge": 0.1, "draw": 0.48}')
    a = p.parse_args()
    post = dict(POST, **(json.loads(a.post) if a.post else {}))
    images, anns, by_file = load_coco(a.annotations)
    assigned = fold_of_files(by_file)
    total, rows = PQ(), []
    for fold in a.folds:
        with np.load(a.run / f'probs_f{fold}.npz', allow_pickle=False) as z:
            files, prob = [str(x) for x in z['val_files']], z['val_prob']
        assert set(files) == {f for f, k in assigned.items() if k == fold}, f'fold {fold} files do not match the split'
        one = PQ()
        for image, prediction in zip(files, prob):
            up = upsample(prediction)
            labels, feat = components(up, post['t_lo'], post['close'], post.get('bridge', 0))
            keep = (feat[:, 0] >= post['min_area']) & (feat[:, 1] >= post['min_mean'])
            if post.get('draw', post['t_lo']) > post['t_lo']:   # written extent is tighter than the instance definition
                labels = np.where(up >= post['draw'] * 255, labels, 0)
                keep &= np.bincount(labels.ravel(), minlength=len(feat) + 1)[1:] > 0
            remap = np.zeros(len(feat) + 1, np.int32)
            remap[1:][keep] = np.arange(1, int(keep.sum()) + 1)
            recs = [[np.flatnonzero(ann_mask(an)).astype(np.int32) for an in anns[r]] for r in by_file[image]]
            one.merge(score_file(remap[labels], recs, PQ()))
        total.merge(one)
        rows.append(dict(fold=fold, images=len(files), pq=round(one.value, 4), tp=one.tp, fp=one.fp, fn=one.fn))
    print(json.dumps(dict(run=str(a.run), folds=rows, combined_pq=round(total.value, 4), post=post), indent=2))


if __name__ == '__main__':
    main()
