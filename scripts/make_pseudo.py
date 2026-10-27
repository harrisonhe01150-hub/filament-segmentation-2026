"""Soft pseudo-labels for unlabeled frames (experiment S1): one teacher per fold, no test images.

For student fold k the teacher is the existing fold-k model, which never saw fold k's validation images,
so the student's out-of-fold score stays honest. Writes images.npy (shared) and probs_f<k>.npy (8-way TTA
probability maps, uint8) at the training size.

  python scripts/make_pseudo.py --images data/gong_unlabeled --teachers runs/v2 runs/v2b \
      --config configs/v2.json --folds 0 1 --out data/pseudo_s1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from train_seg import CFG, pick_device, predict
from predict import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--teachers', type=Path, nargs='+', required=True, help='folders holding model_f<k>.pt')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--folds', type=int, nargs='+', default=[0, 1])
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    cfg = dict(CFG, **json.loads(a.config.read_text()))
    S = cfg['size']
    a.out.mkdir(parents=True, exist_ok=True)
    files = sorted(x.name for x in a.images.glob('*.png'))
    assert files, f'no .png frames in {a.images}'
    imgs = np.stack([cv2.resize(cv2.imread(str(a.images / f), cv2.IMREAD_GRAYSCALE), (S, S), interpolation=cv2.INTER_AREA)
                     for f in files])
    np.save(a.out / 'images.npy', imgs)
    (a.out / 'files.json').write_text(json.dumps(files))
    device = pick_device()
    for k in a.folds:
        t0 = time.time()
        weights = next(d / f'model_f{k}.pt' for d in a.teachers if (d / f'model_f{k}.pt').is_file())
        probs = predict(load_model(weights, cfg['encoder'], device), imgs, cfg['tta'])
        np.save(a.out / f'probs_f{k}.npy', probs)
        print(json.dumps(dict(fold=k, teacher=str(weights), images=len(files), mean_prob=round(float(probs.mean()) / 255, 4),
                              frac_ge_04=round(float((probs >= 102).mean()), 4), secs=round(time.time() - t0))), flush=True)


if __name__ == '__main__':
    main()
