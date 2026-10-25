"""Predict test masks from saved fold weights and write the submission CSV. No training.

Loads every weights/model_f*.pt (or --folds), runs each on the test images with the same 8-way
TTA as training, averages the fold maps at 2048 and applies the v2 post-processing, then checks
that no two masks of an image overlap. Runs on CUDA, Apple MPS or CPU. Use --fp32 for the final
submission so other machines reproduce it (float16 is only used on CUDA and is faster).

  python src/predict.py --weights weights --config configs/v2.json --fp32 --out outputs/submission.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_seg import CFG, build_model, pick_device, predict, read_gray
from make_submission import build_rows, check_disjoint, write_csv

# v2 post-processing, tuned on the five-fold OOF predictions (docs/BASELINE_AND_PROVENANCE.txt)
POST = dict(t_lo=.4, close=8, min_area=300, min_mean=.85, min_peak=0)
TEST_IMAGES = Path('data/MAGFiLO_1.0_Kaggle_2026/test/test_images')


def load_model(path, encoder, device):
    import torch
    model = build_model(encoder, weights=None)  # architecture only; no ImageNet download
    state = torch.load(path, map_location='cpu', weights_only=True)
    # accept float16 checkpoints (saved that way to stay under GitHub's 100 MB file limit)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=Path, default=Path('weights'))
    p.add_argument('--folds', type=int, nargs='*', help='default: every model_f*.pt in --weights')
    p.add_argument('--config', type=Path, required=True, help='training config used for the weights (encoder, size, tta)')
    p.add_argument('--images', type=Path, default=TEST_IMAGES)
    p.add_argument('--device', help='cuda, mps or cpu (default: best available)')
    p.add_argument('--fp32', action='store_true', help='float32 on CUDA too, so results match other devices')
    p.add_argument('--post', help='JSON overriding the post-processing settings')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()

    cfg = dict(CFG, **json.loads(a.config.read_text()))
    post = dict(POST, **(json.loads(a.post) if a.post else {}))
    if a.folds:
        paths = [a.weights / f'model_f{f}.pt' for f in a.folds]
    else:
        paths = sorted(a.weights.glob('model_f*.pt'))
    if not paths or not all(path.is_file() for path in paths):
        raise FileNotFoundError(f'missing weights: {[str(path) for path in paths] or a.weights}')
    files = sorted(path.name for path in a.images.glob('*.jpeg'))
    if not files:
        raise FileNotFoundError(f'no .jpeg images in {a.images}')
    if a.out.exists():
        raise FileExistsError(f'{a.out} exists; choose a new path')

    device = pick_device(a.device)
    t0 = time.time()
    imgs = np.stack([read_gray(a.images / f, cfg['size']) for f in files])
    stacks = []
    for path in paths:
        model = load_model(path, cfg['encoder'], device)
        stacks.append(predict(model, imgs, cfg['tta'], fp16=not a.fp32))
        print(json.dumps(dict(event='fold_predicted', weights=str(path), secs=round(time.time() - t0))), flush=True)
        del model
    rows, empty = build_rows(files, stacks, post)
    checked = check_disjoint(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, a.out)
    print(json.dumps(dict(rows=len(rows), images=len(files), images_checked_disjoint=checked,
                          empty_images=empty, device=str(device), fp32=a.fp32, folds=[str(x) for x in paths],
                          post=post, secs=round(time.time() - t0))), flush=True)


if __name__ == '__main__':
    main()
