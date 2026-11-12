"""Final pipeline, end to end, for a folder of images: dense maps -> seeds -> learned keep rule -> refined shapes.

Nothing is trained here; the keep model is refitted from its training table (a few seconds, deterministic).

Stages, per image
  1. dense map   two full-data U-Nets (EfficientNet-B4, 1536 px input, 8-way TTA), averaged at 2048 px
  2. seeds       pixels >= 0.4, grouped by a closing of radius 8 and by pixels >= 0.1 (refine.image_bank, bridge 0.1);
                 seeds of at least 100 px are candidates
  3. keep        the learned keep rule (src/keep.py): a seed is submitted when its predicted score > keep.TAU
  4. shape       the stage-2 refiners (src/refine.py) are averaged, blended 50/50 with the dense probability inside
                 the seed's crop and thresholded at 0.5; the pieces that touch the seed are the mask; seeds whose
                 refined masks claim the same pixels are joined (refine.assemble, absorb 0.3; rare: 1 of 4360
                 kept candidates out of fold, none in the test set; kept as a safeguard)
                 --no-refiner: each kept seed is written at probability >= 0.48 instead (lighter variant)

Usage (dense maps from weights)
  python src/predict_final.py --dense weights/model_full_s2026.pt weights/model_full_s2027.pt --config configs/v2_full.json
      --refiners weights/refiner_f0.pt weights/refiner_f1.pt weights/refiner_f2.pt weights/refiner_f3.pt weights/refiner_f4.pt
      --keep-table weights/keep_table.npz --fp32 --out outputs/submission.csv
(--fp32 is what the submitted file was made with: on CUDA the default is float16, which moves a few borderline
candidates; on CPU and Apple MPS everything is float32 anyway)
or with dense maps saved by train_seg.py (test_files / test_prob):
  python src/predict_final.py --probs runs/v2_full_5070ti/probs_f-1.npz runs/v2_full_s2027_5070ti/probs_f-1.npz ...

Only the image itself is used for an image's prediction (no file-name information, no other frames).
Out-of-fold scores of the stages are in EXPERIMENTS.md.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keep
import refine
from make_submission import check_disjoint, write_csv
from postproc import upsample

FINAL = dict(bridge=0.1, blend=0.5, t_r=0.5, absorb=0.3, draw=0.48)
TEST_IMAGES = Path('data/MAGFiLO_1.0_Kaggle_2026/test/test_images')


def dense_maps(model_paths, cfg, images, files, device, fp32):
    """uint8 probability maps (N, size, size) of every dense model."""
    from predict import load_model
    from train_seg import predict, read_gray
    imgs = np.stack([read_gray(images / f, cfg['size']) for f in files])
    out = []
    for path in model_paths:
        model = load_model(path, cfg['encoder'], device)
        out.append(predict(model, imgs, cfg['tta'], fp16=not fp32))
        del model
    return out


def image_instances(members, gray, keep_model, refiners, tau=keep.TAU, device='cpu', fp16=True):
    """Label map (0 = background, 1..K) of one image and a few counts.

    members: the dense models' maps of the image at 2048 px (uint8); gray: the uint8 image.
    refiners: loaded refiner models (several are averaged), or an empty list for the lighter variant.
    fp16: float16 refiner inference on CUDA (faster); False = float32, the same numbers as on other devices.
    """
    prob = np.mean([m.astype(np.float32) for m in members], 0)
    seeds = refine.image_bank(prob, gray, None, bridge=FINAL['bridge'])
    info = dict(seeds=len(seeds['label']), fixed_rule=int(seeds['kept'].sum()), kept=0)
    labels = np.zeros(refine.IMAGE_SHAPE, np.int32)
    if not len(seeds['label']):
        return labels, info
    score = keep_model.predict(keep.image_descriptors(seeds, members))
    chosen = np.flatnonzero(score > tau)
    info['kept'] = len(chosen)
    if not len(chosen):
        return labels, info
    if not refiners:
        flat = labels.ravel()
        for n, j in enumerate(chosen, 1):
            flat[seeds['pix'][j]] = n
        return np.where(prob >= FINAL['draw'] * 255, labels, 0), info
    import torch
    planes = torch.from_numpy(np.stack([seeds[name][chosen] for name in refine.PLANES[:refine.N_INPUT]], 1)).to(device)
    acc = np.zeros((len(chosen), refine.CROP_H, refine.CROP_W), np.float32)
    for model in refiners:
        acc += refine.predict_crops(model, planes, fp16=fp16)
    refined = np.round(acc / len(refiners))
    w = FINAL['blend']
    blended = np.round(w * refined + (1 - w) * seeds['prob'][chosen].astype(np.float32)).astype(np.uint8)
    items = [dict(pix=seeds['pix'][j], M=seeds['M'][j], seed=seeds['seed'][j], refined=blended[t]) for t, j in enumerate(chosen)]
    labels, _, _ = refine.paste_image(refine.IMAGE_SHAPE, items, FINAL['t_r'], absorb=FINAL['absorb'])
    return labels, info


def rows_of(labels, file_name):
    stem = file_name.rsplit('.', 1)[0]
    rows = []
    for n in range(1, int(labels.max()) + 1):
        m = np.asfortranarray((labels == n).astype(np.uint8))
        if m.any():
            rows.append((f'{stem}_{len(rows) + 1}', mu.encode(m)['counts'].decode()))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--probs', type=Path, nargs='+', help='.npz files with test_files and test_prob (saved dense maps), one per model')
    p.add_argument('--dense', type=Path, nargs='+', help='dense model weights (used when --probs is not given)')
    p.add_argument('--config', type=Path, help='training config of the dense models (with --dense)')
    p.add_argument('--images', type=Path, default=TEST_IMAGES)
    p.add_argument('--refiners', type=Path, nargs='*', default=[], help='refiner_f*.pt files, averaged')
    p.add_argument('--no-refiner', action='store_true', help='lighter variant: kept seeds written at probability >= 0.48')
    p.add_argument('--keep-table', type=Path, required=True)
    p.add_argument('--tau', type=float, default=keep.TAU)
    p.add_argument('--device', help='cuda, mps or cpu (default: best available)')
    p.add_argument('--fp32', action='store_true', help='float32 on CUDA too (dense models with --dense, and the refiners), so other machines reproduce the file')
    p.add_argument('--save-probs', type=Path, help='folder to store the dense maps computed with --dense (dense_<n>.npz, usable as --probs)')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(f'{a.out} exists; choose a new path')
    if bool(a.refiners) == a.no_refiner:
        p.error('give --refiners, or --no-refiner for the lighter variant')
    from train_seg import CFG, pick_device
    device = pick_device(a.device)
    t0 = time.time()
    if a.probs:
        files, maps = None, []
        for path in a.probs:
            with np.load(path, allow_pickle=False) as z:
                names = [str(x) for x in z['test_files']]
                if files is not None and names != files:
                    raise ValueError(f'{path} lists other images than {a.probs[0]}')
                files = names
                maps.append(z['test_prob'])
    else:
        if not a.dense or not a.config:
            p.error('give --probs, or --dense with --config')
        cfg = dict(CFG, **json.loads(a.config.read_text()))
        files = sorted(path.name for path in a.images.glob('*.jpeg'))
        if not files:
            raise FileNotFoundError(f'no .jpeg images in {a.images}')
        maps = dense_maps(a.dense, cfg, a.images, files, device, a.fp32)
        if a.save_probs:
            a.save_probs.mkdir(parents=True, exist_ok=True)
            for n, m in enumerate(maps):
                np.savez_compressed(a.save_probs / f'dense_{n}.npz', test_files=np.array(files), test_prob=m)
    keep_model = keep.load_model(a.keep_table, members=len(maps))
    refiners = [refine.load_refiner(path, refine.ENCODER, device) for path in a.refiners]
    rows, empty, total = [], [], dict(seeds=0, fixed_rule=0, kept=0)
    for i, f in enumerate(files):
        gray = cv2.imread(str(a.images / f), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != refine.IMAGE_SHAPE:
            raise ValueError(f'cannot read {a.images / f} as a {refine.IMAGE_SHAPE} image')
        labels, info = image_instances([upsample(m[i]) for m in maps], gray, keep_model, refiners, a.tau, device, fp16=not a.fp32)
        new = rows_of(labels, f)
        rows += new
        if not new:
            empty.append(f)
        for k in total:
            total[k] += info[k]
    checked = check_disjoint(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, a.out)
    print(json.dumps(dict(rows=len(rows), images=len(files), images_checked_disjoint=checked, empty_images=empty,
                          candidates=total['seeds'], kept=total['kept'], fixed_rule_would_keep=total['fixed_rule'],
                          tau=a.tau, refiners=[str(x) for x in a.refiners], settings=FINAL, device=str(device), fp32=a.fp32,
                          secs=round(time.time() - t0))), flush=True)


if __name__ == '__main__':
    main()
