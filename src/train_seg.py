"""Solar filament semantic segmentation (soft multi-annotator targets) for Kaggle GPU.

Pipeline: full-disk 2048 -> `size` grayscale (1536 for v2, see configs/), U-Net (ImageNet
encoder), soft target = mean of annotator masks, BCE + soft Dice, GPU-side augmentation. Per
fold, saves the final-epoch weights plus 8-way dihedral TTA probability maps for the held-out
fold and the test set. Instances are derived later (connected components + tuned
post-processing). To predict from saved weights without training, use predict.py.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

CFG = dict(
    encoder='efficientnet-b4', size=1024, crop=768, epochs=40, batch=6, lr=4e-4, wd=1e-4,
    warmup=150, folds_to_run=[0, 1], n_folds=5, seed=2026, tta=8, run_name='seg_v1',
    # Experiment B (TEAM_ROLES.md), both off by default so the baseline is unchanged:
    # small_crop_p = share of training crops centred on a filament smaller than small_area_max px;
    # area_weight_ref > 0 weights each filament's pixels in the BCE by clip(sqrt(ref / area), 1, max).
    small_crop_p=0.0, small_area_max=1000, area_weight_ref=0, area_weight_max=4.0,
    # Experiment A: segmentation-derived auxiliary centerline, off for the v2 control.
    centerline_weight=0.0, centerline_radius=1, predict_test=True,
)
COMP = 'filament-segmentation-2026'


def find_root():
    for base in [Path(os.environ.get('SOLAR_DATA', '/kaggle/input')), Path('/kaggle/input'), Path('data')]:
        if base.exists():
            hits = list(base.rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
            if hits:
                return hits[0].parent.parent
    raise FileNotFoundError('competition data not found')


def fold_of_files(files, n_folds=5, seed=2026):
    """Group by observation date (same day never split), deterministic greedy balance."""
    groups = defaultdict(list)
    for f in sorted(files):
        groups[f[:8]].append(f)
    keys = sorted(groups, key=lambda k: hashlib.md5(f'{seed}-{k}'.encode()).hexdigest())
    sizes = [0] * n_folds
    out = {}
    for k in sorted(keys, key=lambda k: -len(groups[k])):
        i = min(range(n_folds), key=lambda i: (sizes[i], i))
        for f in groups[k]:
            out[f] = i
        sizes[i] += len(groups[k])
    return out


def load_annotations(root):
    d = json.loads((root / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text())
    recs = defaultdict(list)  # file -> record ids
    for im in d['images']:
        recs[im['file_name']].append(im['id'])
    anns = defaultdict(list)
    for a in d['annotations']:
        seg = a['segmentation']
        seg = json.loads(seg) if isinstance(seg, str) else seg
        if seg and not isinstance(seg[0], list):
            seg = [seg]
        anns[a['image_id']].append(seg)
    return recs, anns


def soft_target(recs_of_file, anns, size):
    import cv2
    from pycocotools import mask as mu
    acc = np.zeros((2048, 2048), np.float32)
    for r in recs_of_file:
        m = np.zeros((2048, 2048), bool)
        for seg in anns[r]:
            m |= mu.decode(mu.merge(mu.frPyObjects(seg, 2048, 2048))).astype(bool)
        acc += m
    acc /= len(recs_of_file)
    return cv2.resize(acc, (size, size), interpolation=cv2.INTER_AREA)


def small_filament_maps(recs_of_file, anns, size, area_ref, w_max, small_max):
    """Experiment B inputs for one image, from every annotator's filament masks at 2048.

    Returns (extra, centres). extra: per-pixel BCE weight minus 1 at `size`, uint8 where 255 means
    w_max - 1; a filament's weight is clip(sqrt(area_ref / area), 1, w_max), overlaps take the larger,
    background stays 0 (area_ref=0 gives all zeros). centres: normalized (x, y) in [-1, 1] of every
    filament smaller than small_max px.
    """
    import cv2
    from pycocotools import mask as mu
    extra = np.zeros((2048, 2048), np.float32)
    centres = []
    for r in recs_of_file:
        for seg in anns[r]:
            m = mu.decode(mu.merge(mu.frPyObjects(seg, 2048, 2048))).astype(bool)
            area = int(m.sum())
            if area == 0:
                continue
            if area_ref:
                w = min(max(math.sqrt(area_ref / area), 1.0), w_max)
                np.maximum(extra, (w - 1) * m, out=extra)
            if area < small_max:
                ys, xs = np.nonzero(m)
                centres.append(((xs.mean() + .5) / 1024 - 1, (ys.mean() + .5) / 1024 - 1))
    extra = cv2.resize(extra, (size, size), interpolation=cv2.INTER_AREA)
    return (extra * (255 / (w_max - 1))).round().clip(0, 255).astype(np.uint8), centres


def read_gray(path, size):
    import cv2
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im.shape != (2048, 2048):
        raise ValueError(f'unexpected shape {im.shape} for {path}')
    return cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)


def build_model(encoder, weights='imagenet', classes=1):
    import segmentation_models_pytorch as smp
    try:
        return smp.Unet(encoder, encoder_weights=weights, in_channels=3, classes=classes)
    except Exception as e:
        raise RuntimeError(f'Cannot load requested encoder {encoder}; no fallback was trained') from e


MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_input(x):
    """x: (B,1,H,W) float in [0,1] -> normalized 3-channel."""
    import torch
    m = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x.expand(-1, 3, -1, -1) - m) / s


def augment(img, tgt, crop, gen, centres=None):
    """GPU augmentation. img: (B,1,S,S), tgt: (B,C,S,S) float. Returns cropped (B,1,crop,crop), (B,C,crop,crop).

    centres: optional (B,2) normalized crop centres (experiment B); NaN rows keep the random centre.
    """
    import torch
    import torch.nn.functional as F
    B, _, S, _ = img.shape
    dev = img.device
    ang = (torch.rand(B, device=dev, generator=gen) * 2 - 1) * math.pi
    scale = torch.exp((torch.rand(B, device=dev, generator=gen) * 2 - 1) * math.log(1.15))
    flip = (torch.rand(B, device=dev, generator=gen) < .5).float() * 2 - 1
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = cos * flip; theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin * flip; theta[:, 1, 1] = cos
    # random crop centre in normalized coords (keep mostly inside the disk region)
    c = crop / S
    off = (torch.rand(B, 2, device=dev, generator=gen) * 2 - 1) * (1 - c) * .9
    if centres is not None:
        # centre these crops on the given filament; clamp so the crop stays inside the image
        forced = ~torch.isnan(centres[:, 0])
        off = torch.where(forced[:, None], centres.nan_to_num().clamp(-(1 - c), 1 - c), off)
    theta[:, :, 2] = off
    # build crop grid: sample a crop x crop window, rotated/scaled around a random centre
    base = F.affine_grid(torch.eye(2, 3, device=dev).unsqueeze(0).expand(B, 2, 3) * 1.0,
                         (B, 1, crop, crop), align_corners=False) * c
    A = theta[:, :, :2]
    grid = torch.einsum('bij,bhwj->bhwi', A, base) + theta[:, :, 2].view(B, 1, 1, 2)
    xi = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    yt = F.grid_sample(tgt, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    # photometric
    g = torch.exp((torch.rand(B, 1, 1, 1, device=dev, generator=gen) * 2 - 1) * math.log(1.3))
    xi = xi.clamp(0, 1) ** g
    a = 1 + (torch.rand(B, 1, 1, 1, device=dev, generator=gen) * 2 - 1) * .15
    b = (torch.rand(B, 1, 1, 1, device=dev, generator=gen) * 2 - 1) * .06
    xi = (xi * a + b + torch.randn(xi.shape, device=dev, generator=gen) * .01).clamp(0, 1)
    return xi, yt


def dihedral(x, k):
    import torch
    if k >= 4:
        x = torch.flip(x, dims=[-1])
    return torch.rot90(x, k % 4, dims=[-2, -1])


def undihedral(x, k):
    import torch
    x = torch.rot90(x, -(k % 4), dims=[-2, -1])
    if k >= 4:
        x = torch.flip(x, dims=[-1])
    return x


def pick_device(name=None):
    """Explicit device name, else CUDA, else Apple MPS, else CPU."""
    import torch
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def predict(model, imgs, tta, bs=4, fp16=True):
    """imgs: uint8 (N,S,S) numpy -> uint8 probability maps (N,S,S).

    Runs on the model's device. On CUDA, fp16=True uses float16 autocast as in training; MPS and CPU
    always run in float32. Use fp16=False for maps that agree across devices (CPU vs a float16 GPU
    run moved a few borderline filaments in or out of a 1255-filament CSV).
    """
    import contextlib
    import torch
    out = np.zeros(imgs.shape, np.uint8)
    model.eval()
    dev = next(model.parameters()).device
    amp = (torch.autocast('cuda', dtype=torch.float16) if dev.type == 'cuda' and fp16
           else contextlib.nullcontext())
    if dev.type == 'cuda' and not fp16:
        # real float32: recent GPUs otherwise run float32 convolutions in TF32 (10-bit mantissa)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode(), amp:
        for i in range(0, len(imgs), bs):
            x = torch.from_numpy(imgs[i:i + bs]).to(dev).float().div(255).unsqueeze(1)
            acc = torch.zeros_like(x)
            for k in range(tta):
                # Channel zero is always the mask; the auxiliary head is training-only.
                acc += undihedral(model(to_input(dihedral(x, k)))[:, :1].float().sigmoid(), k)
            out[i:i + bs] = (acc / tta * 255).round().clamp(0, 255).byte().cpu().numpy()[:, 0]
    return out


def train_fold(fold, cfg, out_dir):
    import torch
    import torch.nn.functional as F
    t0 = time.time()
    aux_weight = float(cfg.get('centerline_weight', 0))
    if not math.isfinite(aux_weight) or aux_weight < 0:
        raise ValueError('centerline_weight must be finite and nonnegative')
    exp_b = bool(cfg.get('area_weight_ref', 0) or cfg.get('small_crop_p', 0))
    if aux_weight and exp_b:
        raise ValueError('A1 must be tested without experiment B options')
    root = find_root()
    recs, anns = load_annotations(root)
    files = sorted(recs)
    folds = fold_of_files(files, cfg['n_folds'], cfg['seed'])
    tr = [f for f in files if folds[f] != fold] if fold >= 0 else files
    va = [f for f in files if folds[f] == fold] if fold >= 0 else []
    test_files = (sorted(p.name for p in (root / 'test/test_images').glob('*.jpeg'))
                  if cfg.get('predict_test', True) else [])
    S = cfg['size']
    X = np.stack([read_gray(root / 'train/train_images' / f, S) for f in tr])
    Y = np.stack([(soft_target(recs[f], anns, S) * 255).round().astype(np.uint8) for f in tr])
    extra_info = {}
    if aux_weight:
        from centerline import centerline_target
        Z = np.stack([(centerline_target(recs[f], anns, S, cfg.get('centerline_radius', 1)) * 255)
                      .round().astype(np.uint8) for f in tr])
        # Keep the additional full-image target bank on the host to preserve T4 headroom.
        Zh = torch.from_numpy(Z)
        extra_info = dict(auxiliary_target='segmentation_derived_centerline',
                          human_spine_used=False, centerline_weight=aux_weight,
                          auxiliary_positive_pixels=int(np.count_nonzero(Z)))
    if exp_b:
        w_max = cfg.get('area_weight_max', 4.0)
        W = np.zeros((len(tr), S, S), np.uint8)
        cents = []
        for i, f in enumerate(tr):
            W[i], cs = small_filament_maps(recs[f], anns, S, cfg.get('area_weight_ref', 0), w_max,
                                           cfg.get('small_area_max', 1000))
            cents.append(cs)
        # host memory: at 1536 the GPU is nearly full already. Pinning can fail when RAM is busy
        # (B2 fold 0 crashed that way); unpinned memory only makes the copy a little slower.
        Wh = torch.from_numpy(W)
        try:
            Wh = Wh.pin_memory()
        except (RuntimeError, getattr(torch, 'AcceleratorError', RuntimeError)) as e:
            print(json.dumps(dict(event='warning', fold=fold, message=f'weight map not pinned: {e}'[:200])), flush=True)
        cen = torch.full((len(tr), max(1, max(map(len, cents))), 2), float('nan'))
        for i, cs in enumerate(cents):
            if cs:
                cen[i, :len(cs)] = torch.tensor(cs)
        ncen = torch.tensor([len(cs) for cs in cents]).cuda()
        cen = cen.cuda()
        extra_info = dict(small_filaments=int(ncen.sum()), images_with_small=int((ncen > 0).sum()))
    print(json.dumps(dict(event='data', fold=fold, train=len(tr), val=len(va), secs=time.time() - t0, **extra_info)), flush=True)
    torch.manual_seed(cfg['seed'] + fold)
    gen = torch.Generator(device='cuda').manual_seed(cfg['seed'] + fold)
    # A1 backward needs the space occupied by the v2 full-dataset GPU cache.
    # Host uint8 storage preserves the batch, precision, sample order and objective.
    on_gpu = S <= 1536 and not aux_weight
    Xg, Yg = torch.from_numpy(X), torch.from_numpy(Y)
    if on_gpu:
        Xg, Yg = Xg.cuda(), Yg.cuda()
    elif not aux_weight:
        Xg, Yg = Xg.pin_memory(), Yg.pin_memory()
    model = build_model(cfg['encoder'], classes=2 if aux_weight else 1).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    scaler = torch.amp.GradScaler('cuda')
    steps_per_epoch = math.ceil(len(tr) / cfg['batch'])
    total = steps_per_epoch * cfg['epochs']
    step = 0
    log = []
    for ep in range(cfg['epochs']):
        model.train()
        perm = torch.randperm(len(tr), device='cuda', generator=gen)
        losses = []
        mask_losses, auxiliary_losses = [], []
        for i in range(steps_per_epoch):
            idx = perm[i * cfg['batch']:(i + 1) * cfg['batch']]
            if len(idx) < 2:
                continue
            ic = idx if on_gpu else idx.cpu()
            img = Xg[ic].cuda(non_blocking=True).float().div(255).unsqueeze(1)
            tgt = Yg[ic].cuda(non_blocking=True).float().div(255).unsqueeze(1)
            wt = None
            auxiliary_target = None
            if aux_weight:
                z = Zh[idx.cpu()].cuda().float().div(255).unsqueeze(1)
                xi, both = augment(img, torch.cat([tgt, z], 1), cfg['crop'], gen)
                yt, auxiliary_target = both[:, :1], both[:, 1:2]
            elif exp_b:
                centres = None
                if cfg.get('small_crop_p', 0) > 0:
                    n = ncen[idx]
                    pick = (torch.rand(len(idx), device='cuda', generator=gen) < cfg['small_crop_p']) & (n > 0)
                    k = (torch.rand(len(idx), device='cuda', generator=gen) * n.clamp(min=1)).long()
                    chosen = cen[idx, k]
                    centres = torch.where(pick[:, None], chosen, torch.full_like(chosen, float('nan')))
                extra = Wh[idx.cpu()].cuda(non_blocking=True).float().unsqueeze(1) * ((w_max - 1) / 255)
                xi, both = augment(img, torch.cat([tgt, extra], 1), cfg['crop'], gen, centres)
                yt, wt = both[:, :1], 1 + both[:, 1:]  # padding outside the image gets weight 1
            else:
                xi, yt = augment(img, tgt, cfg['crop'], gen)
            lr =cfg['lr'] * min(1, (step + 1) / cfg['warmup']) * 0.5 * (1 + math.cos(math.pi * step / total))
            for g in opt.param_groups:
                g['lr'] = lr
            with torch.autocast('cuda', dtype=torch.float16):
                all_logits = model(to_input(xi))
            logit = all_logits[:, :1].float()
            p = logit.sigmoid()
            if wt is None:
                bce = F.binary_cross_entropy_with_logits(logit, yt)
            else:  # weighted mean keeps the loss on the same scale as the baseline
                bce = (F.binary_cross_entropy_with_logits(logit, yt, reduction='none') * wt).sum() / wt.sum()
            dice = 1 - (2 * (p * yt).sum() + 1) / (p.sum() + yt.sum() + 1)
            loss = bce + dice
            if aux_weight:
                from centerline import centerline_loss
                auxiliary = centerline_loss(all_logits[:, 1:2].float(), auxiliary_target)
                mask_losses.append(float(loss.detach()))
                auxiliary_losses.append(float(auxiliary.detach()))
                loss = loss + aux_weight * auxiliary
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            losses.append(float(loss))
        row = dict(event='epoch', fold=fold, epoch=ep + 1, loss=float(np.mean(losses)), lr=lr,
                   secs=time.time() - t0, steps=step)
        if aux_weight:
            row.update(mask_loss=float(np.mean(mask_losses)), centerline_loss=float(np.mean(auxiliary_losses)))
        log.append(row)
        print(json.dumps(row), flush=True)
    del Xg, Yg
    if aux_weight:
        del Zh, Z
    torch.cuda.empty_cache()
    torch.save(model.state_dict(), out_dir / f'model_f{fold}.pt')
    res = {}
    if va:
        Xv = np.stack([read_gray(root / 'train/train_images' / f, S) for f in va])
        res['val_files'] = np.array(va)
        res['val_prob'] = predict(model, Xv, cfg['tta'])
    if test_files:
        Xt = np.stack([read_gray(root / 'test/test_images' / f, S) for f in test_files])
        res['test_files'] = np.array(test_files)
        res['test_prob'] = predict(model, Xt, cfg['tta'])
    np.savez_compressed(out_dir / f'probs_f{fold}.npz', **res)
    (out_dir / f'log_f{fold}.json').write_text(json.dumps(dict(cfg=cfg, log=log, secs=time.time() - t0)))
    print(json.dumps(dict(event='fold_done', fold=fold, secs=time.time() - t0)), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fold', type=int)
    p.add_argument('--out', default='/kaggle/working')
    p.add_argument('--config', help='JSON file overriding CFG (see configs/)')
    a = p.parse_args()
    if a.config:
        CFG.update(json.loads(Path(a.config).read_text()))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.fold is not None:
        train_fold(a.fold, CFG, out)
        return
    # launcher: install deps, then run folds in parallel across available GPUs
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch==0.5.0'], check=True)
    import torch
    ngpu = torch.cuda.device_count()
    if ngpu == 0:
        raise RuntimeError('CUDA GPU required; no fold was started')
    print(json.dumps(dict(event='start', gpus=ngpu, cfg=CFG, torch=torch.__version__)), flush=True)
    pending = list(CFG['folds_to_run'])
    if not pending or len(pending) != len(set(pending)):
        raise ValueError('folds_to_run must be nonempty and unique')
    required = lambda f: [out / f'model_f{f}.pt', out / f'probs_f{f}.npz', out / f'log_f{f}.json']
    for f in pending:
        if any(p.exists() for p in required(f)):
            raise FileExistsError(f'Fold {f} already has outputs; use a fresh output directory')
    failures = []
    completed = []
    running = {}
    while pending or running:
        for g in range(max(ngpu, 1)):
            if g not in running and pending:
                f = pending.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
                logf = open(out / f'stdout_f{f}.txt', 'w')
                running[g] = (f, subprocess.Popen([sys.executable, os.path.abspath(__file__), '--fold', str(f), '--out', str(out)] + (['--config', os.path.abspath(a.config)] if a.config else []),
                                                  env=env, stdout=logf, stderr=subprocess.STDOUT), logf)
        time.sleep(20)
        for g in list(running):
            f, proc, logf = running[g]
            if proc.poll() is not None:
                logf.close()
                print(json.dumps(dict(event='proc_exit', fold=f, code=proc.returncode)), flush=True)
                print(open(out / f'stdout_f{f}.txt').read()[-3000:], flush=True)
                missing = [p.name for p in required(f) if not p.is_file() or p.stat().st_size == 0]
                if proc.returncode != 0 or missing:
                    failures.append(dict(fold=f, exit_code=proc.returncode, missing=missing))
                else:
                    completed.append(f)
                del running[g]
    summary = dict(status='FAILED' if failures else 'COMPLETED', completed_folds=completed, failures=failures, cfg=CFG)
    (out / 'RUN_STATUS.json').write_text(json.dumps(summary, indent=2))
    if failures:
        raise RuntimeError(f'Fold execution/output verification failed: {failures}')
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
