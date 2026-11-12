"""Train the stage-2 refiner (src/refine.py) of one fold and predict that fold's seeds.

  python analysis/refine_train.py --bank scratch/ws/refine/bank/oof --fold 0 --out scratch/ws/refine/run1
The refiner for test images, trained on the seeds of all five folds (nothing is held out, only the weights are written):
  python analysis/refine_train.py --bank scratch/ws/refine/bank/oof --fold -1 --out scratch/ws/refine/run1
Apply saved refiners to a bank without training (test seeds; several refiners are averaged). Average exactly
the refiners the paste setting was tuned for (scratch/ws/refine/FINAL_SETTING.json): more refiners give less
extreme probabilities, and the margin of the paste-back depends on that.
  python analysis/refine_train.py --bank scratch/ws/refine/bank/test --apply scratch/ws/refine/run1/refiner_f-1.pt scratch/ws/refine/run_kept/refiner_f0.pt --out scratch/ws/refine/sub/pred_pair

The refiner of fold k learns from the seeds of the other four folds and predicts the seeds of fold k, which it
never saw. All crops come from out-of-fold stage-1 maps, so the refiner is trained on stage-1 output of the same
quality it will meet on test images. Writes refiner_f<k>.pt (float16 state dict), pred_f<k>.npz (fname, label and
the refined uint8 crop of every seed of the fold, in bank order) and log_f<k>.json.

By default every seed of at least 100 px that has a target is a training example, also the small or faint ones
the stage-1 rule filters out; --kept-only trains on the kept seeds alone (the only ones that are pasted back).
Out of fold the two are equally good, and the average of their predictions is better than either.

The whole bank (about 2 GB of uint8 crops) lives on the GPU and is augmented there. After every epoch the state
is saved to resume_f<k>.pt; a run that died (the machine is shared, and three runs were killed by memory shortage)
continues from its last finished epoch when it is started again with the same arguments.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'src'))
import refine

CFG = dict(epochs=30, batch=32, lr=3e-4, wd=1e-4, warmup=200, seed=2026, t_report=0.4)


def to_device(bank_dir, keys, dev):
    """uint8 tensor (N, len(keys), H, W) of the named crops, read one plane at a time to keep host memory low."""
    out = None
    for c, name in enumerate(keys):
        plane = torch.from_numpy(refine.load_bank(bank_dir, [name])[name])
        if out is None:
            out = torch.empty((len(plane), len(keys)) + plane.shape[1:], dtype=torch.uint8, device=dev)
        out[:, c] = plane.to(dev)
    return out


def augment(planes, target, gen):
    """Random variations of a batch. planes: uint8 (B, 5, H, W) in refine.PLANES order, target: uint8 (B, H, W).

    Returns float planes (B, 4, H, W) in [0, 1] and the target (B, 1, H, W).
      - flips: the crop has no preferred left/right or up/down
      - brightness and contrast of the image plane: exposure and seeing differ between observatories
      - the seed drawn 0-2 px fatter or thinner, and in 30 % of the cases cut at another threshold (0.3 to 0.55)
        inside its zone: the refiner must find the outline from the image, not copy the seed's edge
    """
    dev, B = planes.device, len(planes)

    def rand(*shape):
        return torch.rand(*shape, device=dev, generator=gen)

    both = torch.cat([planes, target[:, None]], 1).float() / 255
    both = torch.where((rand(B) < 0.5)[:, None, None, None], both.flip(-1), both)
    both = torch.where((rand(B) < 0.5)[:, None, None, None], both.flip(-2), both)
    image, prob, seed, others, zone, y = both.unbind(1)
    gain = torch.exp((rand(B, 1, 1) * 2 - 1) * math.log(1.25))
    image = ((image - 0.5) * gain + 0.5 + (rand(B, 1, 1) * 2 - 1) * 0.08).clamp(0, 1)
    recut = ((prob >= 0.3 + rand(B, 1, 1) * 0.25) & (zone >= 0.5)).float()
    swap = (rand(B) < 0.3) & (recut.flatten(1).sum(1) > 0)
    seed = torch.where(swap[:, None, None], recut, seed)
    radius = torch.randint(-2, 3, (B,), device=dev, generator=gen)
    out = seed * (radius == 0)[:, None, None]
    for r in (1, 2):
        grown = F.max_pool2d(seed[:, None], 2 * r + 1, 1, r)[:, 0]
        shrunk = -F.max_pool2d(-seed[:, None], 2 * r + 1, 1, r)[:, 0]
        # a seed only a few crop pixels wide would vanish: keep it as it is when more than half would go
        shrunk = torch.where((shrunk.flatten(1).sum(1) >= 0.5 * seed.flatten(1).sum(1))[:, None, None], shrunk, seed)
        out = out + grown * (radius == r)[:, None, None] + shrunk * (radius == -r)[:, None, None]
    return torch.stack([image, prob, out, others], 1), y[:, None]


def loss_fn(logit, y):
    """0.5 BCE + 0.5 soft Dice against the soft target. Dice is taken per crop, then averaged: every filament
    counts the same whatever its size, as in the competition metric."""
    p = logit.sigmoid().flatten(1)
    t = y.flatten(1)
    dice = 1 - (2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)
    return 0.5 * F.binary_cross_entropy_with_logits(logit, y) + 0.5 * dice.mean()


def crop_iou(mask, target):
    """Mean IoU of boolean crops (N, H, W) with the target >= 0.5, and the share above 0.5."""
    t = target >= 128
    inter = (mask & t).flatten(1).sum(1).float()
    iou = inter / (mask | t).flatten(1).sum(1).clamp(min=1)
    return round(float(iou.mean()), 4), round(float((iou > 0.5).float().mean()), 4)


def save_state(path, state):
    """Write the resume file through a temporary file, so that a half-written file never replaces a good one.
    A write that fails is skipped (virus scanners on Windows lock fresh files for a moment): the previous state
    is still valid, and losing one epoch is better than losing the run."""
    tmp = path.with_suffix('.tmp')
    try:
        torch.save(state, tmp)
        os.replace(tmp, path)
    except OSError as e:
        print(json.dumps(dict(event='warning', message=f'resume file not updated: {e}'[:200])), flush=True)


def train(a):
    cfg = dict(CFG, epochs=a.epochs, encoder=a.encoder, kept_only=a.kept_only, seed=a.seed)
    fold = a.fold
    dev = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if dev.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.6)      # the card is shared with other jobs
    t0 = time.time()
    meta = refine.load_bank(a.bank, ['fname', 'label', 'fold', 'n_match', 'kept'])
    held = meta['fold'] == fold
    has_target = meta['n_match'] > 0
    learn = ~held & has_target & (meta['kept'] | (not a.kept_only))
    # no image, and no observation day, of the held-out fold may be among the training seeds
    held_files, learn_files = set(meta['fname'][held]), set(meta['fname'][learn])
    assert not held_files & learn_files, 'held-out images in the training set'
    assert not {f[:8] for f in held_files} & {f[:8] for f in learn_files}, 'held-out observation days in the training set'
    X = to_device(a.bank, refine.PLANES, dev)
    Y = to_device(a.bank, ['target'], dev)[:, 0]
    learn_idx = torch.from_numpy(np.flatnonzero(learn)).to(dev)
    held_idx = torch.from_numpy(np.flatnonzero(held)).to(dev)
    scored = torch.from_numpy(np.flatnonzero(held & has_target & meta['kept'])).to(dev)   # held-out kept seeds with a target
    print(json.dumps(dict(event='data', fold=fold, seeds=len(held), train=len(learn_idx), held=len(held_idx),
                          held_scored=len(scored), train_images=len(learn_files), held_images=len(held_files),
                          device=str(dev), secs=round(time.time() - t0, 1))), flush=True)

    torch.manual_seed(cfg['seed'] + fold)
    gen = torch.Generator(device=dev).manual_seed(cfg['seed'] + fold)
    model = refine.build_refiner(cfg['encoder']).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == 'cuda')
    per_epoch = math.ceil(len(learn_idx) / cfg['batch'])
    total, step, log, first_epoch = per_epoch * cfg['epochs'], 0, [], 0
    a.out.mkdir(parents=True, exist_ok=True)
    resume = a.out / f'resume_f{fold}.pt'
    if resume.exists():
        state = torch.load(resume, map_location=dev, weights_only=True)
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['opt'])
        scaler.load_state_dict(state['scaler'])
        gen.set_state(state['gen'].cpu())
        step, log, first_epoch = state['step'], state['log'], len(state['log'])
        t0 -= log[-1]['secs']                                # keep the clock of the run that was interrupted
        print(json.dumps(dict(event='resumed', fold=fold, after_epoch=first_epoch)), flush=True)
    seed_iou = crop_iou(X[scored, refine.PLANES.index('seed')] >= 128, Y[scored])
    for ep in range(first_epoch, cfg['epochs']):
        model.train()
        perm = learn_idx[torch.randperm(len(learn_idx), device=dev, generator=gen)]
        losses = []
        for i in range(per_epoch):
            idx = perm[i * cfg['batch']:(i + 1) * cfg['batch']]
            if len(idx) < 2:
                continue
            lr = cfg['lr'] * min(1, (step + 1) / cfg['warmup']) * 0.5 * (1 + math.cos(math.pi * step / total))
            for g in opt.param_groups:
                g['lr'] = lr
            x, y = augment(X[idx], Y[idx], gen)
            with torch.autocast(dev.type, enabled=dev.type == 'cuda'):
                logit = model(refine.net_input(x))
            loss = loss_fn(logit.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            losses.append(loss.detach())                     # read once per epoch: every read waits for the GPU
        row = dict(event='epoch', fold=fold, epoch=ep + 1, loss=round(float(torch.stack(losses).mean()), 5), lr=lr,
                   secs=round(time.time() - t0, 1))
        if len(scored) and ((ep + 1) % 5 == 0 or ep + 1 == cfg['epochs']):
            # held-out check in crop space: does the refined outline agree better with the target than the seed did?
            pred = torch.from_numpy(refine.predict_crops(model, X[scored])).to(dev)
            row.update(held_iou_seed=seed_iou, held_iou_refined=crop_iou(pred >= cfg['t_report'] * 255, Y[scored]))
        log.append(row)
        print(json.dumps(row), flush=True)
        save_state(resume, dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
                                gen=gen.get_state(), step=step, log=log))
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, a.out / f'refiner_f{fold}.pt')
    if held.any():
        pred = refine.predict_crops(model, X[held_idx])
        np.savez_compressed(a.out / f'pred_f{fold}.npz', fname=meta['fname'][held], label=meta['label'][held], pred=pred)
    done = dict(event='fold_done', fold=fold, secs=round(time.time() - t0, 1), train_secs=round(log[-1]['secs'], 1),
                gpu_peak_mb=round(torch.cuda.max_memory_allocated() / 2 ** 20) if dev.type == 'cuda' else 0)
    (a.out / f'log_f{fold}.json').write_text(json.dumps(dict(cfg=cfg, log=log, done=done)))
    resume.unlink()
    print(json.dumps(done), flush=True)


def apply(a):
    """Average the outputs of the given refiners over every seed of a bank (each fold of the bank separately)."""
    dev = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if dev.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.6)
    meta = refine.load_bank(a.bank, ['fname', 'label', 'fold'])
    X = to_device(a.bank, refine.PLANES[:refine.N_INPUT], dev)
    acc = np.zeros((len(X), refine.CROP_H, refine.CROP_W), np.float32)
    for path in a.apply:
        acc += refine.predict_crops(refine.load_refiner(path, a.encoder, dev), X)
    pred = np.round(acc / len(a.apply)).astype(np.uint8)
    a.out.mkdir(parents=True, exist_ok=True)
    for fold in np.unique(meta['fold']):
        m = meta['fold'] == fold
        np.savez_compressed(a.out / f'pred_f{fold}.npz', fname=meta['fname'][m], label=meta['label'][m], pred=pred[m])
    print(json.dumps(dict(event='applied', seeds=len(X), refiners=[str(p) for p in a.apply], out=str(a.out))), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--fold', type=int, help='train the refiner of this fold (-1: on all folds, for test images)')
    ap.add_argument('--apply', type=Path, nargs='+', help='refiner_f*.pt files to run on the bank instead of training')
    ap.add_argument('--encoder', default=refine.ENCODER)
    ap.add_argument('--epochs', type=int, default=CFG['epochs'])
    ap.add_argument('--kept-only', action='store_true', help='train on the seeds that pass the stage-1 keep rule only')
    ap.add_argument('--seed', type=int, default=CFG['seed'], help='another seed gives another refiner to average with')
    ap.add_argument('--device', help='cuda or cpu (default: cuda when available)')
    a = ap.parse_args()
    if a.apply:
        apply(a)
    elif a.fold is not None:
        train(a)
    else:
        ap.error('give --fold or --apply')


if __name__ == '__main__':
    main()
