"""Time the v2 training step and 8-way TTA inference on this machine.

Synthetic data with v2's shapes (1536 input, 1024 crop, batch 4, EfficientNet-B4 U-Net); speed does
not depend on image content, so no competition data or pretrained download is needed.

usage: python analysis/bench_device.py [steps] [batch]
On Apple MPS set PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 so an oversized batch fails fast instead of swapping.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
sys.path.insert(0, str(ROOT / 'src'))
import train_seg as ts  # noqa: E402

cfg = {**ts.CFG, **json.loads((ROOT / 'configs/v2.json').read_text())}
if len(sys.argv) > 2:
    cfg['batch'] = int(sys.argv[2])
dev = ts.pick_device().type
sync = {'cuda': torch.cuda.synchronize, 'mps': torch.mps.synchronize}.get(dev, lambda: None)
S, B = cfg['size'], cfg['batch']
TRAIN = 565  # training images per fold, as in the Kaggle logs; train_seg keeps them all on the device at 1536
rng = np.random.default_rng(0)
Xg = torch.from_numpy(rng.integers(0, 256, (TRAIN, S, S), dtype=np.uint8)).to(dev)
Yg = torch.from_numpy(rng.integers(0, 256, (TRAIN, S, S), dtype=np.uint8)).to(dev)
gen = torch.Generator(device=dev).manual_seed(0)
model = ts.build_model(cfg['encoder'], weights=None).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
scaler = torch.amp.GradScaler(dev, enabled=dev != 'cpu')


def step():
    """Same body as one update in train_seg.train_fold."""
    model.train()
    idx = torch.randperm(len(Xg), device=dev, generator=gen)[:B]
    img, tgt = Xg[idx].float().div(255).unsqueeze(1), Yg[idx].float().div(255).unsqueeze(1)
    xi, yt = ts.augment(img, tgt, cfg['crop'], gen)
    with torch.autocast(dev, dtype=ts.amp_dtype(dev)):
        logit = model(ts.to_input(xi))
    logit = logit.float()
    p = logit.sigmoid()
    loss = F.binary_cross_entropy_with_logits(logit, yt) + 1 - (2 * (p * yt).sum() + 1) / (p.sum() + yt.sum() + 1)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    scaler.step(opt)
    scaler.update()
    return float(loss)  # train_seg also pulls the loss to the host every step


for _ in range(5):
    step()
sync()
t = time.perf_counter()
losses = [step() for _ in range(STEPS)]
sync()
per_step = (time.perf_counter() - t) / STEPS
assert np.isfinite(losses).all(), losses

imgs = rng.integers(0, 256, (8, S, S), dtype=np.uint8)
ts.predict(model, imgs[:2], cfg['tta'])
t = time.perf_counter()
ts.predict(model, imgs, cfg['tta'])
per_img = (time.perf_counter() - t) / len(imgs)

peak_gb = {'mps': lambda: torch.mps.driver_allocated_memory(),
           'cuda': lambda: torch.cuda.max_memory_reserved()}.get(dev, lambda: 0)() / 2**30
steps_per_fold = -(-TRAIN // B) * cfg['epochs']
infer_imgs = 142 + 180  # held-out fold + test set
fold_s = steps_per_fold * per_step + infer_imgs * per_img
print(json.dumps(dict(device=dev, torch=torch.__version__, batch=B, device_mem_gb=round(peak_gb, 1),
                      per_step_s=round(per_step, 3), tta8_per_image_s=round(per_img, 2),
                      est_fold_hours=round(fold_s / 3600, 2), est_5fold_hours=round(5 * fold_s / 3600, 1),
                      loss_first_last=[round(losses[0], 3), round(losses[-1], 3)]), indent=1))
