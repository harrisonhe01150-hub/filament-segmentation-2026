"""A1 real-size backward check before preparing data; requires one CUDA GPU."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from train_seg import augment, build_model, to_input
from centerline import centerline_loss

torch.manual_seed(2026)
gen = torch.Generator(device='cuda').manual_seed(2026)
model = build_model('efficientnet-b4', weights=None, classes=2).cuda().train()
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
scaler = torch.amp.GradScaler('cuda')
torch.cuda.reset_peak_memory_stats()
losses = []
for step in range(3):
    # Actual A1 caches full datasets on the host. These are the per-batch GPU tensors.
    img = torch.full((4, 1, 1536, 1536), .5, device='cuda')
    target = torch.zeros_like(img); target[:, :, 700:740, 300:1200] = 1
    line = torch.zeros_like(img); line[:, :, 719:722, 300:1200] = 1
    xi, both = augment(img, torch.cat([target, line], 1), 1024, gen)
    with torch.autocast('cuda', dtype=torch.float16):
        logits = model(to_input(xi))
    mask_loss = centerline_loss(logits[:, :1].float(), both[:, :1])
    auxiliary_loss = centerline_loss(logits[:, 1:2].float(), both[:, 1:2])
    loss = mask_loss + .25 * auxiliary_loss
    assert torch.isfinite(loss), loss
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    scaler.step(optimizer); scaler.update()
    losses.append(float(loss.detach()))
print(json.dumps({'event':'a1_full_batch_backward_passed','steps':3,'batch':4,'crop':1024,
                  'losses':losses,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                  'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}),flush=True)
