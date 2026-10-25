"""CPU checks for experiment B: per-filament weights, small-filament centres and centred crops."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from train_seg import augment, small_filament_maps

# One annotator, two square filaments: small (about 400 px) and large (about 3600 px).
small = [100, 200, 120, 200, 120, 220, 100, 220]
large = [1000, 1000, 1060, 1000, 1060, 1060, 1000, 1060]
anns = {7: [[small], [large]]}
extra, centres = small_filament_maps([7], anns, 2048, area_ref=2000, w_max=4.0, small_max=1000)
w_small = 1 + int(extra[210, 110]) * 3 / 255
w_large = 1 + int(extra[1030, 1030]) * 3 / 255
assert extra[0, 0] == 0 and extra[500, 500] == 0, 'background must keep weight 1'
assert 2.0 < w_small < 2.4, w_small            # sqrt(2000 / ~400) ~ 2.2
assert w_large == 1.0, w_large                  # larger than area_ref: weight 1
assert len(centres) == 1                        # only the small filament is a crop target
cx, cy = centres[0]
assert abs((cx + 1) * 1024 - .5 - 110) < 1.5 and abs((cy + 1) * 1024 - .5 - 210) < 1.5, centres
off, _ = small_filament_maps([7], anns, 2048, area_ref=0, w_max=4.0, small_max=1000)
assert off.max() == 0, 'area_ref=0 must switch weighting off'

# A 4x4 blob near the right edge of a 256 image; crops of 128 centred on it must always contain it.
img = torch.zeros(1, 1, 256, 256)
img[0, 0, 38:42, 228:232] = 1
centre = torch.tensor([[(229.5 + .5) / 128 - 1, (39.5 + .5) / 128 - 1]])
seen = 0
for seed in range(200):
    g = torch.Generator().manual_seed(seed)
    _, t = augment(img, img.clone(), 128, g, centre)
    seen += int(t.max() > .2)
assert seen == 200, seen
random_hits = sum(int(augment(img, img.clone(), 128, torch.Generator().manual_seed(s))[1].max() > .2) for s in range(200))

# A NaN centre leaves the random crop unchanged (same random draws as the baseline path).
a = augment(img, img.clone(), 128, torch.Generator().manual_seed(3))[1]
b = augment(img, img.clone(), 128, torch.Generator().manual_seed(3), torch.full((1, 2), float('nan')))[1]
assert torch.equal(a, b)

# Extra target channels go through the same geometry as the target.
two = torch.cat([img, img * 2], 1)
_, t2 = augment(img, two, 128, torch.Generator().manual_seed(5), centre)
assert torch.allclose(t2[:, 1], t2[:, 0] * 2, atol=1e-6)

result = dict(small_weight=round(float(w_small), 3), large_weight=float(w_large), centred_crops_with_blob=f'{seen}/200',
              random_crops_with_blob=f'{random_hits}/200', nan_centre_matches_baseline=True, channels_share_geometry=True)
(ROOT / 'scratch').mkdir(exist_ok=True)
(ROOT / 'scratch' / 'SMALL_FILAMENT_TEST_RESULTS.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
