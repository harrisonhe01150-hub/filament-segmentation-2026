"""CPU self-test of the learned keep rule (src/keep.py) on a synthetic image. No data, no GPU, no network.

Checks: the descriptors are finite, deterministic and do not change when annotations are passed to the seed bank
(the keep rule must not see them); the labels follow the definition; the table round trip refits the same model;
src/predict_final.image_instances returns disjoint instances and drops what the model scores low.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import keep
import refine
from predict_final import image_instances

rng = np.random.default_rng(0)
H, W = refine.IMAGE_SHAPE
yy, xx = np.mgrid[0:H, 0:W]


def blob(cx, cy, a, b, angle, peak):
    """an elongated Gaussian bump (a filament-like ridge) with the given peak probability"""
    c, s = np.cos(angle), np.sin(angle)
    u, v = (xx - cx) * c + (yy - cy) * s, -(xx - cx) * s + (yy - cy) * c
    return peak * np.exp(-0.5 * ((u / a) ** 2 + (v / b) ** 2))


shapes = [(500, 600, 60, 9, 0.3, 1.0), (1200, 900, 35, 7, 1.2, 0.95), (1500, 1500, 18, 6, 2.0, 0.7), (800, 1400, 90, 10, 0.0, 1.0),
          (300, 1700, 12, 5, 0.5, 0.6)]
prob = np.zeros((H, W), np.float32)
for sh in shapes:
    prob = np.maximum(prob, blob(*sh))
members = [np.round(np.clip(prob * f, 0, 1) * 255).astype(np.uint8) for f in (1.0, 0.94)]
avg = np.mean([m.astype(np.float32) for m in members], 0)
gray = np.clip(140 - 60 * prob + rng.normal(0, 3, (H, W)), 0, 255).astype(np.uint8)

seeds = refine.image_bank(avg, gray, None, bridge=0.1)
n = len(seeds['label'])
assert n >= 4, n
X = keep.image_descriptors(seeds, members)
assert X.shape == (n, len(keep.NAMES)) and np.isfinite(X).all()
assert np.array_equal(X, keep.image_descriptors(seeds, members)), 'descriptors are not deterministic'
col = {name: i for i, name in enumerate(keep.NAMES)}
assert np.allclose(X[:, col['area']], seeds['area']) and np.all(X[:, col['n_seeds']] == n)
assert np.all(X[:, col['mem_a']] >= X[:, col['mem_b']]), 'the second member was scaled down'
assert np.all((X[:, col['rank_mean']] >= 0) & (X[:, col['rank_mean']] < 1))

# annotations must not reach the descriptors: the same seeds and descriptors with and without them
fake = [('r1', [p.astype(np.int32) for p in seeds['pix'][:2]]), ('r2', [seeds['pix'][0].astype(np.int32)])]
with_gt = refine.image_bank(avg, gray, fake, bridge=0.1)
assert np.array_equal(with_gt['label'], seeds['label'])
assert np.array_equal(keep.image_descriptors(with_gt, members), X), 'descriptors changed when annotations were given'

# labels: u = mean over records of the matched IoU, a match needs IoU > 0.5
u, R, s, tp = keep.target(np.array([[0.8, 0.4, -1], [0.6, 0.7, 0.9], [0.5, -1, -1]]))
assert np.allclose(u, [0.4, (0.6 + 0.7 + 0.9) / 3, 0.0]) and list(R) == [2, 3, 1] and list(tp) == [1, 3, 0]
u_img, R_img, _, _ = keep.target(with_gt['best_iou'])
assert R_img.tolist() == [2] * n and u_img[0] == 1.0 and u_img[1] == 0.5 and np.all(u_img[2:] == 0)

# a table of made-up seeds whose value depends on two descriptors: the refitted model must reproduce itself
Xt = rng.normal(size=(600, len(keep.NAMES))).astype(np.float32)
ut = (1 / (1 + np.exp(-(Xt[:, col['mean']] + 0.5 * Xt[:, col['rank_mean']])))).astype(np.float32)
Rt = rng.integers(1, 4, 600)
with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as tmp:
    keep.save_table(Path(tmp) / 'table.npz', Xt, ut, Rt)
    a, b = keep.load_model(Path(tmp) / 'table.npz'), keep.load_model(Path(tmp) / 'table.npz')
pa = a.predict(Xt)
assert np.array_equal(pa, b.predict(Xt)), 'refitting the table gave another model'
assert np.corrcoef(pa, ut)[0, 1] > 0.9


class ByArea:
    """stand-in for the keep model: scores a seed by its area, so the threshold has a known effect"""

    def predict(self, X):
        return X[:, col['area']] / 1000.0


labels, info = image_instances(members, gray, ByArea(), [], tau=0.5)       # lighter variant, no refiner
big = int((seeds['area'] > 500).sum())
assert info['seeds'] == n and info['kept'] == big and labels.max() == big, (info, big)
assert set(np.unique(labels)) == set(range(big + 1))
written = int(np.count_nonzero(labels))
assert written <= sum(int(a) for a in seeds['area'] if a > 500), 'the draw level may only shrink the stage-1 masks'
labels_none, info_none = image_instances(members, gray, ByArea(), [], tau=1e9)
assert labels_none.max() == 0 and info_none['kept'] == 0

result = dict(seeds=n, descriptors=len(keep.NAMES), annotations_change_descriptors=False, table_round_trip_identical=True,
              kept_by_area_rule=big, pixels_written=written)
(ROOT / 'scratch').mkdir(exist_ok=True)
(ROOT / 'scratch' / 'KEEP_RULE_TEST_RESULTS.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
