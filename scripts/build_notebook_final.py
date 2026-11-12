"""Write notebooks/end_to_end_final.ipynb, the end-to-end notebook of the final pipeline
(dense maps -> seeds -> learned keep rule -> refined shapes). Edit the cell sources here, then re-run this script.

notebooks/end_to_end.ipynb (scripts/build_notebook.py) stays the notebook of the v2 baseline.
"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebook import VIS, VIS_MD

INTRO = """# Solar filament segmentation: end-to-end pipeline (final)

Team notebook for the Solar Filament Segmentation Challenge 2026. It turns GONG H-alpha test images into the
submission CSV using the files stored in `weights/`, so **no training is needed**. Each image is predicted from
that image alone.

**Pipeline**
1. **Dense map.** Two U-Nets with an EfficientNet-B4 encoder (same recipe, two random seeds, trained on all 707
   training images at 1536 x 1536 with soft targets = the mean of all annotators' masks). Each predicts the image
   8 times (4 rotations x 2 flips); the maps are averaged at 2048 x 2048.
2. **Seeds.** Pixels with probability >= 0.4 are filament pixels. Pieces belong to the same filament when a
   closing of radius 8 connects them or when they are linked through pixels of probability >= 0.1 (annotators
   draw one outline around a filament whose middle is faint). Every group of at least 100 px is a candidate.
3. **Keep rule (learned).** For every candidate, about 50 descriptors are computed from the image and the maps:
   size and shape, position on the disk, how the probability is spread, whether the two models agree, how dark it
   is against its surroundings, and how it compares with the other candidates of the same image. A gradient
   boosting model predicts the IoU the candidate would earn per annotator record (0 if nobody drew it). PQ is a
   ratio, and a candidate raises it exactly when that expected IoU is above PQ / 2, so one threshold decides.
   The model is refitted here from its training table (a few seconds, deterministic).
4. **Shape (refiner).** For every kept candidate, a small U-Net looks at an oriented crop of the image and predicts
   the outline the annotators who drew this filament would draw. Its probability is averaged 50/50 with the dense
   probability and thresholded at 0.5; candidates whose refined masks claim the same pixels are joined.
   `USE_REFINER = False` writes each kept candidate at probability >= 0.48 instead (lighter variant).
5. **Output.** One row per filament, `filament_id = <image>_<n>`, COCO RLE counts. Every pixel has at most one
   label, so **masks never overlap**; the CSV is still checked explicitly.

Run all cells from the repository root or from `notebooks/`. Settings are in the next cell."""

SETTINGS = """from pathlib import Path
import json, os, sys, time

ROOT = next(p for p in [Path.cwd(), Path.cwd().parent] if (p / 'src' / 'predict_final.py').is_file())
sys.path.insert(0, str(ROOT / 'src'))

WEIGHTS = Path(os.environ.get('FILAMENT_WEIGHTS', ROOT / 'weights'))
DENSE = [WEIGHTS / 'model_full_s2026.pt', WEIGHTS / 'model_full_s2027.pt']   # the two dense models
CONFIG = ROOT / 'configs' / 'v2_full.json'                                    # config they were trained with
REFINERS = [WEIGHTS / f'refiner_f{k}.pt' for k in range(5)]                    # averaged
KEEP_TABLE = WEIGHTS / 'keep_table.npz'                                       # training table of the keep model
USE_REFINER = True                                 # False: lighter variant without the refiner networks
TEST_IMAGES = ROOT / 'data' / 'MAGFiLO_1.0_Kaggle_2026' / 'test' / 'test_images'
OUT = ROOT / 'outputs' / 'submission_final.csv'
DEVICE = os.environ.get('FILAMENT_DEVICE')         # None: CUDA if available, else Apple MPS, else CPU
LIMIT = int(os.environ.get('FILAMENT_LIMIT', 0)) or None   # predict only the first N images (quick check)
FP32 = True                                        # float32 on every device, so any machine reproduces the CSV"""

ENV = """from importlib.metadata import version
import warnings
import numpy as np, torch, cv2
import keep, refine
from train_seg import CFG, pick_device
from postproc import upsample
from predict_final import FINAL, dense_maps, image_instances, rows_of
from make_submission import check_disjoint, write_csv

warnings.filterwarnings('ignore', message='invalid value encountered in divide')   # mean of an empty label
device = pick_device(DEVICE)
print(' | '.join(f'{p} {version(p)}' for p in ['torch', 'segmentation-models-pytorch', 'timm', 'scikit-learn', 'numpy',
                                                'opencv-python-headless', 'pycocotools']))
print('device:', device, torch.cuda.get_device_name(0) if device.type == 'cuda' else '')
print('settings:', FINAL, '| keep threshold', keep.TAU)"""

TRAIN_MD = """## How the files in `weights/` were made (not run here)

GPU training is not bit-for-bit deterministic, which is why the trained files are part of the repository.
The commands below reproduce them (RTX 5070 Ti: about 1 h per dense model, 10 min per refiner, minutes for the rest).

```bash
# 1. dense models: five folds twice (out-of-fold maps for stages 3-4), and two models on all images (test maps)
python src/train_seg.py --config configs/v2.json  --out runs/v2            # folds 0, 1
python src/train_seg.py --config configs/v2b.json --out runs/v2b           # folds 2, 3, 4
python src/train_seg.py --config configs/v2.json  --out runs/v2_5070ti     # second set, same recipe
python src/train_seg.py --config configs/v2b.json --out runs/v2b_5070ti
python src/train_seg.py --config configs/v2_full.json       --out runs/v2_full_5070ti        # -> model_full_s2026.pt
python src/train_seg.py --config configs/v2_full_s2027.json --out runs/v2_full_s2027_5070ti  # -> model_full_s2027.pt

# 2. refiners: seeds of the out-of-fold maps, one refiner per fold (trained without that fold) -> refiner_f*.pt
python analysis/refine_bank.py '{"tag": "oof", "members": [{"0": "v2", "1": "v2", "2": "v2b", "3": "v2b", "4": "v2b"}]}'
python analysis/refine_train.py --bank scratch/ws/refine/bank/oof --fold 0 --out scratch/ws/refine/run1   # and folds 1-4

# 3. keep rule: seeds of the two-model out-of-fold maps, their descriptors, cross-fitted check, final table
python analysis/refine_bank.py '{"tag": "kc_bridge", "out": "scratch/ws/main/rbank", "bridge": 0.1, "members": [<set 1>, <set 2>]}'
python analysis/keep_bank.py '{"bank": "scratch/ws/main/rbank/kc_bridge", "out": "scratch/keep/oof", "members": [<set 1>, <set 2>]}'
python analysis/keep_fit.py --tables scratch/keep/oof --bank scratch/ws/main/rbank/kc_bridge --table weights/keep_table.npz
```

Out-of-fold scores of every stage are printed at the end of this notebook; the experiments behind each choice
are in `EXPERIMENTS.md`."""

PRED_MD = """## Predict the test images"""

PRED = """cfg = dict(CFG, **json.loads(CONFIG.read_text()))
files = sorted(p.name for p in TEST_IMAGES.glob('*.jpeg'))[:LIMIT]
assert files, f'no test images in {TEST_IMAGES}'
needed = DENSE + [KEEP_TABLE] + (REFINERS if USE_REFINER else [])
assert all(p.is_file() for p in needed), f'missing: {[str(p) for p in needed if not p.is_file()]}'

t0 = time.time()
maps = dense_maps(DENSE, cfg, TEST_IMAGES, files, device, FP32)                 # stage 1
t_dense = time.time() - t0
keep_model = keep.load_model(KEEP_TABLE)                                        # stage 3 model, refitted from its table
refiners = [refine.load_refiner(p, refine.ENCODER, device) for p in REFINERS] if USE_REFINER else []

rows, empty, counts = [], [], dict(seeds=0, fixed_rule=0, kept=0)
for i, f in enumerate(files):
    gray = cv2.imread(str(TEST_IMAGES / f), cv2.IMREAD_GRAYSCALE)
    labels, info = image_instances([upsample(m[i]) for m in maps], gray, keep_model, refiners, keep.TAU, device,
                                   fp16=not FP32)                               # stages 2-4 for one image
    new = rows_of(labels, f)
    rows += new
    if not new:
        empty.append(f)
    for k in counts:
        counts[k] += info[k]
seconds = time.time() - t0
print(f'{len(files)} images: dense maps {t_dense:.0f} s, seeds + keep rule + shapes {seconds - t_dense:.0f} s '
      f'({seconds / len(files):.1f} s per image on {device})')
print(f"candidates {counts['seeds']}, kept by the learned rule {counts['kept']} "
      f"(the fixed rule would keep {counts['fixed_rule']})")"""

POST_MD = """## CSV and overlap check"""

POSTC = """checked = check_disjoint(rows)          # raises if two masks of one image share a pixel
OUT.parent.mkdir(parents=True, exist_ok=True)
write_csv(rows, OUT)
per_image = np.bincount([files.index(r[0].rsplit('_', 1)[0] + '.jpeg') for r in rows], minlength=len(files))
print(f'{len(rows)} filaments in {len(files)} images (median {np.median(per_image):.0f}, max {per_image.max()} per image); '
      f'{len(empty)} images without filaments; no overlaps in {checked} images')
print('written:', OUT)"""

VAL_MD = """## Local validation (out-of-fold)

Scores on held-out training images with the official self-evaluation rule (`src/common.py`): every annotator
record is scored separately, and the counts are pooled. All maps are out of fold; the refiner and the keep model
that process an image were trained without that image's fold; the keep threshold is chosen per fold on the other
four folds."""

VAL = """oof = json.loads((ROOT / 'results' / 'evidence' / 'final_sprint_1112' / 'SUMMARY.json').read_text())
print(f"{oof['images']} images, {oof['records']} annotator records")
for s in oof['stages']:
    pq = s.get('pq', s.get('pq_nested'))
    folds = s.get('per_fold', s.get('per_fold_nested'))
    print(f"  PQ {pq:.4f}  {s['name']}")
    print('           per fold: ' + ' '.join(f'{x:.4f}' for x in folds))"""


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_code_cell(SETTINGS), nbf.v4.new_code_cell(ENV),
                nbf.v4.new_markdown_cell(TRAIN_MD),
                nbf.v4.new_markdown_cell(PRED_MD), nbf.v4.new_code_cell(PRED),
                nbf.v4.new_markdown_cell(POST_MD), nbf.v4.new_code_cell(POSTC),
                nbf.v4.new_markdown_cell(VIS_MD), nbf.v4.new_code_cell(VIS),
                nbf.v4.new_markdown_cell(VAL_MD), nbf.v4.new_code_cell(VAL)]
    nb.metadata['kernelspec'] = dict(name='python3', display_name='Python 3', language='python')
    nbf.write(nb, ROOT / 'notebooks' / 'end_to_end_final.ipynb')
    print('written notebooks/end_to_end_final.ipynb')


if __name__ == '__main__':
    main()
