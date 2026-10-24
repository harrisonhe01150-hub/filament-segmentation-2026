"""Write notebooks/end_to_end.ipynb from the cell sources below (edit here, then re-run this script)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

INTRO = """# Solar filament segmentation: end-to-end pipeline

Team notebook for the Solar Filament Segmentation Challenge 2026. It turns GONG H-alpha test images into the
submission CSV using the model weights stored in this repository, so **no training is needed**.

**Pipeline**
1. **Model:** U-Net with an EfficientNet-B4 encoder (ImageNet start), one output channel: the probability that a
   pixel belongs to a filament. Trained at 1536 x 1536 on 1024 x 1024 crops, with soft targets (the mean of all
   annotators' masks for the photo), BCE + soft Dice loss, 50 epochs per fold. Folds are grouped by observation day.
2. **Prediction:** every fold model predicts each test image 8 times (4 rotations x 2 flips, each undone afterwards)
   and the maps are averaged; then the fold models are averaged at the full 2048 x 2048 resolution.
3. **Instances:** pixels with probability >= 0.4 form the mask. A morphological closing (radius 8) is used only to
   decide which nearby pieces belong to the same filament; each group of connected pixels is one filament. Groups
   smaller than 300 px or with mean probability below 0.85 are dropped. Because every pixel gets at most one label,
   **masks never overlap**; the CSV is still checked explicitly.
4. **Output:** one row per filament, `filament_id = <image>_<n>`, COCO RLE counts.

Run all cells from the repository root or from `notebooks/`. Settings are in the next cell."""

SETTINGS = """from pathlib import Path
import json, os, sys, time

ROOT = next(p for p in [Path.cwd(), Path.cwd().parent] if (p / 'src' / 'predict.py').is_file())
sys.path.insert(0, str(ROOT / 'src'))

WEIGHTS = Path(os.environ.get('FILAMENT_WEIGHTS', ROOT / 'weights'))   # model_f*.pt files
CONFIG = ROOT / 'configs' / 'v2.json'                                     # config the weights were trained with
TEST_IMAGES = ROOT / 'data' / 'MAGFiLO_1.0_Kaggle_2026' / 'test' / 'test_images'
OUT = ROOT / 'outputs' / 'submission.csv'
DEVICE = os.environ.get('FILAMENT_DEVICE')         # None: CUDA if available, else Apple MPS, else CPU
LIMIT = int(os.environ.get('FILAMENT_LIMIT', 0)) or None   # predict only the first N images (quick check)
RUN_TRAINING = False                               # True: retrain all folds first (CUDA GPU, ~1 h per fold)"""

ENV = """from importlib.metadata import version
import numpy as np, torch, cv2
from train_seg import CFG, pick_device, predict, read_gray
from predict import POST, load_model
from make_submission import build_rows, check_disjoint, write_csv

device = pick_device(DEVICE)
print(' | '.join(f'{p} {version(p)}' for p in ['torch', 'segmentation-models-pytorch', 'timm', 'numpy',
                                                'opencv-python-headless', 'pycocotools']))
print('device:', device, torch.cuda.get_device_name(0) if device.type == 'cuda' else '')"""

TRAIN_MD = """## Optional: retrain

The weights in `weights/` were produced by `src/train_seg.py` with the config above (one model per fold). Set
`RUN_TRAINING = True` to reproduce them; retrained weights will differ slightly because GPU training is not
bit-for-bit deterministic."""

TRAIN = """if RUN_TRAINING:
    import subprocess
    WEIGHTS = ROOT / 'runs' / 'notebook_training'
    subprocess.run([sys.executable, str(ROOT / 'src' / 'train_seg.py'), '--config', str(CONFIG), '--out', str(WEIGHTS)],
                   cwd=ROOT, check=True)"""

PRED_MD = """## Predict the test images"""

PRED = """cfg = dict(CFG, **json.loads(CONFIG.read_text()))
paths = sorted(WEIGHTS.glob('model_f*.pt'))
assert paths, f'no model_f*.pt in {WEIGHTS}'
files = sorted(p.name for p in TEST_IMAGES.glob('*.jpeg'))[:LIMIT]
assert files, f'no test images in {TEST_IMAGES}'

t0 = time.time()
imgs = np.stack([read_gray(TEST_IMAGES / f, cfg['size']) for f in files])
stacks = []
for path in paths:
    model = load_model(path, cfg['encoder'], device)
    stacks.append(predict(model, imgs, cfg['tta']))
    del model
seconds = time.time() - t0
print(f'{len(paths)} fold models x {cfg["tta"]} TTA views on {len(files)} images: {seconds:.0f} s '
      f'({seconds / len(files):.1f} s per image on {device})')"""

POST_MD = """## Instances, CSV and overlap check"""

POSTC = """rows, empty = build_rows(files, stacks, POST)
checked = check_disjoint(rows)          # raises if two masks of one image share a pixel
OUT.parent.mkdir(parents=True, exist_ok=True)
write_csv(rows, OUT)
per_image = np.bincount([files.index(r[0].rsplit('_', 1)[0] + '.jpeg') for r in rows], minlength=len(files))
print(f'{len(rows)} filaments in {len(files)} images (median {np.median(per_image):.0f}, max {per_image.max()} per image); '
      f'{len(empty)} images without filaments; no overlaps in {checked} images')
print('written:', OUT)"""

VIS_MD = """## What the masks look like

Each colour is one predicted filament. Right: the averaged probability map (brighter = more likely filament)."""

VIS = """import matplotlib.pyplot as plt
from pycocotools import mask as mu

def label_map(image):
    stem = image.rsplit('.', 1)[0]
    lab = np.zeros((2048, 2048), np.int32)
    for n, (fid, counts) in enumerate([r for r in rows if r[0].rsplit('_', 1)[0] == stem], 1):
        lab[mu.decode({'size': [2048, 2048], 'counts': counts.encode()}).astype(bool)] = n
    return lab

show = [files[i] for i in np.argsort(-per_image)[:3]]
fig, axes = plt.subplots(len(show), 2, figsize=(12, 6 * len(show)))
rng = np.random.default_rng(0)
for ax, image in zip(np.atleast_2d(axes), show):
    i = files.index(image)
    gray = cv2.imread(str(TEST_IMAGES / image), cv2.IMREAD_GRAYSCALE)
    lab = label_map(image)
    colours = rng.uniform(.2, 1, (lab.max() + 1, 3)); colours[0] = 0
    overlay = np.where(lab[..., None] > 0, .45 * np.stack([gray / 255] * 3, -1) + .55 * colours[lab], np.stack([gray / 255] * 3, -1))
    ax[0].imshow(overlay); ax[0].set_title(f'{image}: {lab.max()} filaments'); ax[0].axis('off')
    prob = np.mean([cv2.resize(s[i], (2048, 2048), interpolation=cv2.INTER_LINEAR) for s in stacks], 0)
    ax[1].imshow(prob, cmap='magma', vmin=0, vmax=255); ax[1].set_title('probability'); ax[1].axis('off')
plt.tight_layout()"""

VAL_MD = """## Local validation (out-of-fold)

Scores on held-out training images, computed with the official self-evaluation notebook's PQ rules
(`src/common.py`). Per-experiment results are in `EXPERIMENTS.md`; re-score a run with
`python analysis/score_folds.py --run <run folder> --folds 0 1`."""

VAL = """oof = json.loads((ROOT / 'results' / 'evidence' / 'full_v2_oof' / 'SUMMARY.json').read_text())
print(f"v2, five folds, {oof['images']} images: PQ {oof['pq']:.4f}  (TP {oof['tp']}, FP {oof['fp']}, FN {oof['fn']}; "
      f"mean IoU of matches {oof['matched_iou_sum'] / oof['tp']:.3f})")
for f in oof['folds']:
    print(f"  fold {f['fold']}: PQ {f['pq']:.4f} on {f['images']} images")"""


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_code_cell(SETTINGS), nbf.v4.new_code_cell(ENV),
                nbf.v4.new_markdown_cell(TRAIN_MD), nbf.v4.new_code_cell(TRAIN),
                nbf.v4.new_markdown_cell(PRED_MD), nbf.v4.new_code_cell(PRED),
                nbf.v4.new_markdown_cell(POST_MD), nbf.v4.new_code_cell(POSTC),
                nbf.v4.new_markdown_cell(VIS_MD), nbf.v4.new_code_cell(VIS),
                nbf.v4.new_markdown_cell(VAL_MD), nbf.v4.new_code_cell(VAL)]
    nb.metadata['kernelspec'] = dict(name='python3', display_name='Python 3', language='python')
    nbf.write(nb, ROOT / 'notebooks' / 'end_to_end.ipynb')
    print('written notebooks/end_to_end.ipynb')


if __name__ == '__main__':
    main()
