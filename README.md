# Solar Filament Segmentation Challenge 2026

Team repository for the Kaggle [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026/overview)
(NSF National Solar Observatory, IEEE BigData Cup 2026).

**Task:** produce one segmentation mask per solar filament in GONG H-alpha full-disk images (2048 x 2048),
using the MAGFiLO ground truth.

## Key dates

| Date | What |
|---|---|
| 2026-11-15, 11:59 pm CDT | Final deadline: technical report + code repository + notebook (Google form linked on the Overview page) |
| 2026-11-30 | Winners announced |
| 2026-12-14 to 17 | IEEE BigData 2026, Phoenix, AZ (optional) |

## Final submission checklist

A Kaggle leaderboard submission alone is not enough. Before the deadline:

- [ ] 4-page technical report (PDF, organizers' Overleaf template)
- [ ] This repository made **public** at competition close, and kept public until winners are announced
- [ ] `requirements.txt` with every package and its exact version
- [ ] One Jupyter notebook that runs the whole pipeline end to end, with no extra files needed
- [ ] Google form filled in (linked on the Overview page)
- [ ] Up to 2 final Kaggle submissions selected

## How we are scored

- **70% quantitative:** Panoptic Quality (PQ) on the leaderboard, plus the distributions of Dice and IoU
  scores, and how often one filament is split into several predictions (one-to-many) or several filaments
  are merged into one (many-to-one).
- **30% qualitative:** clarity of the pipeline description, how the predicted masks look on the images,
  and code quality (modularity and documentation).

PQ = (sum of IoU over matched pairs) / (TP + 0.5 FP + 0.5 FN). Every spurious blob (FP) and every missed
filament (FN) pulls the score down, so getting the *number* of filaments right matters as much as
getting their edges right.

Official self-evaluation notebook: https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook

## Submission format

One CSV for the whole test set, one row per predicted filament:

| column | content |
|---|---|
| `filament_id` | `<image_id>_<n>`, e.g. `20150125172714Mh_1` |
| `segmentation_rle` | COCO RLE counts (from `pycocotools`), no quotes |

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Training (`train_seg.py`), prediction from saved weights (`predict.py`), PQ metric, post-processing, CSV export |
| `configs/` | Training configs (`v2.json` = folds 0/1, `v2b.json` = folds 2/3/4) |
| `notebooks/` | End-to-end notebook |
| `analysis/`, `tests/` | Five-fold OOF re-scoring; CPU self-tests |
| `results/evidence/` | Small summaries of past runs and leaderboard submissions |
| `scripts/` | `update_manifest.py`: run after changing code (the notebook checks file hashes) |
| `weights/` | Final model weights only (added at the model freeze) |
| `report/` | Technical report source |
| `data/` | Competition data, local only (not committed) |

## Setup

Python 3.12. PyTorch comes from the PyTorch index (CUDA 12.8 build shown; drop `--index-url` for CPU or Apple):

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements.txt
```

(On macOS/Linux use `.venv/bin/python`.) Download the competition data with your own Kaggle account and unzip it so
that `data/MAGFiLO_1.0_Kaggle_2026/train/` and `data/MAGFiLO_1.0_Kaggle_2026/test/` exist.

Train (CUDA only), then predict from saved weights (CUDA, MPS or CPU):

```bash
python src/train_seg.py --config configs/v2.json --out runs/v2
python src/predict.py --weights runs/v2 --config configs/v2.json --fp32 --out outputs/submission.csv
```

## Rules worth remembering

- Team size: up to 5. Submissions: up to 5 per day.
- External data and pretrained models are allowed; at inference time only the provided test H-alpha images
  may be used, and no extra ground-truth metadata may be used for training.
- Competition data is for non-commercial and academic use only.

## License

MIT (see `LICENSE`). Winning solutions must be released under an OSI-approved license that allows
commercial use.
