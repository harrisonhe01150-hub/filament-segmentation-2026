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
| `src/` | Reusable code (data loading, model, post-processing, RLE encoding) |
| `notebooks/` | Exploration, plus the final end-to-end pipeline notebook |
| `report/` | Technical report source |
| `data/` | Competition data, local only (not committed) |

## Rules worth remembering

- Team size: up to 5. Submissions: up to 5 per day.
- External data and pretrained models are allowed; at inference time only the provided test H-alpha images
  may be used, and no extra ground-truth metadata may be used for training.
- Competition data is for non-commercial and academic use only.

## License

MIT (see `LICENSE`). Winning solutions must be released under an OSI-approved license that allows
commercial use.
