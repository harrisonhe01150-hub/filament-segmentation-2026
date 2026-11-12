# Out-of-fold predictions of the final pipeline (707 training images)

Made 2026-11-12 by `analysis/export_oof_final.py` on Harrison's machine, from the seed bank, the cross-fitted keep
scores and the refiner predictions that EXPERIMENTS.md ("K1", strict nesting) is computed from. Use these files for
the report figures and the error analysis; the 2 GB seed bank is not needed for that.

Everything is out of fold. The keep model of fold k is trained on the other four folds. Its threshold tau is the one an
inner cross-fit picked without fold k (0.22, 0.25, 0.25, 0.21, 0.23 for folds 0–4). The refiner of fold k never saw
fold k. The stage-1 maps are the average of the two out-of-fold model sets (`runs/v2*`, `runs/v2*_5070ti`).

| File | Content |
|---|---|
| `oof_full.csv` | **full pipeline** (learned keep rule + refiner outlines), 4489 filaments, submission format |
| `oof_light.csv` | **light variant** (learned keep rule + stage-1 mask drawn where the probability is at least 0.48), 4490 filaments |
| `oof_images.csv` | one row per image: `fname`, `fold`, `records` (annotator records), `tau`, `candidates`, `kept`, and `tp`, `fp`, `fn`, `iou_sum` of both variants |
| `oof_candidates.csv` | one row per candidate (stage-1 seed of at least 100 px, 6788 in all), see below |
| `SUMMARY.json` | pooled and per-fold PQ of both variants, thresholds, settings |

Submission format: `filament_id` = `<image file name without extension>_<n>`, `segmentation_rle` = COCO RLE counts of the
2048 × 2048 mask; decode with `pycocotools.mask.decode({'size': [2048, 2048], 'counts': s.encode()})`. Images with no
filament have no row. Masks of one image do not overlap.

`oof_candidates.csv` columns: `fname`, `fold`, `label` (candidate number in the bank), `area` (px), `mean` (mean
probability), `u` (cross-fitted keep score, the predicted mean IoU-if-matched), `tau`, `keep` (1 if `u > tau`),
`fixed_rule` (1 if the v2 rule keeps it: area ≥ 300 and mean ≥ 0.85), `iou_r1`–`iou_r3` (best IoU of its **stage-1** mask
with a filament of annotator record 1–3; empty when the image has fewer records), `rle` (its stage-1 mask). Candidate
numbers are not the `_<n>` of the prediction files: kept candidates are renumbered per image, and the refiner can join two.

## Scores

| | PQ | TP | FP | FN | per fold |
|---|---|---|---|---|---|
| full | 0.4566 | 5197 | 2130 | 3002 | 0.4485 / 0.4619 / 0.4568 / 0.4623 / 0.4532 |
| light | 0.4537 | 5180 | 2150 | 3019 | 0.4441 / 0.4609 / 0.4545 / 0.4599 / 0.4490 |

Organisers' rule (every annotator record scored separately, pooled; all pairs with IoU > 0.5 are matches). Both rows
equal EXPERIMENTS.md. They were checked a second way: decoding the CSVs and scoring them against the TRAIN annotations
read from the COCO file (`common.load_coco`, `ann_mask`, `iou_from_labels`) gives the same TP / FP / FN.

Per-image PQ = `iou_sum / (tp + fp/2 + fn/2)` from `oof_images.csv`.

## For the report figures (`analysis/report_figs.py` draws them for v2)

- IoU / Dice of overlapping pairs, 1:n (split) and n:1 (merged) counts: decode `oof_full.csv` and compare each image
  with each annotator record of the TRAIN annotations.
- Why ground-truth filaments are missed: a filament with no prediction at IoU > 0.5 that overlaps no prediction but
  overlaps a candidate with `keep = 0` was removed by the keep rule; one that overlaps no candidate was not detected at
  all; the rest overlap a prediction and are outline errors (split, merged, too large, too small).
- Good and poor examples: per-image PQ from `oof_images.csv`, outlines from `oof_full.csv`.

Regenerate (needs the seed bank and model runs on Harrison's machine, see HANDOFF.md section 5.3):

    python analysis/export_oof_final.py --scores scratch/keep/oof/scores.npz --bank scratch/ws/main/rbank/kc_bridge \
        --pred scratch/ws/main/kcb/pred --fit scratch/keep/oof/fit_result.json --out results/oof_final \
        --members '[{"0": "v2", "1": "v2", "2": "v2b", "3": "v2b", "4": "v2b"}, {"0": "v2_5070ti", "1": "v2_5070ti", "2": "v2b_5070ti", "3": "v2b_5070ti", "4": "v2b_5070ti"}]'
