# K1 audit: the learned keep rule

## Verdict
The out-of-fold gain is real; I could not refute it. Every reported number reproduces exactly, no annotation reaches an image's descriptors or its keep decision, the fast formula agrees with real label maps, and every null test comes out at zero or below.

The size is overstated by about 0.001. +0.0082 is what this split gives when every fold picks tau 0.25. Strictly nested it is **+0.0072**, at the threshold fixed in advance (PQ/2 = 0.2245) it is +0.0067, and over 10 other day-grouped splits the same procedure averages +0.0074. Best estimate **+0.007**, plausible range +0.005 to +0.010.

With refined shapes I get 0.4560 vs 0.4490 (+0.0070), as reported; strictly nested +0.0069.

Two limits on how far it carries:
- **2018-2022 images:** no gain, probably a loss (-0.0087, 5-95% -0.0210..+0.0027).
- **Single-record images:** about half the gain (+0.0040), and none on single-model seeds.

Terms used below:
- **as claimed**: tau per fold chosen on the other four folds from the same cross-fitted scores (what keep_fit2.py does).
- **strictly nested**: tau for fold k chosen from models that never saw fold k (inner cross-fit).
- **fixed 0.2245**: tau set to PQ/2 in advance, no selection.

## 1. Leakage: none found
- **(a) Descriptors without annotations.** I rebuilt the bank with `refine.image_bank(prob, gray, None)` plus the bridge patch for all 707 images in all five folds, then recomputed the 50 descriptors. Result: 6788 seeds, 339,400 descriptor cells, 0 differ (bitwise). Every bank field the descriptors use is identical too. The rebuild script never opens an annotation file.
- **(b) Folds.**
  - The K and C `val_files` are identical per fold, cover the 707 images once, and equal `train_seg.fold_of_files` (seed 2026) and `image_folds.pkl`.
  - There are 656 observation days and none is in two folds.
  - All 10 stage-1 runs trained on 565/566 images, with no pseudo-labels in their configs.
  - The bitwise rebuild used the `val_prob` maps of those files, so the stored seeds do come from out-of-fold maps.
- **(c) Cross-fitting.** `keep_fit2.py:131-133` trains on `fold != k` only. My own cross-fit reproduces the script's 6788 scores bitwise; the main session's saved scores differ by at most 1.1e-16, with 0 decisions changed.
- **Only coupling found:** the threshold choice is not strictly nested (first entry in `problems`).

## 2. Evaluation (707 images, real label maps through `postproc.score_file`)
| Stage-1 masks | PQ | TP / FP / FN | vs fixed rule | folds better |
|---|---|---|---|---|
| fixed rule | 0.4439 | 5268 / 2601 / 2931 | | |
| learned, tau 0.25 (as claimed) | 0.4521 | 5098 / 2016 / 3101 | +0.0082 | 5/5 |
| learned, strictly nested | 0.4511 | 5182 / 2202 / 3017 | +0.0072 | 5/5 |
| learned, fixed 0.2245 | 0.4506 | 5179 / 2213 / 3020 | +0.0067 | 5/5 |
| keep every seed | 0.3990 | 5710 / 5345 / 2489 | | |

- **Per fold:** as claimed +0.0045 / +0.0087 / +0.0106 / +0.0093 / +0.0080; strictly nested +0.0034 / +0.0079 / +0.0083 / +0.0099 / +0.0064.
- **Bootstrap 5-95%:** as claimed +0.0057..+0.0108 by image and +0.0058..+0.0107 by observation day (0 of 10,000 draws at or below zero); strictly nested +0.0048..+0.0096. Paired sign-flip test: p < 0.0001.
- **Fast formula:** it differs from `score_file` in one image by one TP (5267 / 2601 / 2932).
- **Images without seeds:** 9, and their 16 drawn filaments are counted as FN.
- **Stored IoUs:** my independent recomputation matches to 3e-8 over 11,054 seed-record pairs.
- **Baseline is fair:** 300 / 0.85 is the best (area, mean) pair on this bank in a 10 x 12 grid; re-tuned nested by fold it scores 0.4432.
- **Ceiling:** keeping exactly the seeds with true u > PQ/2 gives 0.5149.

| Refined shapes (kcb2/w05, t_r 0.5, absorb 0.3) | PQ | TP / FP / FN | vs fixed rule |
|---|---|---|---|
| fixed rule | 0.4490 | 5300 / 2560 / 2899 | |
| learned, tau 0.25 | 0.4563 | 5116 / 1995 / 3083 | +0.0074 |
| learned, nested among 0.22 / 0.2245 / 0.25 | 0.4560 | | +0.0070 |
| learned, strictly nested | 0.4558 | 5209 / 2172 / 2990 | +0.0069 |
| learned, fixed 0.2245 | 0.4552 | 5204 / 2185 / 2995 | +0.0062 |

## 3. Null tests (20 permutations each)
| Null | Learned minus reference | Range |
|---|---|---|
| A: targets permuted within fold x record count, train and evaluate | -0.0002 vs keep-all | max -0.00001 |
| B: train on permuted targets, evaluate on true ones | -0.0451 vs fixed rule | max -0.0446 |
| C: targets permuted within each image | -0.0004 vs keep-all | max +0.0002 |
| D: permuted within fold x records x rule decision | -0.0006 vs fixed rule | max -0.0003, 0 of 20 positive |

- Under A and C the fixed rule becomes a random subset (0.3395 / 0.3496), so keep-all (0.3989) is the honest reference. The learned rule learns to keep everything.
- Under D the fixed rule keeps its real 0.4439, and the learned rule never beats it.

## 3b. Other splits (descriptors unchanged)
| 10 random day-grouped 5-fold splits | mean | sd | min | max |
|---|---|---|---|---|
| as claimed | +0.0074 | 0.0009 | +0.0062 | +0.0089 |
| strictly nested | +0.0072 | 0.0008 | +0.0058 | +0.0081 |
| fixed 0.2245 | +0.0078 | 0.0006 | +0.0069 | +0.0087 |
| fixed 0.25 | +0.0070 | 0.0009 | +0.0057 | +0.0084 |

- **Folds better:** 5/5 in nine splits, 4/5 in one.
- **Per split, as claimed / strictly nested:** 0.0073/0.0062, 0.0074/0.0081, 0.0080/0.0080, 0.0089/0.0073, 0.0074/0.0072, 0.0063/0.0058, 0.0076/0.0080, 0.0062/0.0065, 0.0062/0.0070, 0.0089/0.0080.
- **Chronological 5 blocks:** +0.0076 overall, per block +0.0127 / +0.0070 / +0.0081 / +0.0062 / -0.0052. The last block (2017-04 to 2022) has a 5-95% range of -0.0144..+0.0038.
- **Calendar-year blocks:** +0.0074; the 2018-2022 block scores -0.0076.
- **Other fold counts:** 10 folds +0.0078 (10/10); 3 folds +0.0053 (3/3).

## 3c. By number of annotator records (original folds)
| Records | Images | Fixed | Learned | As claimed (5-95%) | Strictly nested | Mean over 10 splits |
|---|---|---|---|---|---|---|
| 1 | 411 | 0.4478 | 0.4518 | +0.0040 (-0.0000..+0.0080) | +0.0045 | +0.0055 |
| 2 | 145 | 0.4527 | 0.4623 | +0.0096 (+0.0046..+0.0145) | +0.0089 | +0.0081 |
| 3 | 151 | 0.4343 | 0.4457 | +0.0114 (+0.0076..+0.0156) | +0.0086 | +0.0088 |

## 4. Hyper-parameters (19 settings, original folds; all 5/5 folds better)
| Measure | mean | sd | min | max |
|---|---|---|---|---|
| as claimed | +0.0073 | 0.0008 | +0.0060 | +0.0084 |
| strictly nested | +0.0071 | 0.0007 | +0.0056 | +0.0083 |
| fixed 0.2245 | +0.0070 | 0.0006 | +0.0053 | +0.0080 |

As claimed / strictly nested, per setting:
- claimed setting 0.0082/0.0072
- depth 2: 0.0063/0.0066; depth 3: 0.0078/0.0068; depth 5: 0.0083/0.0078; depth 6: 0.0067/0.0066
- 150 iterations: 0.0082/0.0072; 600: 0.0070/0.0075; 1000: 0.0067/0.0056
- leaf 15: 0.0082/0.0072; leaf 60: 0.0081/0.0082; leaf 120: 0.0068/0.0069
- lr 0.1 x 100: 0.0066/0.0065; lr 0.01 x 900: 0.0084/0.0080
- l2 0: 0.0080/0.0072; l2 10: 0.0076/0.0083
- depth 3, 600 it, leaf 60: 0.0074/0.0073; depth 5, 150 it, leaf 15: 0.0065/0.0075
- over-fitted on purpose (depth 6, 600 it, leaf 15): 0.0060/0.0059; under-fitted (depth 3, 150 it, leaf 60): 0.0060/0.0067

Other robustness checks:
- **Other learners, same descriptors:** random forest +0.0077, extra trees +0.0052, ridge on rank-normalised descriptors +0.0058, boosting without record weights +0.0061, absolute-error loss +0.0069; all 5/5 folds.
- **Training images resampled with replacement (12 times):** mean +0.0058, min +0.0043, max +0.0072.
- **Saved variants of the main session, re-scored:** +0.0037, +0.0075, +0.0076, +0.0080, +0.0082 (the claim), +0.0082, +0.0085, +0.0086.

## 5. Anything else
- **Why now and not before.** On this bank, 11 descriptors like the 10/24 attempt give +0.0010 (3/5 folds), which matches the earlier null result. Boosting on area + mean alone gives -0.0043. The 19 seed-only shape and probability descriptors give +0.0049, and all 50 give +0.0082.
- **No single group carries it.** Dropping one group leaves +0.0049 to +0.0079. Dropping the 11 within-image descriptors leaves +0.0034 (0.4473, the same number as in EXPERIMENTS.md). No group alone on top of area + mean adds anything (-0.0027 to +0.0002).
- **Where the gain comes from.** The rule removes 601 kept seeds (mean u 0.170), worth +0.0068, and adds 137 (mean u 0.273), worth +0.0014.
- **Single-model seed banks.**
  - K only: 0.4480 vs 0.4429, +0.0051 (5/5 folds); strictly nested +0.0057.
  - C only: 0.4478 vs 0.4423, +0.0055 (5/5 folds); strictly nested +0.0068.
- **Time adjacency.** The gain does not depend on a near-in-time training image: by distance to the nearest image of another fold, <= 1 day +0.0097, 2-3 days +0.0102, 4-7 days +0.0031, > 7 days +0.0084. Test images are closer to the training days than out-of-fold images are (49% within 1 day vs 35%; 33 of 180 on the same day as a train image).
- **Out-of-fold vs test seeds** (our own predictions only). A classifier cannot tell them apart (AUC 0.516). At tau 0.25 the model drops 11.6% of rule-kept seeds on test vs 12.5% out of fold.
- **By period.** 2011-2017 +0.0090 (5-95% +0.0066..+0.0117); 2018-2022 -0.0087 (-0.0210..+0.0027).
- **By site.** B +0.0133, C +0.0069, L +0.0066, M +0.0115, T +0.0007, U +0.0107.
- **Single images.** Without the 5 / 10 / 20 / 50 most favourable images the gain is +0.0071 / +0.0063 / +0.0047 / +0.0013. Decisions change in 412 images: 260 help, 152 hurt.

## Not checked
- The refiner's own cross-fitting. The apply logs show `refiner_f<k>` used on fold k; I only re-scored its outputs.
- The "trim 0.48" row (0.4539 vs 0.4456).
- Anything about test labels or the leaderboard.

## Rule compliance
- No test annotation or credential was opened; I used test file names and our own test descriptors only.
- No GPU.
- Writes went only to my folder; the original `keep_fit2.py` was re-run with its output redirected there.
- One slip on the process limit: for about a second, a small JSON-printing Python command ran while two of my jobs were active.

Files are in C:\Users\harri\code\filament-segmentation-2026\scratch\ws\k1_audit:
- audit_lib.py
- rebuild_feats.py
- score_maps.py
- exp_*.py
- *.log
- *.json