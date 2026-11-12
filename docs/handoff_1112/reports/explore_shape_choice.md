# Exploration: per-seed shape choice (NOT passed)

Written by an exploration agent on 2026-11-12 (CPU only, out of fold, TRAIN annotations only). Paths refer to Harrison's machine (scratch/ is not in git).

**Headline:** Per-seed shape choice fails: the oracle is +0.018 PQ (3 alternatives) but the strictly nested model scores 0.4562 (-0.0002, 0 of 5 folds); the oracle is annotator habit, not something the image shows.

**Recipe:**

No change to the pipeline: keep FINAL shapes (blend 0.5, threshold 0.5, absorb 0.3) for every kept seed. Do not add a per-seed (or per-image) shape choice.

To reproduce (repo root, CPU only, CUDA_VISIBLE_DEVICES empty, OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=4):
1. python scratch/ws/x_shape/build.py --folds 0 1 2 3 4 --check   (alternatives table for all 6788 seeds + baseline maps; asserts each of the 707 maps equals refine.paste_image; prints 0.4563)
2. python scratch/ws/x_shape/oracle.py ; python scratch/ws/x_shape/evalmaps.py --choices scratch/ws/x_shape/choices_oracle.npz --out scratch/ws/x_shape/maps_oracle.json   (oracle, isolated and on real label maps)
3. python scratch/ws/x_shape/fit.py --name primary --family R35 R50 R65 ; python scratch/ws/x_shape/evalmaps.py --choices scratch/ws/x_shape/choices_primary.npz --out scratch/ws/x_shape/maps_primary.json   (pre-registered model, strictly nested margin, real label maps)
4. bash scratch/ws/x_shape/run_secondary.sh ; bash scratch/ws/x_shape/run_third.sh signmag multi signmag_mix ; python scratch/ws/x_shape/build_fine.py ; python scratch/ws/x_shape/fit_cont.py ; python scratch/ws/x_shape/evalmaps.py --choices scratch/ws/x_shape/choices_cont.npz --out scratch/ws/x_shape/maps_cont.json ; python scratch/ws/x_shape/summary.py   (all other variants and the table)
5. python scratch/ws/x_shape/diag.py, diag2.py ... diag8.py, rho_models.py   (why it fails)

## x_shape: per-seed shape choice. NOT PASSED

Keep decisions fixed (K1 at tau 0.25, 4360 kept seeds). Everything below is out of fold on the 707 training images, TRAIN annotations only, scored on real label maps (refine.assemble, absorb 0.3, joins decided by the FINAL pieces) with postproc.score_file.

### 1. Baseline reproduced
0.4563 (TP 5116 / FP 1995 / FN 3083; folds 0.4479 / 0.4623 / 0.4573 / 0.4612 / 0.4527; 1/2/3-record images 0.4570 / 0.4681 / 0.4479). Each of the 707 maps is identical to refine.paste_image's (asserted in build.py). Cross-checks from my table: stage-1 masks 0.4521, stage-1 at >= 0.48 0.4539, as in the reference numbers.

### 2. Oracle (perfect per-seed choice, real label maps)
| alternatives per seed | oracle PQ | gain |
|---|---|---|
| R35 / R50 / R65 (blended refiner crop at 0.35 / 0.5 / 0.65) | 0.4742 | **+0.0179** |
| R50 or R65 (tighter only) | 0.4661 | +0.0098 |
| R50 or R35 (wider only) | 0.4645 | +0.0082 |
| R25 / R35 / R50 / R65 / R80 | 0.4852 | +0.0289 |
| R50, stage-1 mask, stage-1 >= 0.48, stage-1 >= 0.60 | 0.4701 | +0.0137 |
| R50, dilate 1 px, dilate 2 px, erode 1 px | 0.4796 | +0.0232 |
| R35, R50, R65, S48, D1, E1 | 0.4832 | +0.0269 |
| all 11 | 0.4907 | +0.0344 |

- 3-alternative oracle: folds +0.0195 / +0.0163 / +0.0176 / +0.0177 / +0.0183; 1/2/3-record images +0.0181 / +0.0185 / +0.0172; bootstrap +0.0166 to +0.0191. Half of it comes from 129 seeds that newly cross IoU 0.5, half from higher IoU of seeds already matched (mean matched IoU 0.683 -> 0.692).
- With the kept set fixed the PQ denominator does not move, so the oracle just takes the largest matched IoU per seed. The estimate from masks taken alone (+0.0182) agrees with the real maps (+0.0179).
- Every alternative applied to all seeds is worse than the baseline: R35 0.4487, R65 0.4529, S48 0.4539, D1 0.4316, E1 0.4292. Global thresholds 0.47 to 0.56 are flat at 0.456. The oracle is well above +0.005, so I went on to the model.

### 3. Pre-registered model (declared before any model output)
Alternatives R35 / R50 / R65. For R35 and R65 a HistGradientBoosting regressor (keep.GBM settings, 92 descriptors: 50 keep descriptors + 39 new crop descriptors of the bands between the levels + area ratios + K1 score) predicts the DIFFERENCE of matched IoU against R50 per record. Switch to the alternative with the larger prediction when it exceeds a margin. Model cross-fitted by fold (days never shared, asserted); margin per fold from an inner cross-fit that never touches that fold; the K1 score of training rows recomputed by keep models that never saw the held-out fold.

**Result: 0.4562, -0.0002. 0 of 5 folds better (0.0000 / -0.0009 / 0 / 0 / 0). 1/2/3-record -0.0005 / -0.0002 / +0.0002. Paired bootstrap -0.0004 to 0.0000. FAILED** (needed >= +0.0020, 4 of 5 folds, 5% bound above 0). The nested margin is "never switch" in 4 folds; in the fifth, 55 seeds are switched and lose.

### 4. Everything tried (strictly nested, real label maps, baseline 0.4563)
| variant | rule | seeds switched | PQ | gain | folds better | bootstrap 5-95% |
|---|---|---|---|---|---|---|
| primary | best of R35/R65 | 55 | 0.4562 | -0.0002 | 0 | -0.0004 .. 0.0000 |
| primary | wider only | 97 | 0.4560 | -0.0003 | 0 | -0.0006 .. 0.0000 |
| primary | tighter only | 1 | 0.4563 | 0.0000 | 0 | 0 |
| smoothed target | best / wider / tighter | 6 / 98 / 113 | 0.4563 / 0.4562 / 0.4561 | -0.0001 / -0.0001 / -0.0002 | 0 | all include 0 or below |
| stronger regularisation | best / wider / tighter | 2 / 122 / 1 | 0.4563 / 0.4560 / 0.4563 | 0 / -0.0004 / 0 | 0 | |
| trained on all 6788 seeds | best / wider / tighter | 221 / 82 / 146 | 0.4557 / 0.4562 / 0.4560 | -0.0007 / -0.0001 / -0.0003 | 0 / 1 / 0 | |
| crop descriptors only | best / wider / tighter | 19 / 54 / 712 | 0.4562 / 0.4562 / 0.4554 | -0.0001 / -0.0002 / -0.0010 | 0 | |
| trained on 2/3-record images only | best / wider / tighter | 3 / 78 / 2 | 0.4563 / 0.4562 / 0.4563 | -0.0001 each | 0 | |
| sign classifier x size regressor | best / wider / tighter | 388 / 100 / 476 | 0.4562 / 0.4560 / 0.4557 | -0.0001 / -0.0003 / -0.0006 | 2 / 0 / 2 | -0.0007 .. +0.0005 (best) |
| D1 / E1 | best / D1 / E1 | 0 / 34 / 0 | 0.4563 / 0.4562 / 0.4563 | 0 / -0.0001 / 0 | 0 | |
| S40 / S48 / S60 | best / S40 / S48 / S60 | 17 / 14 / 20 / 226 | 0.4562 / 0.4564 / 0.4563 / 0.4562 | -0.0001 / +0.0001 / 0 / -0.0001 | 0 / 2 / 0 / 0 | S40: 0.0000 .. +0.0003 |
| 5 levels R25..R80 | best / per level | 0 / 0-94 | 0.4560-0.4563 | 0 .. -0.0003 | 0 | |
| 6 alternatives | best / per alternative | 1 / 0-92 | 0.4560-0.4563 | 0 .. -0.0003 | 0-1 | |
| 6 alternatives, sign x size | best / per alternative | 12 / 15-666 | 0.4565 / 0.4561-0.4562 | +0.0002 / -0.0001 .. -0.0003 | 1 / 0-2 | best: 0.0000 .. +0.0004 |
| continuous threshold per seed from predicted size mismatch | one rule | 600 | 0.4558 | -0.0005 | 0 | -0.0009 .. -0.0002 |

13 variants, 46 rules, none passes. Range -0.0010 to +0.0002; the +0.0002 comes from 12 seeds and is noise. Picking the margin with the held-out folds in view (not honest) still tops out at +0.0004. All variants except the primary were defined after the primary failed. A 14th variant (2/3-record training with the sign x size model) was started and stopped unfinished because the process slot was needed.

### 5. Why it cannot be learned
- **No direction signal.** Cross-fitted predictions correlate 0.07 (wider) and 0.01 (tighter) with the true difference. The same pipeline predicts the SIZE of the change with r 0.28 / 0.25, so features and targets are aligned; only the sign is missing. Direction-only classifiers reach AUC 0.55 to 0.60.
- **Size mismatch decides the best alternative, but it is not predictable.** rho = log(annotator's pixel count / our mask's) has sd 0.25 on matched pairs. Tighter gains +0.02 to +0.03 per pair when rho < -0.2, wider the same when rho > 0.2. Five different learners explain at most 1.2% of rho (correlation 0.10).
- **It is the annotator's habit, not the image.** Mean rho on two disjoint halves of an image's seeds: same annotator r = 0.56, another annotator of the same image r = 0.06. Two annotators of the same seed agree on rho with r = 0.15 and on the gain of an alternative with r = 0.02 to 0.07. For comparison, the matched IoU itself, which the keep rule predicts, agrees with r = 0.51.
- **One choice per image already gives 64% of the per-seed oracle** (+0.0117 of +0.0182), and it does not transfer: chosen from the other annotators of the same image and scored on the one left out, it gains 0.0000 (bootstrap -0.0026 .. +0.0024, 296 images).
- **Per seed, a second annotator's drawing of the same filament is worth +0.0014** of the +0.0181 oracle on those images (-0.0004 .. +0.0033). Tighter only: +0.0021 (+0.0006 .. +0.0037). With 5 levels or the morphology family it is negative (-0.002 to -0.006): choosing the best of many alternatives for one annotator overfits that annotator.
- A second opinion is not a strict ceiling for a model (for keeping, K1 beats "keep what the other annotator matched", 0.468 against 0.452 on 2-record images), so the verdict rests on the model results above together with the tiny shared component.
- The baseline sits on the flat top of the threshold curve, so a per-seed shift pays only in the square of what can be predicted. With about 1% of the size mismatch predictable, the continuous rule comes out at -0.0005.

### 6. Recommendation
Close the shape-choice line; keep FINAL shapes. The +0.02 is real only if you know which annotator scores the image. This matches the unchanged mean matched IoU of 0.68 in every earlier experiment.

### 7. Rules followed
No test annotations, no network, no GPU (CUDA_VISIBLE_DEVICES empty), at most 2 Python processes of mine, OMP 4 / OPENBLAS 1. Only wrote inside scratch/ws/x_shape; git status is unchanged. I stopped one process, my own.

### Files (all in C:\Users\harri\code\filament-segmentation-2026\scratch\ws\x_shape)
- Code: xlib.py (alternatives, exact IoU against records, descriptors, label maps), build.py, oracle.py, fit.py, evalmaps.py, summary.py, build_fine.py, fit_cont.py, diag.py, diag2.py to diag8.py, rho_models.py, run_secondary.sh, run_secondary2.sh, run_third.sh
- Tables: tab_f0..4.npz (11 alternatives x 6788 seeds), fine_f0..4.npz (thresholds 0.35 to 0.65), base_f0..4.json (baseline maps per image), nested_k1_scores.npz
- Results: oracle.json, maps_oracle.json, maps_oracle_joinself.json, fit_primary.json, maps_primary.json, fit_<variant>.json, maps_<variant>.json, choices_<variant>.npz, summary.json, diag.json, diag2.json to diag8.json, rho_models.json
