# Exploration: stronger keep model K2 (passed)

Written by an exploration agent on 2026-11-12 (CPU only, out of fold, TRAIN annotations only). Paths refer to Harrison's machine (scratch/ is not in git).

**Headline:** PASSED: averaging two boosting models and two small-network ensembles over two descriptor sets beats K1 by +0.0018 PQ on the original folds and +0.0028 on the mean of 10 re-splits (10/10 positive), both strictly nested; with refiner shapes on real label maps 0.4571 vs 0.4558 strict, but only 0.4570 vs 0.4563 at the fixed tau 0.25.

**Recipe:**

score = plain mean of four predictions of u (u = mean over the image's records of IoU if IoU > 0.5 else 0), each model trained on the training seeds with sample weight = number of records:
1. k1: HistGradientBoostingRegressor(**keep.GBM) on K1's 50 descriptors (src/keep.py unchanged).
2. r66: the same boosting model on the 66 descriptors of scratch/ws/k1_reimpl/feats.py::image_feats(image float32, average map 0-255, the two member maps upsampled, label map of the >=100 px seeds, area, mean, peak). They use only the image, the two maps and the seeds of the same image (code read: no annotation, no file name). Aligned to the keep tables by seed order within the image; area and mean probability agree exactly for all 6788 seeds.
3. mlp: mean of 5 sklearn MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1.0, learning_rate_init=2e-3, max_iter=300, early_stopping=False), different random_state each, on the 50 descriptors after QuantileTransformer(n_quantiles=1000, output_distribution='normal') fitted on the training rows only, clipped to +-4.
4. mlp_r66: the same 5 networks on the 66 descriptors.
Keep a seed when score > tau; tau chosen exactly as K1's (best of 0.16..0.33 step 0.01 on an inner cross-fit that never touches the held-out fold). Strict taus on the original folds: 0.25 / 0.22 / 0.25 / 0.22 / 0.26. The curve is flat: stage-1 0.4534-0.4538 for tau 0.21-0.23 and 0.4527 at 0.25; refiner shapes on real maps 0.4578-0.4581 for 0.21-0.23 and 0.4570 at 0.25.
Nothing was tuned: boosting settings are K1's, the network settings were written down once before any result.
As one class: scratch/ws/x_keepmodel/candidate.py (Blend.fit(X50, X66, u, R) / .predict(X50, X66)); it reproduces the cached cross-fit of fold 0 with max difference 0.0. Training table of the final model (all 6788 seeds): scratch/ws/x_keepmodel/blend_table.npz.
Cache name of the candidate in my scripts: "k1+r66|mlp+mlp_r66".

## Verdict
Passed by the pre-registered rule. The gain is real but small, about +0.002 PQ, and on the original folds alone the evidence is thin (bootstrap lower bound about zero).

The candidate is the plain mean of four predictions of u: K1's boosting model on K1's 50 descriptors, the same model on the 66 descriptors of the independent re-implementation (`scratch/ws/k1_reimpl/feats.py`), and 5 small networks on each descriptor set. I fixed it in `prereg.txt` (file time 10:29) before any of its numbers on rs1..rs10 existed.

## The test (stage-1 masks, stored-IoU formula, strict nesting, K1's threshold rule for both)
K1 reproduced first: `analysis/keep_fit.py` gives 0.4510 strict with taus 0.23/0.26/0.22/0.21/0.21, and my harness gives the same. My K1 numbers on rs1..rs10 equal the audit's (mean +0.0072 over the fixed rule).

| | K1 | candidate | difference |
|---|---|---|---|
| original folds | 0.4510 | 0.4528 | **+0.0018** |
| per fold | 0.4400 / 0.4578 / 0.4523 / 0.4581 / 0.4466 | 0.4410 / 0.4575 / 0.4544 / 0.4602 / 0.4504 | 4 of 5 better |
| 1 / 2 / 3-record images | 0.4523 / 0.4616 / 0.4429 | 0.4531 / 0.4635 / 0.4456 | all better |
| TP / FP / FN | 5181 / 2202 / 3018 | 5169 / 2120 / 3030 | |
| paired bootstrap over images, 5-95% | | | +0.0002 .. +0.0032 (97% of resamples positive) |
| 10 re-splits rs1..rs10 | | | mean **+0.0028**, sd 0.0008, min +0.0017, max +0.0043, **10/10 positive** |

- Per re-split: +0.0035, +0.0024, +0.0018, +0.0029, +0.0023, +0.0043, +0.0017, +0.0033, +0.0033, +0.0025. The bootstrap 5% bound is above zero in each.
- Against the fixed rule on the original folds: +0.0089 (+0.0066 .. +0.0114).

## Real label maps (postproc.score_file, original folds)
| shapes | K1 | candidate | difference | bootstrap 5-95% |
|---|---|---|---|---|
| stage-1, strict | 0.4511 | 0.4529 | +0.0018 | +0.0002 .. +0.0033 |
| refiner shapes (keep_eval --shape refine), strict | 0.4558 | **0.4571** | +0.0013 | -0.0002 .. +0.0028 |
| refiner shapes, tau 0.25 | 0.4563 | 0.4570 | +0.0006 | -0.0009 .. +0.0021 |
| refiner shapes, tau 0.23 | 0.4556 | 0.4581 | +0.0024 | +0.0009 .. +0.0039 |

- Both K1 baselines reproduce (0.4563 at tau 0.25, 0.4558 strict).
- Refiner shapes, strict: 4 of 5 folds better; 1/2/3-record images 0.4583 / 0.4694 / 0.4480 against 0.4573 / 0.4688 / 0.4458.
- The candidate's curve on refiner shapes is above K1's at every tau from 0.20 to 0.28 (+0.0007 to +0.0035). Its top is at 0.21-0.23 (0.4578-0.4581); K1's is at 0.24-0.25 (0.4563).
- Refined-shape formula (each seed's final shape taken alone) over the re-splits: original folds +0.0013, mean of rs1..rs10 +0.0026, 10/10.

## What limits the claim
- **One split is weak evidence.** The original-folds bootstrap bounds are +0.0002 (stage-1) and -0.0002 (refiner shapes). The re-splits agree, but they and my development splits are partitions of the same 707 images, so they are not independent confirmation. The original-folds number was also part of the development data.
- **At tau 0.25 on the original folds the gain is only +0.0006.** That is where K1 happens to peak on this split. At the same fixed threshold on the ten re-splits the gain is +0.0023 to +0.0028 for every tau from 0.21 to 0.26, 10/10 each. TAU would have to be chosen again; the candidate prefers about 0.22-0.23.
- **The result on one split moves with the network seeds** (checked after the test). A second independent set of 5+5 networks gives +0.0013 on the original folds and +0.0028 on rs1..rs5. All 10+10 together give +0.0028 (bootstrap +0.0014 .. +0.0042) and +0.0030, and 0.4582 strict on real maps with refiner shapes (+0.0024, +0.0009 .. +0.0038). Ten or more networks per descriptor set would be the safer build.
- **No help on 2018-2022 images:** +0.0005 (-0.0067 .. +0.0069); K1 and the candidate both stay below the fixed rule there (0.4352 / 0.4357 / 0.4390). On 2011-2017: +0.0018.
- **Cost to use it.** The 66 descriptors are not in `src/`; `image_feats` has to be ported and run on the test images. The networks are refitted by scikit-learn, which may differ slightly between machines, so the fitted weights or the scores should be shipped.
- **Lighter variant, declared as secondary before the test:** K1 averaged with 5 networks on the same 50 descriptors, no new descriptor code. Original folds +0.0014 (+0.0002 .. +0.0027), re-splits mean +0.0016, 9/10 positive (min -0.0001).

Checks that came out clean:
- Targets permuted within fold x record count: K1 0.3987, candidate 0.3989, keep-all 0.3989.
- Targets permuted within fold x records x fixed-rule decision: K1 0.4430, candidate 0.4429, fixed rule 0.4439.
- Where the gain comes from (original folds): the candidate drops 187 seeds K1 keeps (mean u 0.195) and adds 122 (mean u 0.251).

## Everything tried
Development splits = original folds + five re-splits with other seeds (rs101..rs105). Numbers are the mean strict gain over K1, with the count of positive splits out of 6. 44 scorer variants and 6 threshold rules; the candidate was chosen from these, using a rule written down beforehand (best of three structured designs, simpler one if within 0.0003).

| same 50 descriptors, one model | gain |
|---|---|
| random forest | -0.0002 (3) |
| extra trees | -0.0009 (2) |
| ridge on rank-normal descriptors | -0.0009 (0) |
| additive splines + ridge | -0.0004 (1) |
| additive logistic | -0.0004 (3) |
| boosting with cross-entropy on fractional u | +0.0006 (4) |
| two-part P(match) x E[IoU given match] | +0.0004 (3) |
| three-part (drew x matched x IoU) | -0.0003 (3) |
| 5 small networks | +0.0005 (5) |
| four targets at once: networks / forest / extra trees | -0.0016 (1) / -0.0005 (2) / -0.0019 (0) |
| image-level descriptors cut into 8 bins | -0.0006 (1) |
| monotone constraints | -0.0001 (2) |
| K1 on 10 bootstrap samples of the training images | +0.0004 (4) |
| trained on the refined-shape IoU (scored on refined-shape IoU) | -0.0008 (1) |

| more descriptors | gain |
|---|---|
| boosting on 50 + my 28 crop descriptors | +0.0006 (4) |
| boosting on the 66 | +0.0011 (4) |
| boosting on 50 + 66 | +0.0011 (4) |
| 5 networks on the 66 | +0.0014 (6) |

| blends (equal weights) | gain |
|---|---|
| K1 + forest | +0.0007 (5) |
| K1 + forest + extra trees + ridge | +0.0009 (5) |
| K1 + forest + extra trees + splines + networks | +0.0012 (5) |
| K1 + cross-entropy + two-part | +0.0007 (6) |
| K1 + three-part | +0.0008 (5) |
| K1 + cross-entropy + two-part + three-part | +0.0008 (6) |
| K1 + bagged K1 | +0.0004 (5) |
| six models / all nine | +0.0012 (5) / +0.0013 (5) |
| cross-entropy + two-part + networks | +0.0017 (6) |
| cross-entropy + networks | +0.0015 (5) |
| K1 and networks (the secondary) | +0.0016 (6) |
| bagged K1 and networks | +0.0016 (5) |
| K1 and four-target networks | +0.0006 (4) |
| K1 + boosting-66 | +0.0018 (5) |
| K1 + boosting-66 + boosting-78 | +0.0021 (5) |
| boosting-116 + boosting-78 | +0.0017 (6) |
| boosting-116 and networks | +0.0018 (5) |
| K1 + boosting-66, and networks-50 | +0.0019 (6) |
| boosting-116 + cross-entropy + two-part, and networks | +0.0021 (6) |
| boosting-66 and networks-66 | +0.0016 (6) |
| **candidate** | **+0.0023 (6)** |
| candidate + four-target networks | +0.0024 (6) |
| candidate + cross-entropy + two-part | +0.0023 (6) |

- **Threshold rules on K1's own scores:** 0.001-step grid -0.0001; isotonic calibration then "above PQ/2" -0.0001; smoothed grid 0.0000; half of the best PQ -0.0002; fixed 0.2245 -0.0001; fixed 0.25 -0.0001. Calibration gains nothing.
- **Refined-shape target:** the refiner gains 280 record matches and loses 93 over all seeds, but the seeds K1 drops still average u 0.13 with the refined shape, so retraining on it cannot flip decisions profitably.
- **Members alone on the confirmation splits** (read after the test; original folds / mean of rs1..rs10 / positive): boosting-66 -0.0004 / +0.0014 / 10; networks-50 +0.0005 / +0.0003 / 7; networks-66 +0.0017 / +0.0021 / 10; K1 + boosting-66 +0.0017 / +0.0018 / 10. The second descriptor set carries most of the gain; mixing trees with networks adds the rest.
- **Not finished:** raw-IoU model with isotonic mapping, and networks on the 78 descriptors; I stopped both to free a process slot and looked at no result.
- **How far any model can go:** two annotators' outcomes on the same seed correlate 0.53, which caps the explainable share of one record's target; K1 explains 0.40. On 3-record images K1 already predicts an annotator better than the other two annotators' outcomes do. Most of the 0.515 "ceiling" is annotator disagreement.

## Rule compliance
- CPU only, TRAIN annotations only, no file-name information in any descriptor (year used once, for the by-period read-out). I wrote only inside `scratch/ws/x_keepmodel`.
- OMP_NUM_THREADS=4 and OPENBLAS_NUM_THREADS=1 were set for every process; inside the long jobs I limited OpenMP to 1 thread, which is faster for these small fits and gives bit-identical K1 scores.
- Process limit: three slips while two of my jobs were running. A one-second syntax check, an instant `python3 --version`, and a stray `python -` in a shell line that sat idle for about two minutes until I stopped it.
- I stopped two of my own jobs by process id and nothing else.

## Reproduce (repo root, CUDA_VISIBLE_DEVICES empty)
All cross-fitted scores are cached in `scratch/ws/x_keepmodel/cache`, so these run in minutes:
- `.venv\Scripts\python.exe scratch/ws/x_keepmodel/final_eval.py --cand "k1+r66|mlp+mlp_r66" --scores scratch/ws/x_keepmodel/scores_cand.npz`
- `.venv\Scripts\python.exe analysis/keep_eval.py --scores scratch/ws/x_keepmodel/scores_cand.npz --bank scratch/ws/main/rbank/kc_bridge --shape refine --pred scratch/ws/main/kcb/pred --strict-taus 0.25 0.22 0.25 0.22 0.26`
- From scratch: `scratch/ws/x_keepmodel/run.py --omp 1 --fill-only --learners k1,r66,mlp,mlp_r66 --splits all` (about 50 minutes).

Files are in C:\Users\harri\code\filament-segmentation-2026\scratch\ws\x_keepmodel:
- prereg.txt
- result.json
- final_cand.log
- maps_refine_compare.log
- dev_table.log
- candidate.py
- blend_table.npz
- scores_cand.npz
