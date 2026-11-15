# Independent verification of K2 (confirmed, gain about +0.002)

## Problems found

- No independent data: every number, the explorer's and mine, comes from the same 707 training images. The ten 'confirmation' re-splits and the development splits are re-partitions of those images, and the original-folds result was already known when the test was pre-registered. Mitigation I measured: when the choice among the 44 cached variants is itself made without the held-out fold, the gain is +0.0018 instead of +0.0023 (6/6 development splits positive).
- The headline +0.0018 is for stage-1 masks with the stored-IoU formula. On the final pipeline output (refiner shapes on real label maps, original folds) the explorer's own scores give +0.0013 with a bootstrap interval that includes zero (-0.0003..+0.0027), and its second network set gives +0.0009 (-0.0006..+0.0024). One split alone does not establish the gain.
- At the pipeline's present threshold tau 0.25 on the original folds the gain is only +0.0005 to +0.0011 for all five network seed sets I checked, never distinguishable from zero. The candidate needs tau re-chosen (about 0.22-0.23). On fresh re-splits the gain at 0.25 is +0.0017 (11/11), so the original folds at 0.25 are the unfavourable spot, but that is the setting the pipeline uses today.
- Network-seed dependence on a single split: final-shape gain on the original folds ranges from +0.0009 to +0.0018 over five seed sets (stage-1: +0.0013 to +0.0023), mostly through which threshold each fold's inner cross-fit picks. Five networks per descriptor set is too few for a stable build.
- The gain is not uniform (final shapes, 11 splits pooled): +0.0029 on 2011-2014 images, +0.0003 on 2015-2017 (-0.0015..+0.0019), -0.0004 on 2018-2022 (-0.0047..+0.0036); none on images with 7 or fewer seeds (-0.0001); +0.0011 on 1-record images (-0.0003..+0.0025). According to another agent's log (scratch/ws/k1_audit/exp_year.log, not re-derived by me) the later years weigh somewhat more in the test set.
- Mild favourable choice of settings: four other reasonable hyper-parameter settings give +0.0014 to +0.0016 (final shapes, 6 splits) against +0.0015 to +0.0020 for the claim's settings over three seed sets. Nothing collapses, but the claim's settings sit at the top of the range.
- Small effect: only about 4% of the seeds (roughly 300 of 6788) are decided differently from K1; the gain is about twice the 0.001 noise yardstick.
- Day grouping does not separate the same filament seen on consecutive days: 114 of 707 images have an image within 24 hours in another fold, 333 within 2 days. This applies to K1 and the candidate alike and does not produce the gain (time-blocked splits: +0.0026, 6/6).
- Reproducibility and cost: reimpl66.npz has no build script in the explorer's folder (I rebuilt it from the raw inputs, it is bit-identical); the 66 descriptors are not in src/ and hard-code disk centre 1023.5 and radius 900; network scores or weights would have to be shipped.
- Process: the explorer self-reported three slips of the 2-process limit. Mine: one stray command started a system Python that hung for about two minutes while none of my other processes ran; I stopped it.

## Report

## Verdict
Confirmed. I tried to refute it and could not. The gain is real on the out-of-fold data but small, about +0.002 PQ. My number for the final pipeline (refiner shapes on real label maps, strictly nested, original folds, my own network seeds) is **+0.0018**. Over the original folds plus 10 fresh re-splits it is +0.0020, with a split-averaged paired bootstrap over images of +0.0011..+0.0029.

It is not explained by leakage, by picking the best of 44 variants, by network seeds or by the particular hyper-parameters. On the original folds alone, and at the pipeline's present tau 0.25, it is not distinguishable from zero.

My rule, fixed before I had my own re-split numbers: confirm only if the final-pipeline gain is at least +0.001 over original folds plus fresh re-splits with the bootstrap 5% bound above zero, no setting collapses, the null is clean and the nested choice of variant stays positive. All held.

## 1. Re-running the claim
- **From its cache:** `final_eval.py` prints the same lines as `final_cand.log`.
- **From raw inputs:** I rebuilt every per-seed table from the train image, the two out-of-fold dense maps and the TRAIN index. The seed bank, the 50 descriptors, the IoU targets and the explorer's 66 descriptors are bit-identical for all 6788 seeds.
- **From scratch:** my own cross-fit loop with the explorer's network random states reproduces its cached outer and inner scores of all four members on the original folds and rs1..rs10 with max difference 0 (44 of 44).
- **Its own scripts:** `candidate.py --check` gives 0.0; `analysis/keep_eval.py` on `scores_cand.npz` gives 0.4571 strict and 0.4570 at tau 0.25.
- **Ground truth:** `gt_index.pkl` matches the train annotation file (707 files, 1154 records, 8199 filaments; 410 polygons of 40 random images decoded, all identical).

| quantity | claimed | reproduced |
|---|---|---|
| original folds, stage-1 formula, strict: K1 / candidate | 0.4510 / 0.4528 (+0.0018) | same; same thresholds and TP/FP/FN |
| rs1..rs10, stage-1 formula, strict | mean +0.0028, 10/10 | same, every split equal |
| original folds, real maps, stage-1 masks, strict | 0.4511 / 0.4529 | same |
| original folds, real maps, refiner shapes, strict | 0.4558 / 0.4571 (+0.0013) | same; bootstrap -0.0003..+0.0027 |
| same at tau 0.25 | 0.4563 / 0.4570 | same; bootstrap -0.0009..+0.0021 |
| fixed rule, stage-1 / refiner shapes | 0.4439 / 0.4490 | 0.4439 (5268/2601/2931) / 0.4490 (5300/2560/2899) |
| decisions changed | drops 187 (mean u 0.195), adds 122 (0.251) | same |

## 2. Leakage and selection
- **Inputs:** both descriptor functions receive only the image, the two maps and the seeds; no annotation and no file name reaches them. The rebuild from raw being bit-identical confirms this.
- **Cross-fitting:** every fit is by fold with observation days disjoint (asserted in my loop). The rank transform is fitted on training rows only. Thresholds come from an inner cross-fit that never touches the held-out fold.
- **Nothing tuned:** equal blend weights, K1's boosting settings, and the cache holds no other network settings.
- **Selection:** 44 scorers and blends were tried on the same images (development mean: best +0.0024, median +0.0008). With the variant chosen per held-out fold from inner scores only, the gain over K1 is +0.0018 (6/6 development splits) against +0.0023 for the candidate. On the original folds it is +0.0014 (bootstrap -0.0000..+0.0028). Selection inflates by about 0.0005, not more.
- **Label-permutation null, my loop:** permuted within fold and record count: keep-all 0.3989, K1 0.3988, candidate 0.3990. Permuted within image, two seeds: candidate minus K1 +0.0007 and -0.0003.

## 3. My own evaluation
Own tables, own loop, own network random states, real label maps through `postproc.score_file`. Gain over K1 on the original folds, with paired bootstrap 5-95% over images:

| | set 1 | set 2 | set 3 |
|---|---|---|---|
| stage-1 masks, strict | +0.0023 (+0.0007..+0.0037) | +0.0021 (+0.0006..+0.0035) | +0.0021 (+0.0005..+0.0036) |
| refiner shapes, strict | +0.0018 (+0.0003..+0.0033) | +0.0018 (+0.0003..+0.0032) | +0.0017 (+0.0002..+0.0032) |
| refiner shapes, tau 0.25 | +0.0006 (-0.0011..+0.0021) | +0.0005 (-0.0010..+0.0021) | +0.0011 (-0.0004..+0.0027) |
| refiner shapes, tau 0.23 | +0.0022 | +0.0020 | +0.0022 |

Set 1, refiner shapes, strict:
- K1 0.4558 (TP 5209 / FP 2172 / FN 2990), candidate 0.4576 (5198 / 2093 / 3001).
- Per fold: K1 0.4471 / 0.4615 / 0.4557 / 0.4629 / 0.4516; candidate 0.4472 / 0.4615 / 0.4589 / 0.4648 / 0.4556.
- 1/2/3-record images: K1 0.4573 / 0.4688 / 0.4458; candidate 0.4583 / 0.4695 / 0.4492.

The explorer's two network sets on the same maps give +0.0013 and +0.0009. The mean of all five sets is +0.0015.

## 4. Robustness
Gain over K1, both strict, real label maps; in brackets the number of positive splits.

| configuration | splits | stage-1 masks | refiner shapes | refiner shapes, tau 0.25 |
|---|---|---|---|---|
| claim settings, my set 1: original folds + 10 fresh day-grouped re-splits | 11 | +0.0023 (11/11) | +0.0020 (11/11, min +0.0009) | +0.0017 (11/11) |
| same, bootstrap over images of the split average | | +0.0014..+0.0032 | +0.0011..+0.0029 | |
| explorer's rs1..rs10 with its seeds | 10 | +0.0028 (10/10) | +0.0025 (10/10, min +0.0013) | +0.0024 (10/10) |
| time-blocked: 5 blocks in time, 3 by month, 2 by year | 6 | +0.0028 (6/6) | +0.0026 (6/6, min +0.0022) | +0.0028 |
| my sets 1 / 2 / 3, original folds + v1..v5 | 6 | +0.0024 / +0.0018 / +0.0022 | +0.0020 / +0.0015 / +0.0019 | +0.0015 / +0.0016 / +0.0015 |
| A: networks (32,16), alpha 0.3 | 6 | +0.0017 (5/6) | +0.0016 (5/6, min -0.0003) | +0.0019 |
| B: networks (128,64), alpha 3, lr 1e-3, 200 epochs | 6 | +0.0019 | +0.0015 (6/6) | +0.0015 |
| C: boosting depth 3, lr 0.05, 200 iterations | 6 | +0.0017 | +0.0014 (6/6) | +0.0016 |
| D: networks alpha 1e-4 with early stopping | 6 | +0.0016 | +0.0014 (6/6) | +0.0010 (5/6) |
| 66 descriptors without the 8 image-level ones | 6 | +0.0020 | +0.0016 (6/6) | +0.0014 |

- **Parts alone** (11 splits, stage-1 formula): boosting on 66 +0.0008, networks on 50 -0.0005, networks on 66 +0.0014, K1 + boosting-66 +0.0013, K1 + networks-50 +0.0013, all four +0.0023. The full blend is also the best on fresh splits.
- **Decisions changed:** in all 11 splits the dropped seeds average u 0.13-0.19 and the added ones 0.23-0.28, against a break-even of 0.226. About 250-310 seeds differ per split.
- **Where the gain sits** (refiner shapes, 11 splits pooled):
  - Period: 2011-2014 +0.0029 (+0.0018..+0.0039); 2015-2017 +0.0003; 2018-2022 -0.0004.
  - Records: 1 record +0.0011; 2 records +0.0025; 3 records +0.0025.
  - Seeds per image: 7 or fewer -0.0001; medium +0.0029; more than 12 +0.0020.

## What limits it
- Everything rests on the same 707 images; no submission was allowed, so there is no outside check.
- On the original folds alone the final-shape gain is +0.0009 to +0.0018 depending on the network seeds, with a bootstrap lower bound near zero.
- At tau 0.25 on the original folds it is +0.0005 to +0.0011 and not significant. Tau would have to be re-chosen, about 0.22-0.23.
- The gain comes from the busy 2011-2014 images and is about zero afterwards.

## Compliance
CPU only, TRAIN annotations only, at most two Python jobs at a time. I wrote only inside `C:\Users\harri\code\filament-segmentation-2026\scratch\ws\x_keepmodel_verify`.

Files are in that folder:
- build_raw.py
- vlib.py
- run_cf.py
- eval_cf.py
- eval_maps.py
- vmaps.py
- selection_nested.py
- null_test.py
- subsets.py
- table_variants.log
- maps_fresh_s1.log
- check_cache_all.log
