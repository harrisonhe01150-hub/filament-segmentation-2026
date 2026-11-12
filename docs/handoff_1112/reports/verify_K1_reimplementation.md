K1 INDEPENDENT RE-IMPLEMENTATION: the claim holds out of fold on stage-1 masks.

All code is mine, written from the raw inputs (probs files, training images, scratch/bank/gt_index.pkl) plus src/postproc (upsample, score_file), src/common (PQ) and src/fastsweep (components). I did not read keep_feats2.py, keep_fit2.py, keep_fit.py, k1_eval.py, the kf pickles or the seed bank. Every keep set below was scored by rebuilding the real 2048x2048 label map and calling postproc.score_file; a cached-IoU shortcut was asserted equal to it each time.

1. FIXED RULE REPRODUCED EXACTLY
707 images, 6788 seeds of at least 100 px. Area >= 300 and mean >= 0.85 keeps 4824 seeds: PQ 0.44392, TP 5268 / FP 2601 / FN 2931. Per fold 0.4366 / 0.4499 / 0.4440 / 0.4485 / 0.4402.

2. LEARNED KEEP RULE (first run, nothing tuned)
66 descriptors of my own in 8 groups: size/shape 12, disk position 7, stage-1 probability inside/around 12, two-model agreement 6, flat-fielded image 7, neighbouring seeds 8, rank/counts per image 11, image-level stats 3. Target u computed by me. HistGradientBoostingRegressor with the stated settings, weight = records, tau on a 0.005 grid.

"Stated" = tau per fold from the other four folds' cross-fitted predictions. "Strict" = tau from an inner cross-fit that never touches the held-out fold.

                    fixed    stated tau         strict tau
all 707            0.4439   0.4521 (+0.0082)   0.4511 (+0.0072)
TP/FP/FN       5268/2601/2931  5265/2317/2934  5244/2314/2955
fold 0             0.4366   0.4404 (+0.0037)   0.4400 (+0.0033)
fold 1             0.4499   0.4572 (+0.0074)   0.4544 (+0.0045)
fold 2             0.4440   0.4545 (+0.0105)   0.4533 (+0.0094)
fold 3             0.4485   0.4596 (+0.0111)   0.4588 (+0.0102)
fold 4             0.4402   0.4485 (+0.0083)   0.4491 (+0.0089)
1-record (411)     0.4478   0.4545 (+0.0067)   0.4532 (+0.0054)
2-record (145)     0.4527   0.4620 (+0.0094)   0.4609 (+0.0082)
3-record (151)     0.4345   0.4433 (+0.0089)   0.4427 (+0.0083)
bootstrap 5-95%             +0.0060..+0.0104   +0.0048..+0.0095

- Bootstrap is paired over images, 2000 resamples; resampling days or months gives the same intervals.
- Chosen tau: stated 0.21 / 0.215 / 0.21 / 0.21 / 0.21; strict 0.195 / 0.25 / 0.19 / 0.205 / 0.215.
- No tau search at all (tau = PQ_fixed/2 = 0.222 for every fold): 0.4513 (+0.0074).
- Reported claim: 0.4521, +0.0082, 5/5 folds, +0.0057..+0.0108. Mine agrees.

3. WHICH DESCRIPTORS CARRY THE GAIN (delta vs fixed: stated / strict)
all 66                                  +0.0082 / +0.0072
drop size/shape                         +0.0057 / +0.0064
drop position                           +0.0055 / +0.0058
drop probability                        +0.0047 / +0.0057
drop agreement                          +0.0069 / +0.0073
drop flat-fielded image                 +0.0078 / +0.0075
drop neighbours                         +0.0053 / +0.0058
drop rank/counts                        +0.0065 / +0.0080
drop image-level stats                  +0.0079 / +0.0071
drop rank/counts + image-level          +0.0070 / +0.0077
per-seed only (also no neighbours), 44  +0.0054 / +0.0071
drop agreement + image + image-level, 50  +0.0093 / +0.0082
only probability group, 12              +0.0049 / +0.0048
only probability + shape, 24            +0.0061 / +0.0054
only probability + shape + neighbours + position, 39  +0.0070 / +0.0074
only area + mean prob, 2                -0.0031 / -0.0033
area + mean prob + rank/counts, 13      +0.0003 / -0.0010
only rank/counts, 11                    -0.0021 / -0.0013
no probability descriptor at all, 34    -0.0019 / -0.0004

- The gain is still there without any rank/count descriptor: +0.0065 / +0.0080, and +0.0082 / +0.0077 with a random forest. That group adds roughly 0.001 at most.
- No single group is indispensable. Stated vs strict on the same model differ by up to 0.0015, so most single-group differences are within noise.
- The carrier is a detailed description of the probability map in and around the seed (about +0.005 alone); shape, neighbours and position add about +0.002. Largest PQ drops when one descriptor is shuffled: median probability 0.0085, faint-halo area (p >= 0.1 around the seed) 0.0066, edge probability 0.0040, distance to nearest seed 0.0028.
- The 2-descriptor result is a useful negative control: my pipeline does not produce a gain from nothing.

4. WHERE THE GAIN COMES FROM (stated tau)
- Learned keeps 4645 seeds, 4422 in common with the fixed rule.
- It removes 402 fixed-rule seeds (mean u 0.161; 154 matches against 511 false positives) and adds 223 (mean u 0.281).
- Removal alone gives +0.0067, addition alone +0.0016; never keeping seeds under 300 px still gives +0.0076.
- Net effect is almost pure false-positive removal: FP -284 at TP -3.

5. HARD-LOOK CHECKS, ALL PASSED
- Baseline is not weak: 300 / 0.85 is the in-sample optimum of a 10 x 19 threshold grid; re-tuned nested it scores 0.4435.
- Other cross-fitting partitions (5 random, chronological blocks, whole months x2, whole years): +0.0074..+0.0097 stated, +0.0082..+0.0100 strict, 5/5 folds every time.
- Other boosting settings (8): +0.0063..+0.0078 stated, +0.0063..+0.0076 strict. Random forest: +0.0080 / +0.0076.
- The number of annotator records is not guessable from my descriptors (cross-validated AUC 0.505, 0.511, 0.492).
- Same-day images never sit in different folds (0 of 101).
- Out-of-fold maps look like genuinely unseen predictions: test maps built the same way give the same seed statistics (fixed rule keeps 71.1% in both, mean probability 0.850 in both).
- Predicted u is well calibrated (out-of-fold R^2 0.50).
- Gain is positive for every observatory, period and seed-count bin.
- Ceiling for any keep rule on these seeds is about 0.515, so the learned rule captures about 11% of the headroom.

6. VERDICT
A learned keep rule on stage-1 seeds beats the fixed rule out of fold by about +0.007 to +0.008 PQ (5-95% roughly +0.005..+0.010). I found no leak or bug pattern that would explain it away. Caveats are in the problems list; the main ones are the unverified refined-shape numbers and the 2-model vs 10-model map difference at test time.

Files are in C:\Users\harri\code\filament-segmentation-2026\scratch\ws\k1_reimpl
- Code: seeds.py, feats.py, k1lib.py, run_main.py, run_ablate.py, run_ablate2.py, run_extra1.py, run_extra2.py, run_extra3.py, run_extra4.py, shift.py
- Results: main_result.txt, ablate_result.txt, ablate2_result.txt, extra1_result.txt, extra2_D.txt, extra2_EN.txt, extra2_MO.txt, extra3_result.txt, extra4_result.txt, extra5_result.txt, shift_result.txt
- Data: seeds_f0..4.pkl, feats_f0..4.pkl, main_result.pkl, ablate_result.pkl, ablate2_result.pkl, shift_feats.pkl
Nothing outside that folder was written or changed.