# Exploration: 2018-2022 weakness of the keep rule (NOT passed)

Written by an exploration agent on 2026-11-12 (CPU only, out of fold, TRAIN annotations only). Paths refer to Harrison's machine (scratch/ is not in git).

**Headline:** No robust fix. Bagged K1 (48 bootstrap refits averaged) meets the pre-registered numbers on the registered split (2018-2022 +0.0081 folds / +0.0091 leave-period-out, overall +0.0011), but 0 of 4 re-splits reproduce the block gain (+0.000..+0.004): with nested thresholds the single K1 is below the fixed rule on 2018-2022 only on the registered split. Reproducible: bagging gives +0.001 overall on 5 of 5 splits. K1 under-predicts that period by 0.045 and no image-based rule removes that.

**Recipe:**

Not a fix of the 2018-2022 block, but worth adopting for a replicated +0.001 overall and to take the fit lottery out: replace the single keep regressor by the average of 48 refits of the same regressor (src/keep.py GBM settings, same 50 descriptors, same target u, record weights), each fitted on a bootstrap resample of the training IMAGES (all candidates of a drawn image come along, an image drawn twice counts twice); keep a candidate when the averaged score > tau (tau unchanged). Drop-in module: scratch/ws/x_late/keep_bagged.py (load_model(table).predict(X)); table: scratch/ws/x_late/keep_table_bagged.npz (= weights/keep_table.npz plus the image column needed for resampling). Fit about 140 s on CPU, deterministic. On the 1727 test candidates it changes 26 decisions at tau 0.25 (16 newly kept, 10 newly dropped; 2 of them on 2018-2022 images). Nothing in src/ was changed.

VERDICT: NOT PASSED as a robust fix. One principled change, bagging the keep model, meets the pre-registered numbers on the registered split, but four other day-grouped splits do not reproduce the block gain, so I do not count it. The reason is the main finding: with nested thresholds the single K1 is below the fixed rule on 2018-2022 only on the registered split.

All numbers are stage-1 masks, stored-IoU formula of analysis/keep_fit.py, strict nesting, unless stated. "LPO" = 2018-2022 scored by models trained on 2011-2017 only, every setting chosen inside 2011-2017.

## 1. Baseline (reproduced exactly; real label maps agree to 0.0001)
- Fixed rule 0.4439, K1 nested 0.4521, K1 strict 0.4510 (taus .23/.26/.22/.21/.21; folds .4400/.4578/.4523/.4581/.4466).
- 2018-2022 block (89 images, 384 drawn filaments, 310 candidates, 193 rule-kept): fixed rule 0.4390; K1 strict 0.4352 (-0.0039, 5-95% -0.013..+0.006); K1 at tau 0.25: 0.4303 (-0.0087, -0.021..+0.003).
- LPO (tau chosen inside 2011-2017 = 0.23): 0.4339. Having the period in the training folds changes nothing (0.4303 with, 0.4312 without, at tau 0.25).
- Control one step earlier (train <= 2016, score the 71 sparse images of 2017): K1 0.4485 against 0.4356 for the fixed rule.

## 2. Diagnosis
- **Which seeds (tau 0.25).** K1 drops 22 rule-kept seeds there (31 records) with record-weighted mean u 0.333 (it predicted 0.194); 9 are clear matches (IoU 0.53-0.83) scored 0.19-0.25. On 2011-2017 the 579 dropped ones have mean u 0.164. The drops cost -0.0106 on the block; the 11 additions (mean u 0.245) give +0.0016.
- **A calibration offset of the period.** Rule-kept seeds are predicted 0.452 and are worth 0.497 (+0.045; image bootstrap 5-95% +0.008..+0.079; p = 0.001 against random sets of 89 images). Annotators match them more often (0.743 against 0.666) with the same IoU when matched. By year: 2018 -0.002, 2019-20 +0.113, 2021 -0.010, 2022 +0.101. It survives bagging (+0.044), 40 image descriptors (+0.036) and a random forest (+0.058).
- **Not sparseness.** Sparse 2011-2017 images are calibrated (residual -0.011..+0.006) and are where K1 gains most (+0.016 with <= 3 candidates, +0.015 with 4-6). Sets of 2011-2017 images matched to the late images' candidate counts: K1 minus rule = +0.014 +- 0.006 (5000 draws); the late block's -0.004 is below 99.9% of them. Best threshold: 0.18 on late images with <= 4 candidates, 0.24 on 2011-2017 images with <= 4 candidates (2017 alone: 0.24), the same as on crowded images.
- **Not who annotated.** From the 2011-2017 residuals of the same 37 record groups the late residual should be -0.007; it is +0.045, spread over the groups.
- **Pixels and maps.** Late images have 3.5 candidates against 10.5 (AUC 0.88 from this alone; 28 of 89 have 0 or 1 candidate, 4 of 618 before), a larger disk in 2021 (903 px) and 2022 (915 px; 893-894 px in 2011-2020), a higher fine-to-mid texture ratio (1.34 against 1.09), fewer bright plage pixels (0.35% against 0.68%), and more disagreement between the two dense models in 2020-2022 (0.09-0.115 against 0.06-0.07). 40 image descriptors separate late from early with AUC 0.92 (0.79 against sparse 2011-2017 images), but none explains where K1 under-predicts (rank correlation with the per-image residual at most 0.10).
- **Fit lottery.** Refitting K1 on another sample of the training images moves a candidate's score by 0.04 (1 sd). On this block single refits have a PQ sd of 0.005 at tau 0.25.
- **Five splits, single K1 minus fixed rule on 2018-2022.** Nested taus: -0.0039 (registered), +0.0023, +0.0029, +0.0018, +0.0022. At tau 0.25: -0.0087, -0.0048, -0.0083, -0.0034, +0.0025 (mean -0.0045). The robust part of the weakness is at the production threshold 0.25; the block wants a lower one because of the offset.
- **Site T (side question).** No gain at any tau (+0.003..-0.004 for tau 0.16-0.30); K1 under-predicts there by 0.027. T images are sharper and the dense models disagree more, as on late images, but inside 2011-2017 neither property goes with a smaller gain. Unexplained; bagging does not change it.

## 3. What I tried
Differences to K1 strict on the registered split. Paired bootstrap over images, 2000 resamples, 5-95% in brackets. "met" = meets the four pre-registered numbers on that split.

| candidate | overall | 2011-2017 | 2018-2022 folds | 2018-2022 LPO | rule |
|---|---|---|---|---|---|
| fixed rule (reference) | -0.0072 | -0.0077 | +0.0039 [-0.006,+0.013] | +0.0052 | - |
| K1 at tau 0.25 / 0.22 / 0.20 everywhere | +0.0010 / +0.0001 / -0.0003 | +0.0013 / 0.0000 / -0.0005 | -0.0049 / +0.0022 / +0.0041 | -0.0026 / +0.0060 / +0.0115 | no |
| D1 do-no-harm: rule-kept seeds dropped only below tau - m (m nested) | -0.0005 | -0.0005 | +0.0006 | -0.0024 | no |
| D2 same margin, larger on sparse images, m*min(1, n0/n) (nested) | +0.0003 | +0.0004 | -0.0016 | -0.0001 | no |
| D3 lower tau on sparse images for every candidate (m, n0 nested) | +0.0009 | +0.0006 | +0.0075 [+0.001,+0.016] | +0.0030 [-0.002,+0.010] | no |
| D3 with m = 0.04, n0 = 4 fixed | +0.0005 | +0.0002 | +0.0082 [+0.001,+0.016] | +0.0100 [+0.002,+0.021] | met, not counted |
| D4 images with <= n0 candidates keep what the fixed rule keeps (nested / n0 = 6) | 0.0000 / -0.0002 | -0.0001 / -0.0004 | +0.0013 / +0.0060 | 0.0000 / +0.0070 | no / met, not counted |
| D5 images with <= n0 candidates use the fixed rule alone (nested) | -0.0001 | -0.0001 | -0.0019 | 0.0000 | no |
| M1 image-balancing weights R*(10/n)^a, a = 0.5 / 1 | +0.0004 / +0.0002 | +0.0003 / 0.0000 | +0.0022 / +0.0040 | +0.0069 / +0.0060 | no |
| M2 monotone in crowding / plus in confidence | +0.0005 / -0.0001 | +0.0005 / 0.0000 | +0.0005 / -0.0024 | +0.0110 / +0.0053 | no |
| M3 without position / without member-agreement descriptors | -0.0012 / +0.0006 | -0.0013 / +0.0004 | +0.0017 / +0.0042 | -0.0045 / +0.0066 | no |
| G + image descriptors from gray image and maps (40 / 18 gray / 22 maps / 9 chosen / member disagreement) | 0.0000 / -0.0012 / -0.0001 / 0.0000 / 0.0000 | -0.0002 / -0.0014 / -0.0002 / -0.0001 / -0.0001 | +0.0032 / +0.0018 / +0.0017 / +0.0026 / +0.0029 | +0.0106 / +0.0080 / +0.0071 / +0.0133 / +0.0099 | no |
| R1 + 16 within-image relative descriptors | +0.0009 | +0.0006 | +0.0068 [-0.000,+0.015] | +0.0127 [+0.002,+0.024] | met, not counted |
| U benefit of the doubt: bagged mean + c*spread (c nested) | +0.0007 | +0.0005 | +0.0066 | +0.0068 | same as the bagged mean (c mostly 0) |
| 8 null variants (noise columns, equivalent hyper-parameters) | -0.0008..+0.0011 | -0.0006..+0.0010 | -0.0051..+0.0046 | -0.0022..+0.0071 | 0 of 8 |
| **Bagged K1, 48 refits** | **+0.0011** [-0.0001,+0.0023] | **+0.0008** [-0.0004,+0.0021] | **+0.0081** [+0.002,+0.016] | **+0.0091** [+0.001,+0.019] | **met** |
| Bagged, refits 0-23 / 24-47 | +0.0015 / +0.0004 | +0.0012 / +0.0002 | +0.0072 / +0.0046 | +0.0078 / +0.0082 | met / no |
| Subagged, 6 members on 80% of the images | +0.0005 | +0.0002 | +0.0067 | +0.0080 | met |
| Random forest instead of boosting (2 seeds) | +0.0006 / +0.0001 | +0.0004 / -0.0002 | +0.0051 / +0.0069 | +0.0016 / +0.0053 | no / met |

Why the "met, not counted" rows do not count:
- **D3 and D4 presets.** 9 of 84 (m, n0) cells meet the numbers, and the version with nested settings fails LPO. At fixed tau 0.25 the D3 preset costs -0.0011 on 2011-2017, because sparse early images want 0.24. On the four re-splits the D3 preset gives -0.0024, -0.0019, +0.0025, +0.0015 on the block; D4 with n0 = 6 gives +0.0052, +0.0010, -0.0004, +0.0022.
- **R1.** At matched thresholds it is +0.0044 (folds) and +0.0059 (LPO) with -0.0005 on 2011-2017. Its nested LPO threshold moved from 0.23 to 0.21, and that step alone is worth +0.007 for plain K1.
- **The LPO criterion with a nested threshold is weak.** 4 of 8 null variants gain +0.005 or more there.

## 4. Bagged K1 in detail
- **Registered split, B = 48.** Overall 0.4522, 2011-2017 0.4526, 2018-2022 0.4433, LPO 0.4430. Folds .4400/.4566/.4527/.4620/.4494 (K1 .4400/.4578/.4523/.4581/.4466). Images with 1/2/3 records .4544/.4613/.4440 (K1 .4523/.4616/.4429).
- **At tau 0.25 against K1 at 0.25.** Overall -0.0001, 2011-2017 -0.0004, 2018-2022 +0.0073 [-0.002,+0.018], LPO +0.0049.
- **Refiner shapes on real label maps (score_file).** Overall 0.4566 against 0.4558 (+0.0008 [-0.0003,+0.0020]); 2018-2022 0.4414 against 0.4328 (+0.0087 [+0.002,+0.016]); LPO 0.4410 against 0.4315.
- **Member sets.** 3 of 4 disjoint sets of 12 and 16 of 20 random sets of 24 meet the numbers.
- **What changes.** 8 decisions on the block (5 newly kept with mean u 0.52, 3 newly dropped with mean u 0.11). Correlation of score with truth 0.6645 -> 0.6688. Inside 2011-2017, images with <= 6 candidates lose 0.003-0.005, 2017 loses 0.0036 [-0.010,+0.003], crowded images gain 0.002.
- **Re-splits (4 other day-grouped splits, 8 refits each, against that split's own single K1).**

| split | overall | 2018-2022 folds | 2018-2022 LPO | bagged level on block, nested taus | single K1 level |
|---|---|---|---|---|---|
| registered (B = 48) | +0.0011 | +0.0081 | +0.0091 | 0.4433 | 0.4352 |
| re-split 1 | +0.0016 | +0.0008 | +0.0061 | 0.4421 | 0.4413 |
| re-split 2 | +0.0005 | +0.0011 | -0.0006 | 0.4431 | 0.4420 |
| re-split 3 | +0.0008 | +0.0000 | +0.0068 | 0.4409 | 0.4409 |
| re-split 4 | +0.0017 | +0.0036 | +0.0001 | 0.4449 | 0.4412 |

  At tau 0.25 the bagged model scores 0.4376/0.4362/0.4409/0.4362/0.4396 on the block (fixed rule 0.4390; single K1 0.4303/0.4342/0.4307/0.4357/0.4416). Overall at tau 0.25: -0.0001, +0.0018, +0.0024, +0.0011, +0.0006.
- **Reading.** Bagging gives about +0.001 overall on every split and puts the block at the fixed rule's level, not above. The +0.008 on the registered split is mostly that split's single fit being a low draw.
- **Threshold.** With the bagged score, overall is 0.4517-0.4522 for every tau from 0.20 to 0.26 (single K1: 0.4508-0.4521, with a bump at 0.24-0.25). The only lever left for the block is a lower threshold. No image-based rule can apply it selectively, because sparse 2011-2017 images want the opposite.

## 5. Not done, and compliance
- Re-splits used 8 refits, not 48. R1 and the G variants were not re-split.
- CPU only, at most 2 processes, writes only in scratch/ws/x_late.
- Year and site were used for splits and analysis only.
- No test annotation was opened. The test candidates' descriptors (our own predictions) were read once, to count changed decisions.
