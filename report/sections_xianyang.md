# Report section drafts: Introduction, Data, Post-processing, Results and error analysis (owner: xianyang)

Draft text for the 4-page report (TEAM_ROLES.md section 7); to be pasted into the organizers' Overleaf template.
Every number is from `EXPERIMENTS.md`, `results/evidence/final_sprint_1112/SUMMARY.json` or `results/oof_final/`
(final-pipeline statistics and figures: `python analysis/report_figs.py --final results/oof_final report/figures/final`). PQ values are five-fold
out-of-fold on the 707 training images, scored with the organizers' self-evaluation rule, unless stated otherwise.
Items marked **TBD** wait for the Nov 13 leaderboard readings.

---

## 1. Introduction

Solar filaments are clouds of cool, dense plasma held above the photosphere by magnetic fields; in H-alpha
full-disk images they appear as dark, elongated structures. They sit at the origin of eruptions that drive space
weather, so reliable, automatic outlines of individual filaments are needed. The challenge asks for one mask per
filament in 2048 x 2048 GONG H-alpha images, with no pixel shared between masks, and ranks submissions by Panoptic
Quality (PQ): the sum of the IoU of matched pairs (IoU > 0.5) divided by TP + FP/2 + FN/2. Missing a filament and
predicting a spurious one cost the same, so counting filaments correctly matters as much as drawing their edges.

Our solution has three stages. (1) A U-Net with an EfficientNet-B4 encoder predicts, for each pixel, the fraction
of experts who marked it as filament; connected regions of this map are candidate filaments. (2) A small
gradient-boosted model decides which candidates to keep, from 50 descriptors of the candidate and of the image
around it. (3) A small second network refines the outline of each kept candidate. On out-of-fold predictions the
pipeline reaches PQ 0.4566, against 0.4409 for the single-model baseline with a fixed keep rule. We also report
what did not work and where the remaining error comes from: most of it is expert-to-expert disagreement about
which filaments to annotate, which no image-based model can remove.

## 2. Data

**Images and annotations.** The training set has 707 images and 8199 filament annotations in 1154
annotator-image records: 411 images were annotated by one expert, 145 by two and 151 by three. We use only the
segmentation polygons. The test set has 180 images. We never use the public MAGFiLO annotations of the test
images, and inference uses each test image alone; no decision uses information from the file name (date, station).

**Experts disagree.** Scoring one expert's annotation of an image against another's gives PQ of about 0.34, lower
than our model scores against either. Grouped by annotator batch (24 groups with at least 20 records), the
baseline's PQ ranges from 0.363 to 0.524: some experts annotate few filaments per image (4.7 per record), others
many (8.4). The model learns an average annotator. This sets the scale for every result below: a spread of 0.16
between annotators, against improvements of 0.002 to 0.008 per method.

**Soft targets.** For an image with several records, the training target of the dense model is the per-pixel
mean of the experts' masks, so disagreement is kept as an intermediate value instead of being forced to 0 or 1.

**Validation split.** Filaments persist for days, and the same image appears once per annotator. We therefore
split by observation date into five folds: all records of an image, and all images of a day, fall in the same
fold. Every score in this report is out of fold: the dense model, the refiner and the keep model that process an
image were trained without that image's fold.

**Seed noise.** Retraining the baseline with a different random seed changes the two-fold PQ by 0.001 in total but
by up to 0.0065 on a single fold, in opposite directions. We therefore required +0.005 over five folds with at
least four folds improving before adopting a change, and we confirmed the adopted ones with paired bootstrap
intervals and independent re-implementations.

## 3. Post-processing: from probability maps to instances

**Test-time augmentation and averaging.** Each dense model predicts the eight flips and 90-degree rotations of the
image at 1536 px and averages them. For the test set, two models trained on all 707 images (two seeds) are
averaged, and the map is upsampled to 2048 px.

**Candidates.** Pixels with probability >= 0.4 are filament pixels. Fragments belong to the same candidate if a
morphological closing of radius 8 joins them, or if they are connected through pixels with probability >= 0.1
("weak-link grouping"). Closing and weak links only decide grouping; they never add pixels to a mask. The
annotation protocol draws each filament as one connected polygon, including faint stretches where the model's
probability is low; without grouping, one filament becomes several predictions and each extra piece is a false
positive. Because every mask is a set of pixels of one label map, masks cannot overlap.

**Keep rule.** The baseline keeps a candidate if its area is >= 300 px and its mean probability is >= 0.85. The
final pipeline replaces this with a learned rule (Section "Method"): a gradient-boosted regressor predicts u, the
candidate's expected matched IoU averaged over the image's records, and the candidate is kept if u > 0.25. The
threshold follows from the metric: keeping a candidate adds u to the numerator of PQ and 1/2 to the denominator
per record, so keeping pays exactly when u > PQ/2.

**Outline.** Grouping and keeping are computed on the 0.4 region, but the written mask is tighter. In the light
variant only pixels with probability >= 0.48 are written (+0.0017, all five folds improve). In the full variant a
refiner network looks at a 384 x 128 crop along each kept candidate; its probability is averaged with the dense
model's and thresholded at 0.5, keeping the part connected to the original candidate.

**What the simple alternatives give.** On the baseline's maps, dilating every mask by 1 px costs 0.032 and eroding
by 1 px costs 0.014; filling holes, opening and closing change nothing. Size-dependent thresholds give +0.0010
(an oracle that picks each candidate's best threshold would give +0.024, but candidate area does not predict it).
A first learned filter with 11 descriptors gave +0.0001; the gain of the final keep rule comes from descriptors it
lacked, chiefly the candidate's standing among the other candidates of the same image.

## 4. Results and error analysis

**Score ladder** (five-fold out-of-fold, 707 images, 1154 records; strictly nested: the keep model and its
threshold for fold k never see fold k).

| Pipeline | PQ | Public LB |
|---|---|---|
| Baseline: one model, fixed keep rule | 0.4409 | 0.39 |
| + two-model average, weak-link grouping, written at >= 0.48 | 0.4456 | 0.39 |
| + refined outlines (fixed keep rule) | 0.4490 | — |
| Learned keep rule, written at >= 0.48 (light variant) | 0.4537 | **TBD** |
| Learned keep rule + refined outlines (final) | 0.4566 | **TBD** |

The learned keep rule keeps 7% fewer candidates than the fixed rule (4489 against 4824); false positives fall by 17%
and true positives by 2% (final pipeline: 5197 TP, 2130 FP, 3002 FN; fixed rule with refined outlines: 5300 / 2560 /
2899).

**Rubric statistics** (final pipeline, out of fold; baseline in brackets). The 6673 overlapping
ground-truth/prediction pairs have mean IoU 0.603 and mean Dice 0.735 (baseline: 7207 pairs, 0.583, 0.715;
Fig. `final/fig_overlap_scores`). Of the 8199 ground-truth filaments, 6267 overlap exactly one prediction, 192 are
split over two or more (1:n; baseline 364) and 1740 overlap none (1414); of the 7326 predictions (counted once per annotator record), 6220 overlap
one filament, 220 cover two or more (n:1; 194) and 886 overlap none (1205) (Fig. `final/fig_overlap_degrees`). The
final pipeline trades overlap for precision: fewer spurious and split predictions, more filaments left without any
prediction, and tighter outlines on the pairs that remain.

**Why filaments are missed** (final pipeline, 3002 misses; Fig. `final/fig_fn_causes`). 1167 were found as
candidates but removed by the keep rule (median size 790 px); 573 were not detected at all (568 px); 537 were drawn
too small (median size 2555 px) and 448 too large (752 px); 152 were merged with a neighbour, 99 split and 26 have
another shape error. Small filaments dominate the misses, and they are also where experts disagree most: of the 1262
misses that overlap a prediction, 540 (43%) have a best prediction that is a true positive for another expert of the
same image. Fig. `final/fig_examples` shows three well-scored and three poorly scored images.

**Remaining error.** If every keep/remove decision were right, the pipeline would score 0.515; it scores 0.457.
The difference is mostly "this expert drew it, that one did not", which cannot be read from the image. Lowering
the threshold finds no new useful candidates: of 541 candidates that appear only below 0.4, 5 are worth keeping.

**Local versus leaderboard.** Every honest pipeline we could check, ours and seven public ones, scores about
0.87-0.90 times its local PQ on the public leaderboard. Our scorer matches the organizers' self-evaluation
notebook exactly (PQ 0.4334, 1051 TP, 567 FP, 596 FN on a held-out fold, both ways). Test and validation images
look the same to the model (kept filaments per image 7.27 vs 7.18, same size and confidence distributions), so the
gap lies on the annotation side of the test set. We select models on out-of-fold PQ only; small local gains did
move our leaderboard rank in the predicted direction (364 -> 353 -> 331 over three verified changes).

**What did not work** (all evaluated against a same-machine control; none passed +0.005 with 4 of 5 folds).
Training changes: small-filament crop sampling and area-weighted loss (-0.003 to +0.002), a boundary output channel
(+0.002, cuts along it never helped), centre-line auxiliary supervision, 2048 px input, self-training on unlabelled
GONG images, label transfer from neighbouring frames, a third dense model. Ensembling: averaging more models
removes false positives and true positives in equal measure. Post-processing: mask reshaping, size-dependent
thresholds, multi-scale inference, a model that merges fragments (wrong merges cost more than right ones gain),
a decision layer over thresholds, per-annotator tuning (annotator cannot be inferred from the image).

**Limitations.** (1) The learned keep rule was an exploratory experiment recorded after the fact, not
pre-registered; it was then verified by an independent re-implementation with its own 66 descriptors (same gain),
label-permutation controls (gain disappears), 10 alternative fold splits and 19 model settings (all positive). We
state its gain as +0.007 (strictly nested), not the +0.008 first measured. (2) It gives no gain on images from
2018-2022 (solar minimum, 2-5 candidates per image; -0.009, interval -0.021 to +0.003); these hold about 5% of the
filaments. (3) Its gain is larger on images with several annotators (+0.004 / +0.010 / +0.011 for one / two /
three); if the test set's mix differs, so will the gain. (4) The refiner's paste-back settings were chosen on all
five folds; with that choice nested too, the final score is about 0.455. (5) The public leaderboard uses 90
images; a single reading has noise of about +-0.004.

**Cross-machine reproduction.** From a fresh clone and a fresh environment (pinned `requirements.txt`, Python 3.12)
on an Apple-silicon Mac, the weights' SHA256 match and the pipeline gives 1163 filaments on the 180 test images, the
same count as the submitted files (RTX 5070 Ti, float32). On the Apple GPU (MPS, float32, 9.6 min) 1161 filaments
match a submitted one; 2 borderline candidates leave and 2 enter. The matched masks differ by a few edge pixels
(median 1 px, 95th percentile 5 px): mean IoU 0.9989, 99.3% of masks at IoU >= 0.99, 63.6% at >= 0.999, minimum
0.969. For the light variant: mean IoU 0.9996, all masks >= 0.99, 86.1% >= 0.999. The strict criterion (every mask
at IoU >= 0.999) is therefore not met on the Mac. The cause is the machine, not the GPU: the Mac CPU run (float32,
3 h 56 min) gives exactly the same 1163 masks as the Mac MPS run (every mask IoU 1.0), so both differ from the
submitted file in the same way. We had seen the same effect with the baseline: OpenCV's image resize rounds
differently on Apple silicon and on x86, which moves pixels that sit on the 0.48 / 0.5 thresholds.
