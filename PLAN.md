# Team Plan

Plan for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026/overview).
Written 2026-10-22. Final deadline: **Nov 15, 11:59 pm Central time**. We aim to be fully done on **Nov 14**.

## What we are building

Input: one grayscale 2048 x 2048 H-alpha image of the Sun.
Output: one mask per solar filament in that image, with **no pixel shared between two masks**.

We submit three things at the end:

1. A Kaggle submission CSV (one row per filament, COCO RLE counts)
2. A public GitHub repo (this one) with `requirements.txt` and one notebook that runs the whole pipeline
3. A 4-page technical report (organizers' Overleaf template), sent with the repo link through the organizers' Google form

## How we are judged

- **70% quantitative:** Panoptic Quality (PQ), plus the distributions of Dice and IoU, plus how often we split one
  filament into pieces (one-to-many) or merge several into one (many-to-one).
- **30% qualitative:** how clearly the report explains the pipeline, how the masks look on the images, and code
  quality (modularity, documentation).
- Only the top 3 teams get prizes ($3,000 total). No Kaggle medals or points.

PQ = (sum of IoU over matched pairs) / (TP + 0.5 FP + 0.5 FN).
Every extra blob and every missed filament costs us. **Getting the number of filaments right matters as much as
getting their edges right.**

## Ignore the public leaderboard

All 180 test images are also in the public MAGFiLO 1.0 dataset, which ships with their annotations
([discussion](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/728062)).
The block of 0.55 scores comes from a public notebook that uses those annotations, and other high scores may too.
Honest teams in the discussions report PQ of about 0.35 to 0.50, and the host wrote that they take 0.41 "very seriously".
The host has said the leaderboard is only a first filter; the final ranking comes from the rubric above.

**Our yardstick is our own local validation score.** We use the leaderboard only as a sanity check.

## Ground rules

1. **Never use the public MAGFiLO annotations of test images, in any form.** Train only on the competition's train
   set. If we ever add external data, we first remove all 180 test filenames from it.
2. **Inference uses the single test image only.** Neighbouring GONG frames may be used for training, not for prediction
   ([host reply](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/743385)).
3. Spine, bbox, area and category labels from the train JSON are allowed as extra training targets
   ([host reply](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/736543)).
4. **Masks in one image must not overlap**, or the submission errors out and wastes a slot. Always run the overlap
   check before submitting.
5. **AI-assisted coding is allowed, but each of us must be able to explain and reproduce every part we own.** The host
   may disqualify solutions that rely on AI-generated code the team does not understand
   ([host reply](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/724503)).
6. Team limit is 5 submissions per day. Log every submission in `EXPERIMENTS.md`.
7. Never commit data, model weights, `kaggle.json` or `.env` (already in `.gitignore`).

## Roles

| Role | Owns | Owner |
|---|---|---|
| **A. Data & evaluation** | Data loading, train/validation split, local PQ scorer, RLE encode/decode, overlap check, submission writer | _TBD_ |
| **B. Models** | Training and inference: baseline U-Net first, then stronger models | _TBD_ |
| **C. Post-processing & report** | Turning pixel predictions into clean, non-overlapping filaments; final notebook; technical report | _TBD_ |

Everyone reviews each other's pull requests, so each of us can explain the whole pipeline at the end.

## Timeline

### Phase 1: Setup and first honest score (Oct 23 to Oct 27)

- [ ] Everyone joins one Kaggle team; pick a team leader
- [ ] Download the data; agree on the folder layout under `data/` (not committed)
- [ ] **Validation split (A):** hold out about 20% of training images. Split by observation date, not by image id:
      the same photo appears several times (once per annotator), and filaments last for days, so photos close in time
      look alike. Both must stay on the same side of the split.
- [ ] **Local scorer (A):** reproduce the PQ from the [official self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook)
      and check that it gives the same numbers on a few test cases
- [ ] **Baseline (B + C):** U-Net for "filament or not" per pixel, then split into connected components, drop tiny
      blobs, encode as RLE, run the overlap check, submit
- [ ] `requirements.txt` with pinned versions

**Done when:** we have a local PQ number and one honest leaderboard submission of about 0.30 or higher.

### Phase 2: Improve (Oct 28 to Nov 8)

Ideas, roughly in order of expected payoff:

1. **Count the filaments right (C):** tune the minimum blob size; join fragments that belong to one filament
   (morphological closing, or merging nearby pieces with the same orientation); separate filaments that touch
2. **Better pixel model (B):** stronger encoder (e.g. EfficientNet / ConvNeXt backbone), higher input resolution or tiling,
   augmentation (rotation, flips, brightness)
3. **Instance model (B):** Mask R-CNN or similar that predicts each filament directly, with score-ordered painting
   so masks never overlap ([discussion](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/735377))
4. **Use the spine labels (B):** extra output head that predicts the filament spine, to keep thin parts connected
5. **Multiple annotators (A):** decide how to use several annotations of the same photo (keep all, or merge them)

Every experiment gets one line in `EXPERIMENTS.md`: date, who, what changed, local PQ, leaderboard PQ (if submitted), commit.

**Done when:** the model is frozen on Nov 8 at our best local PQ. Target: 0.40 or higher.

### Phase 3: Freeze and deliver (Nov 9 to Nov 15)

- [ ] Nov 9: final model chosen; no more training
- [ ] Nov 9 to 10: final notebook runs end to end on a clean environment from `requirements.txt`
- [ ] Nov 10 to 12: 4-page report: data, pipeline, post-processing, results (local PQ, Dice / IoU distributions,
      split / merge counts), examples of good and bad masks, what did not work
- [ ] Nov 12: code cleanup: clear modules, docstrings, README instructions to reproduce
- [ ] **Nov 13: make this repo public** (Settings → Danger Zone → Change visibility)
- [ ] **Nov 14: submit the Google form** (report PDF + repo link); select our 2 final Kaggle submissions
- [ ] Nov 15: buffer only

## Repo conventions

- Branch per task (`<name>/<task>`), pull request into `main`, one teammate reviews
- Code in `src/`, notebooks in `notebooks/`, report in `report/`
- Keep the final notebook thin: it calls functions from `src/`

## Compute

- Kaggle notebooks: free GPU hours every week, per person
- Harrison's local machine: RTX 5070 Ti

## Open questions

- Who is the Kaggle team leader, and which role does each of us take?
- How do we use the multiple annotations of the same photo for training?

## Links

- Competition: [Overview](https://www.kaggle.com/competitions/filament-segmentation-2026/overview) ·
  [Data](https://www.kaggle.com/competitions/filament-segmentation-2026/data) ·
  [Leaderboard](https://www.kaggle.com/competitions/filament-segmentation-2026/leaderboard) ·
  [Rules](https://www.kaggle.com/competitions/filament-segmentation-2026/rules)
- [Official self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook)
- [Organizers' remarks](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/724503)
- [Realistic scores discussion](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/745880)
- PQ paper: Kirillov et al., *Panoptic Segmentation*, CVPR 2019 (doi 10.1109/CVPR.2019.00963)
