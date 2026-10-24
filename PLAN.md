# Team Plan

Plan for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026/overview).
Written 2026-10-22. Final deadline: **Nov 15, 11:59 pm Central time**. We aim to be fully done on **Nov 14**.

## What we are building

Input: one grayscale 2048 x 2048 H-alpha image of the Sun.
Output: one mask per solar filament in that image, with **no pixel shared between two masks**.

We submit three things at the end:

1. A Kaggle submission CSV (one row per filament, COCO RLE counts)
2. A public GitHub repo (this one) with `requirements.txt`, the final model weights, and one notebook that runs
   the whole pipeline without retraining
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
7. Never commit data, training checkpoints, `kaggle.json` or `.env` (already in `.gitignore`). The one exception
   is the final model's weights in `weights/` (see "Final weights" below).

## Roles

Agreed on Oct 24; details, experiment rules and the day-by-day schedule are in [TEAM_ROLES.md](TEAM_ROLES.md).

| Who | Experiment | Also owns |
|---|---|---|
| **Zichang** | A: spine auxiliary output | Report "Method"; Kaggle submissions; Google form |
| **Harrison** | B: small-filament crop sampling and loss weighting | Repo, final notebook, reproducibility |
| **xianyang** | C: boundary channel and splitting | Single evaluation point; report "Results and error analysis" |

Everyone reviews each other's pull requests, so each of us can explain the whole pipeline at the end.

**Kaggle team:** Zichang Lu (team leader), xianyang zeng, Harrison He. The 5 submissions per day are shared by
the whole team.

## Timeline

### Phase 1: Setup and first honest score (Oct 23 to Oct 27)

- [x] Everyone joins one Kaggle team (leader: Zichang Lu)
- [x] Download the data; agree on the folder layout under `data/` (not committed): `data/MAGFiLO_1.0_Kaggle_2026/`
- [x] **Validation split (A):** 5 folds of about 20% each, grouped by observation day (`fold_of_files` in
      `src/train_seg.py`), so the several annotations of one photo stay on the same side. Filaments last for days, so
      neighbouring days in different folds still make the local score somewhat optimistic.
- [x] **Local scorer (A):** reproduce the PQ from the [official self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook)
      and check that it gives the same numbers on a few test cases. `src/common.py` `PQ` follows the same rules
      (checked by reading both); a numeric cross-check is still open.
- [x] **Baseline (B + C):** v2 EfficientNet-B4 U-Net, five-fold OOF PQ 0.44, public leaderboard 0.39
- [x] `requirements.txt` with pinned versions

**Done when:** we have a local PQ number and one honest leaderboard submission of about 0.30 or higher.

### Phase 2: Improve (Oct 28 to Nov 8)

The experiments we are actually running (A, B, C) and their pass rule are in [TEAM_ROLES.md](TEAM_ROLES.md).
Original idea list, roughly in order of expected payoff:

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
- [ ] Nov 9 to 10: final weights committed to `weights/`; the notebook loads them and does not retrain
- [ ] Nov 10: fresh clone into an empty folder, run the notebook, get the same predictions with no extra files
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

## Final weights

The organizers require the model weights in the repo, so the notebook reproduces our predictions without
retraining.

- Only the final model goes in `weights/`. Training checkpoints stay out of git.
- GitHub rejects files over 100 MB. If ours is bigger, save it in half precision or use Git LFS.
- The notebook loads from `weights/` with a relative path.

## Compute

- Kaggle notebooks: free GPU hours every week, per person
- Harrison's local machine: RTX 5070 Ti

## Open questions

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
