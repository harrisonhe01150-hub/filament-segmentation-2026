# Test-set files of the final pipeline (submitted 2026-11-13 00:02 and 00:06 UTC)

Copies of the two files the scheduled submitter on Harrison's machine sends to Kaggle
(`scratch/sub_1113/final/`, made 2026-11-12 with the code of commit bec108b, float32 on the RTX 5070 Ti;
EXPERIMENTS.md, "code review"). Each has 1163 filaments on the 180 test images.

Public leaderboard (2026-11-13 00:06 UTC, 1504 teams): both display 0.40; full rank 210, light rank 219
(our best before was 0.39, rank 334). EXPERIMENTS.md, last section.

| File | Variant | SHA256 |
|---|---|---|
| `K1_refine_fp32.csv` | full: learned keep rule + refiner outlines (`--refiners weights/refiner_f*.pt`) | `133f2c51ffb45a9cf59349a5ac3dec35a26dae5eeb842afe13afbbf1f474cd51` |
| `K1_light_fp32.csv` | light: learned keep rule + stage-1 mask drawn at 0.48 (`--no-refiner`) | `8c6d4fa83cf99a808d6a7f61ee72def01360b22ad2e1b890ae3768d800ee2ffa` |

To check a re-run (HANDOFF.md section 7 commands, with `--fp32`) against them:

    python analysis/compare_csv.py outputs/submission.csv results/test_submissions_1113/K1_refine_fp32.csv

Every mask with IoU >= 0.999 counts as reproduced. Float16 (the CUDA default without `--fp32`) moves a few
borderline candidates.
