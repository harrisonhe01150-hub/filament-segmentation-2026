# Report section draft: Reproducibility (owner: Harrison)

Draft text for the "Reproducibility" part of the 4-page report (TEAM_ROLES.md section 7). Numbers marked
**TBD** are filled in at the model freeze (Nov 8–9) from the final weights; everything else is measured.

---

## Reproducibility

**Code and environment.** The repository contains the full pipeline: training (`src/train_seg.py`), prediction from
saved weights (`src/predict.py`), post-processing and CSV export (`src/make_submission.py`), the PQ metric
(`src/common.py`) and one notebook (`notebooks/end_to_end.ipynb`) that runs weights → test predictions → instances
→ CSV → overlap check → figures. `requirements.txt` pins every package (Python 3.12, PyTorch 2.11.0,
segmentation-models-pytorch 0.5.0, timm 1.0.30, OpenCV 5.0, pycocotools 2.0.11, NumPy 2.5).

**Weights.** The final model's weights (one file per fold, **TBD** folds) are stored in `weights/`, so the notebook
reproduces the submitted predictions without retraining. Each fold model is 81.7 MB in float32. The submitted
CSV is produced from exactly these files.

**Hardware and run time.** Training: one fold of 50 epochs takes about one hour on a single RTX 5070 Ti (16 GB;
about 70 s per epoch at batch 4) and about 1.7 hours on a Kaggle T4. Prediction for all 180 test images with
**TBD** fold models and 8-way test-time augmentation: **TBD** s on the RTX 5070 Ti; on a CPU, about 16 s per
image per fold model.

**Determinism.** The submitted CSV is produced in float32 on every device (`--fp32`). Regenerating the team's
earlier two-fold submission from its weights on a CPU (float32) instead of the original float16 GPU run
reproduced 1252 of 1255 filaments with mean IoU 0.99965 (minimum 0.9935); 3 borderline filaments dropped out
and 1 appeared, which is why float32 is used for the final run. The frozen probability maps reproduce that
submission's CSV exactly, and the five-fold out-of-fold PQ re-scores to the recorded 0.4409. Prediction is
deterministic on a given device: reloading the saved weights on the training GPU
reproduces the probability maps produced during training bit for bit. Across devices, CPU (float32) and GPU (float16
autocast) probability maps differ by at most 1/255 per pixel; the resulting masks agree with IoU 0.9996 at the 0.4
threshold. A fresh clone on Windows, installed only by the README steps, passes `pip check`, both self-tests and the
notebook end to end (Oct 24). A clean-clone run on a different machine (Apple MPS) is accepted when every mask matches the submitted
CSV with IoU ≥ 0.999 (**TBD**: result of the Nov 10 cross-machine test). GPU training itself is not bit-for-bit
deterministic, which is why the weights are part of the repository.

**Metric.** Our local PQ implementation follows the organizers' self-evaluation notebook (IoU > 0.5 pairs are true
positives; unmatched predictions are false positives and unmatched annotations false negatives; scored per
annotator-image). On a held-out fold (v2, fold 0, 142 images), re-implementing the notebook's computation step by
step from the RLE tables (`analysis/official_pq_check.py`) gives exactly the same result as our scorer: PQ 0.4334,
1051 TP, 567 FP, 596 FN.

**Data use.** Training uses only the competition's training images and annotations (all annotators' masks; spine,
area and category fields are not used by the baseline). Inference uses each test image alone. Public MAGFiLO
annotations of the test images are never used.
