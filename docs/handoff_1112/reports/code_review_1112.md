# Code review of the final pipeline (2026-11-12) and its independent verification

13 findings, all confirmed real by the verifier. Status after commit bec108b: fixed unless noted.

## 1. [medium] `src/keep.py:84` — FIXED (symmetric member descriptors, table and candidates rebuilt)

**Finding:** The descriptors mem_a / mem_b are positional, so the keep decision depends on which dense model is passed first, and the two members at prediction time (full s2026, full s2027) are not the two the table was built with (Kaggle-trained set, 5070 Ti set, same seed).

**Scenario:** Swap the two paths of --probs / --dense (or DENSE in the notebook): predict_final.py writes 1159 rows instead of 1158; 6 masks vanish and 7 new ones appear in 13 of 180 test images, with no warning. mean(mem_a - mem_b) is +0.0155 on the test table against -0.0058 in weights/keep_table.npz.

**Verifier:** Code: keep.py:84 takes flat[0] / flat[1] and line 99 writes ma, mb as two columns; the other member columns are symmetric (checked on the table). Seeds do not depend on the order (the mean is commutative), so these two columns are the whole effect.
CLI, run by me on CPU with the saved test maps (v_light_ab.csv / v_light_ba.csv): documented order = 1158 rows, sha256 identical to scratch/sub_1113/B_k1_light.csv; reversed order = 1159 rows; 6 masks only in the documented order, 7 only in the reversed one, 13 of 180 images differ; nothing in the log mentions the order.
Out of fold (my own cross-fit, model for fold k fitted on the other four day-grouped folds, tau fixed 0.25, held-out fold's two columns swapped): 49 of 6788 decisions flip (47 images); PQ 0.4521 -> 0.4517 (-0.0003); per fold 0.4411/0.4586/0.4545/0.4575/0.4482 -> 0.4413/0.4580/0.4537/0.4573/0.4480; 1/2/3-record images 0.4518/0.4623/0.4457 -> 0.4513/0.4620/0.4455; paired bootstrap 5-95% -0.0010..+0.0003. So accuracy is not measurably affected, as the reviewer says.
The model does use both columns: 59 splits on mem_a and 88 on mem_b of 3629 in the final model.
Which model is which (recomputed from the maps, members.json): in the training table mem_a = Kaggle-trained set (runs/v2, v2b) and mem_b = 5070 Ti set, both seed 2026; in the test table mem_a = full model s2026 (5070 Ti), mem_b = full model s2027. The 5070 Ti seed-2026 recipe is therefore 'b' in training and 'a' at prediction, so no order preserves the meaning. mean(mem_a - mem_b) = -0.0058 in weights/keep_table.npz and +0.0155 on the test table. Test table: 13 decisions depend on the order (7 added, 6 removed).
Evidence files: C:\Users\harri\code\filament-segmentation-2026\scratch\ws\x_review_verify\keep_table_native.json, members.json, sym_fix.json, sym_refine.json, v_light_ab.log, v_light_ba.log.

**Suggested fix:** keep.py line 99: write max(ma, mb), min(ma, mb) in place of ma, mb (mem_a = larger member mean, mem_b = smaller), then rebuild the tables (analysis/keep_bank.py, keep_fit.py --table), update the hash in weights/README.md and regenerate the candidate files. This is the uncommitted scratch/keep/oof_sym2 variant; I re-derived it from the shipped table and got a bit-identical model.
Checked honestly (cross-fitted by day-grouped fold; paired bootstrap over images, 2000 resamples): no gain and no loss.
- Stage-1 masks, tau 0.25: 0.4521 vs 0.4521 (bootstrap -0.0008..+0.0008); per fold 0.4410/0.4579/0.4536/0.4585/0.4492 vs 0.4411/0.4586/0.4545/0.4575/0.4482; 1/2/3-record 0.4523/0.4632/0.4447 vs 0.4518/0.4623/0.4457.
- Strictly nested tau: 0.4517 vs 0.4510 (+0.0007, bootstrap -0.0003..+0.0018); per fold 0.4418/0.4579/0.4536/0.4573/0.4477 vs 0.4400/0.4578/0.4523/0.4581/0.4466; 1/2/3-record 0.4536/0.4612/0.4437 vs 0.4523/0.4616/0.4429.
- Final refiner shapes on real label maps, tau 0.25: 0.4566 vs 0.4563 (+0.0003, bootstrap -0.0005..+0.0011); per fold 0.4477/0.4619/0.4568/0.4625/0.4539 vs 0.4479/0.4623/0.4573/0.4612/0.4527.
Cost: 22 of 1727 test decisions change (1158 -> 1162 kept), after which 0 depend on the order.
If the candidate files must not change this late: leave the code and state in README, the CLI help and the notebook that the order (s2026 first) is part of the pipeline. That only documents the dependence.

## 2. [low] `README.md:89` — FIXED

**Finding:** The documented command for the final pipeline (also the usage in src/predict_final.py:16-18) has no --fp32, although the submitted files were made with it.

**Scenario:** A judge with a CUDA GPU copies the README command: dense models and refiners run in float16 autocast, and the CSV is the float16 variant (1158 rows), not the submitted float32 one (1159 rows; EXPERIMENTS.md records 2 candidates in, 1 out, plus small outline differences).

**Verifier:** README.md:89 and predict_final.py:16-18 have no --fp32, while the baseline command at README.md:83 has it and the notebook sets FP32 = True. Flag handling: predict_final.py:53 and :149 pass fp16 = not fp32; train_seg.predict and refine.predict_crops use float16 autocast only on CUDA, so on CPU / MPS the flag changes nothing and on CUDA the README command is the float16 path.
The two variants differ (analysis/compare_csv.py on existing files): A_k1_refine.csv (from the saved float16 maps) 1158 rows vs A_k1_refine_fp32.csv 1159 rows: 1157 matched, 1 only in float16, 2 only in float32, lowest matched IoU 0.955, 73% of matched masks at IoU >= 0.999. Light variant: 1157 matched, 1 / 2 unmatched, lowest IoU 0.995. Float32 GPU vs float32 CPU: 1159 of 1159 matched, lowest IoU 0.9994.
Two wording corrections: (a) nothing has been submitted yet - scratch/ws/main/submit_1113.py lists the two *_fp32.csv files for 00:02 UTC on 11/13 and there is no SUBMIT_LOG.json; (b) '1158 rows' is the count of the file made from the 5070 Ti's saved float16 maps; a judge's GPU would give a float16 result, not necessarily that exact file.

**Suggested fix:** Add --fp32 to the command at README.md:89 and to the usage block in src/predict_final.py (line 18), next to --keep-table.

## 3. [low] `scripts/build_notebook_final.py:92` — FIXED (gt_index built on first use)

**Finding:** The notebook's 'How the files in weights/ were made' recipe omits the step that writes scratch/bank/gt_index.pkl, which refine_bank.py, keep_fit.py and keep_eval.py read and only analysis/tree_bank.py (the abandoned D1 experiment) creates.

**Scenario:** Fresh clone, run the listed command python analysis/refine_bank.py '{"tag": "oof", ...}' -> FileNotFoundError for scratch/bank/gt_index.pkl before anything is computed (*.pkl and scratch/ are git-ignored).

**Verifier:** Exported HEAD with git archive into my folder and ran the listed command python analysis/refine_bank.py '{"tag": "oof", ...}': FileNotFoundError for ...\scratch\bank\gt_index.pkl, exit code 1. refine_bank.py:86 reads the file before any map or image is opened, so the missing data in the export is not the cause. The export was removed afterwards.
Only analysis/tree_bank.py:81 creates the file (grep); refine_bank.py:31, keep_fit.py:32, keep_eval.py:31 and refine_eval.py:46 only read it. scratch/ is git-ignored (scratch/.gitignore '*') and so is *.pkl. The recipe text contains neither 'gt_index' nor 'tree_bank' (reqs_notebook.json), and the committed notebook's cells equal the builder's.

**Suggested fix:** Either add one line before step 2 of the recipe (and rebuild the notebook):
python -c "import sys; sys.path.insert(0, 'analysis'); from pathlib import Path; from tree_bank import gt_index; gt_index(Path('scratch/bank/gt_index.pkl'))"
or, in the four analysis scripts, replace pickle.loads(GT_INDEX.read_bytes()) by tree_bank.gt_index(GT_INDEX), which builds the index when it is missing. I ran that function into my own folder: 103 s, 707 images / 1154 records / 8199 filaments, identical to the existing scratch/bank/gt_index.pkl (gtindex.json).

## 4. [low] `src/refine.py:13` — FIXED

**Finding:** The module docstring and constants of refine.py describe the earlier design, not the final pipeline.

**Scenario:** A judge reads 'Which seeds are kept is still decided by the stage-1 rule' (line 13), 'the stage-1 mask is changed only where the refiner is sure (refined_piece with a margin)' (lines 12-13) and 'only kept seeds are submitted' (line 36), while predict_final keeps seeds by the learned rule (including seeds the fixed rule drops) and pastes with margin 0 after a 50/50 blend. Line 9 points to scratch/ws/refine/RESULTS.md, which is git-ignored.

**Verifier:** Line 13 says 'Which seeds are kept is still decided by the stage-1 rule'. In predict_final.image_instances the seeds are chosen by score > tau: 137 of the 4360 seeds kept out of fold (28 of 1158 on the test table) are seeds the fixed rule drops, and 601 (148) fixed-rule seeds are dropped.
Lines 12-13 say the stage-1 mask is changed only where the refiner is sure (refined_piece with a margin). predict_final.py:90 and keep_eval.py:85 call paste_image without margin (default 0.0) after the 50/50 blend.
The comment at line 36 ('only kept seeds are submitted') refers to the fixed rule's kept column.
Line 9 points to scratch/ws/refine/RESULTS.md, which exists locally but is git-ignored.
The same docstring (lines 11-12) says joining pieces is one of two ways the refiner helps; in the final configuration it fires once in 4360 seeds (see finding 5).

**Suggested fix:** Reword lines 9-13 to the final use: blended 50/50 with the dense probability in the crop, thresholded at 0.5, pieces touching the seed pasted with margin 0; seeds chosen by the learned keep rule (src/keep.py); results in EXPERIMENTS.md (S2, K1) instead of the scratch file. At line 36 say that KEEP_MIN_AREA / KEEP_MIN_MEAN are the fixed rule kept for comparison (the bank's kept column).

## 5. [low] `src/predict_final.py:12` — FIXED (wording)

**Finding:** The join rule (absorb 0.3), presented as part of the shape step here, in docs/FINAL_PIPELINE_zh-CN.md:62 and in the notebook intro, almost never fires in the final configuration.

**Scenario:** Asked how often candidates are joined, the answer is 1 of 4360 kept seeds out of fold and 0 of 1159 on the test set; the +0.002 of the refiner step comes from the outline alone, not from 'merging pieces'.

**Verifier:** My own paste of all 707 train images (cross-fitted keep scores, tau 0.25, fold-k refiner on fold-k seeds, FINAL settings), scored on real label maps with postproc.score_file:
- learned rule: 4360 kept -> 4359 instances with absorb 0.3 and 4360 without; 1 join (20110410141654Bh.jpeg), 0 pasted unchanged, 0 dropped, 420 instances made of more than one piece. PQ 0.4563 (TP 5116 / FP 1995 / FN 3083) with the rule, 0.4562 without; score parts identical in 706 of 707 images.
- fixed rule: 4824 kept -> 4821 instances (3 joins); 0.4490 with, 0.4486 without.
- outline alone: refiner shapes without any join are +0.0041 over the stage-1 masks for the learned rule (5/5 folds, bootstrap +0.0026..+0.0055), so the refiner step's gain does not come from merging.
Test set: scratch/sub_1113/cpu_run.log has rows 1159 = kept 1159, so no join and no drop.
The reference numbers reproduce exactly (0.4439 5268/2601/2931; 0.4490 5300/2560/2899; 0.4521; 0.4563). File: paste_report.json.

**Suggested fix:** No code change (dropping the rule would move one image, +-0.0001). Add a clause in the three places, e.g. 'rare: 1 of 4360 kept candidates out of fold, none in the test set; kept as a safeguard', so nobody presents merging as a source of the gain.

## 6. [low] `src/keep.py:62` — FIXED (stable sort)

**Finding:** rank_mean / rank_area use np.argsort with the default unstable sort, whose order of tied values depends on the CPU's SIMD path.

**Scenario:** Two seeds of an image with the same area get their two ranks in an order that can differ between an AVX2 x86 machine and e.g. Apple Silicon; 22 training seeds (11 images) and 6 test seeds are in such ties.

**Verifier:** Same machine, numpy 2.5.2, only the dispatch changed: with AVX2 dispatch argsort(-[0,0,1,0,1]) = [2,4,1,0,3]; with NPY_DISABLE_CPU_FEATURES=X86_V3 (scalar path) it is [2,4,0,1,3], the stable order; 172 of 195 small test arrays with ties sort differently between the two paths. Exhaustive search: the smallest unstable case here is n = 4 (areas [0,0,1,1] -> [3,2,1,0], stable [2,3,0,1]). Apple Silicon itself was not tested; the scalar path stands in for it.
Real data: 22 training seeds in 11 images and 6 test seeds in 3 images share their area with another seed of the same image; no ties in the mean.
Recomputing the ranks under the scalar path changes rank_area in 6 rows of the training table and 2 rows of the test table: 0 decisions flip and scores do not move. Reversing every tie (the reviewer's variant): score change up to 0.0103, 0 flips. Files: argsort_*.json, keep_table_native.json, keep_table_scalar.json, skew.json.

**Suggested fix:** kind='stable' in the two inner np.argsort calls at keep.py:62-63. Without rebuilding the table this changes 0 decisions (2 test rows get another rank, scores unchanged) and every machine then computes the same ranks; the shipped table keeps 6 rows with the other tie order, which is harmless.
Rebuilding the table as the reviewer proposes is not output-neutral: 6 rows change, the refit moves 36 of 6788 out-of-fold decisions and 10 of 1727 test decisions (PQ 0.4523 vs 0.4521, bootstrap -0.0003..+0.0008, noise). Do the rebuild only together with the one for finding 1 (both together: out of fold 0.4519 vs 0.4521, bootstrap -0.0011..+0.0007; 20 test decisions change).

## 7. [low] `results/evidence/final_sprint_1112/SUMMARY.json:7` — FIXED (wording in SUMMARY.json and notebook)

**Finding:** The 'nested' protocol line (printed by the notebook's validation cell) covers the keep threshold only: the paste settings (blend 0.5, t_r 0.5, absorb 0.3) were picked on all five folds.

**Scenario:** The notebook prints 0.4490 for the refiner row under that protocol; EXPERIMENTS.md's own nested figure for the row is 0.4483, so that row and the final 0.4560 carry up to about 0.0007 of selection that the summary does not mention.

**Verifier:** SUMMARY.json line 7 and the notebook's validation markdown (scripts/build_notebook_final.py:147-150) name only the threshold as nested. The validation cell prints 'PQ 0.4490 previous + refiner shapes (blend 0.5, t_r 0.5, absorb 0.3)'; EXPERIMENTS.md (S2 table) gives 0.4483 nested for that row and calls 0.4490 the overall best setting.
Direct measurement for the final row (my run, real label maps, learned rule at tau 0.25, 24 paste settings = blend 0.3/0.5/0.7 x t_r 0.45/0.50/0.55/0.60 x absorb none/0.3): the shipped setting is the best of the 24 on all folds (0.4563). Choosing the setting per fold on the other four folds gives 0.4554, i.e. -0.0009 (paired bootstrap 5-95% -0.0016..-0.0003); per fold 0.4474/0.4623/0.4555/0.4612/0.4501 vs 0.4479/0.4623/0.4573/0.4612/0.4527; 1/2/3-record images 0.4566/0.4672/0.4464 vs 0.4570/0.4681/0.4479. So the final 0.4560 carries about 0.001 of selection, the same order as the reviewer's 'up to about 0.0007'.
The keep rule's gain is not affected: nested-paste learned rule vs the fixed rule with the shipped settings is +0.0064, 5/5 folds, bootstrap +0.0039..+0.0089.
Limits: my grid is smaller than the team's 92-setting grid (no cover / margin variants, no blend 1.0) and tau was held at 0.25. File: paste_report.json.

**Suggested fix:** One sentence in the SUMMARY.json protocol and in VAL_MD of scripts/build_notebook_final.py (then rebuild the notebook): 'The paste settings (blend 0.5, t_r 0.5, absorb 0.3) were chosen on all five folds; with that choice nested as well, the refiner row is 0.4483 and the final row about 0.455.'

## 8. [low] `src/keep.py:29` — FIXED (early_stopping=False)

**Finding:** keep.GBM leaves scikit-learn's early_stopping at 'auto', which changes the fit as soon as a training table has more than 10,000 rows.

**Scenario:** Not triggered by the shipped table (6788 rows; 5367 per cross-fit). A table with extra rows (the recorded 'single-model seeds as extra rows' experiment, 16,509 rows per fit) is silently fitted on 90% of the rows with a validation split.

**Verifier:** Synthetic, scikit-learn 1.9.1 through keep.fit: 10,000 rows -> do_early_stopping_ False, 10,000 rows in the first tree's root, 300 trees, identical to early_stopping=False. 10,001 rows -> True, 9,000 rows in the root, stopped after 101 trees, identical to early_stopping=True.
Shipped table: 6788 rows, not triggered.
The recorded 'single-model seeds as extra rows' experiment, rebuilt from the stored tables with my own code (cross-fitted by fold): 16,091 to 16,509 rows per fit; with keep.GBM the roots hold 14,481 to 14,858 rows (90%), 300 trees. PQ at tau 0.25 is 0.4515 (best over tau 0.4524, the figure in EXPERIMENTS.md); with early_stopping=False it is 0.4494 (best 0.4518).
Against the shipped rule (0.4521) at tau 0.25: -0.0006 with the silent 90% fit (bootstrap -0.0020..+0.0008) and -0.0027 with all rows (bootstrap -0.0041..-0.0012). The recorded conclusion 'no gain' stands either way. Files: small_checks.json, earlystop.json.

**Suggested fix:** Add early_stopping=False to keep.GBM. For the shipped table this changes nothing: predictions on the training table and on the test table are bit-identical (earlystop_fix.json).

## 9. [low] `src/fastsweep.py:63` — FIXED

**Finding:** With bridge > 0 most labels are empty, and ndi.mean over them raises a RuntimeWarning for almost every image; the CLI does not silence it (the notebook does).

**Scenario:** python src/predict_final.py on the 180 test images prints 'RuntimeWarning: invalid value encountered in divide' 153 times, which reads like a failure to someone running the README command.

**Verifier:** Both of my CLI runs on the 180 test images print 'RuntimeWarning: invalid value encountered in divide' 153 times (v_light_ab.log, v_light_ba.log), about 33 KB of text before the result line; the CSV of the documented order is byte-identical to the stored candidate.
Synthetic map with one bright and one faint piece: components(prob, 0.4, 8, bridge=0.1) gives 2 groups, 1 empty, 1 warning; bridge=0 gives no warning.
Only the notebook filters the message (scripts/build_notebook_final.py:70); nothing under src/ does.

**Suggested fix:** In fastsweep.components put the two ndimage statistics (lines 63-64) inside 'with np.errstate(invalid="ignore"):'. Checked on the synthetic case: 0 warnings, identical labels and features (fixes.json). The notebook's filter line can then go. Copying that filter into predict_final.main() also works.

## 10. [low] `src/predict_final.py:65` — FIXED (any number >= 2 members, table records the count)

**Finding:** image_instances accepts any number of member maps, but the descriptors use exactly the first two.

**Scenario:** With three maps (the pre-registered M3) the seeds come from the three-model mean while mem_* and img_p_frac come from members 0 and 1 only, with no error; analysis/keep_bank.py raises for this case, predict_final does not.

**Verifier:** Synthetic run: image_instances([a, b, c], ...) raises nothing; the seeds come from the three-map mean, while the descriptor matrix equals keep.image_descriptors(seeds, [a, b]) bit for bit. The third map's mean inside the three seeds (0.98 / 0.12 / 0.98) appears in none of the mem_* columns, and img_p_frac is computed from the first two maps (small_checks.json).
keep.py:59 and :84-85 index members[0] and members[1] only. analysis/keep_bank.py:42-43 raises for any count other than two. --probs and --dense are nargs=2, so only a direct caller can hit this.
The pre-registered M3 run would be such a caller, but its script (scratch/ws/main/m3_chain.sh) already builds its tables with a scratch variant that rewrites these descriptors (keep_bank3.py).

**Suggested fix:** At the top of keep.image_descriptors: if len(members) != 2: raise ValueError('the descriptors compare exactly two member models'). This covers predict_final, the notebook and keep_bank at once.

## 11. [low] `src/make_submission.py:43` — NOT changed (CRLF on Windows; compare CSVs by masks, not by hash)

**Finding:** write_csv opens the file in text mode, so the CSV has CRLF line ends on Windows and LF elsewhere.

**Scenario:** The same masks give a different file hash on Windows and on macOS/Linux; it only matters for byte-level comparisons, since Kaggle has accepted the CRLF files.

**Verifier:** write_csv called on this machine writes b'filament_id,segmentation_rle\r\nimg_1,abc\r\n...' (crlf_probe.csv); on macOS / Linux the same code writes LF (Python's text-mode newline translation).
The three float32 candidate files each have 1160 lines, all CRLF. They are otherwise well-formed: ids consecutive from 1, no duplicates, no overlapping pixels, no empty masks, smallest mask 199 px (refiner files) and 142 px (light file) (csv_check.json).
Not verified by me: that Kaggle accepted CRLF files (no network use). The earlier files in scratch/sub_1112, which EXPERIMENTS.md records as submitted, are CRLF as well.

**Suggested fix:** Pin the line end in write_csv: open(out, 'w', newline='\r\n') keeps the files already made byte-reproducible on every platform; newline='\n' is the usual choice but changes the bytes of future Windows runs. Or leave it and compare files by masks (analysis/compare_csv.py), never by hash.

## 12. [info] `src/keep.py:96` — FIXED (renamed p_crop_max)

**Finding:** The descriptor named p_max is the maximum over the whole crop, not over the seed.

**Scenario:** For 380 of 6788 training seeds (5.6%) it is a neighbour's peak, up to 0.49 above the seed's own. It is computed the same way in training and prediction, so there is no skew; only the name misleads a reader.

**Verifier:** Line 96 writes p.max() while its neighbours use p[seed].
Scan of the out-of-fold seed bank against the table: the p_max column equals the crop maximum for all 6788 seeds and the seed's own maximum for 6408. For 380 seeds (5.6%) the crop maximum is higher, by up to 0.494 (median 0.094); in 378 of them the peak lies on another seed, in 2 on a small unlabelled component (pmax.json).
Training and prediction call the same function. On 16 random train images (122 seeds) the descriptors recomputed through the prediction path equal the table bit for bit (skew.json).

**Suggested fix:** A comment at NAMES or at line 96: 'p_max: maximum over the whole crop (can be a neighbour's peak)'. Renaming or changing it to p[seed].max() needs a rebuilt table, because load_model checks the stored names.

## 13. [info] `notebooks/end_to_end_final.ipynb:1` — PARTLY: requirements fixed; executed notebook on all 180 images still TODO

**Finding:** The committed final notebook has never been executed (0 of 6 code cells have outputs), and requirements.txt lacks ipykernel and nbformat.

**Scenario:** Judges opening the notebook on GitHub see no masks or scores; a clean environment built from requirements.txt cannot start a kernel. The figure cell raises ValueError (plt.subplots(0, 2)) if no image has a filament, which only happens with a very small FILAMENT_LIMIT.

**Verifier:** Committed notebook: 6 code cells, 0 with outputs, all execution counts null; cell sources equal scripts/build_notebook_final.py.
Correction 1: 'never been executed' is wrong as history. scratch/ws/main/nb_final_limit3.ipynb and nb_final_limit8.ipynb are executed copies (nbclient via scratch/ws/main/run_nb.py, CUDA, execution counts 1-6, no error outputs; sources equal the committed ones in 12 of 12 and 11 of 12 cells). What is missing is a committed copy with outputs, and a run on all 180 images.
Requirements: the dependency closure of the 12 pinned packages (49 packages, read from the installed metadata) contains neither ipykernel nor nbformat nor any Jupyter package; both are installed in this venv (ipykernel 7.4.0, nbformat 5.11.1). The first line of requirements.txt says 'Everything needed to train, predict and run the notebook', which is therefore not true.
Correction 2: the figure cell does raise 'ValueError: Number of rows must be a positive integer, not 0' when no image has a filament (run with one image the pipeline leaves empty). But the first test image (20110120105534Ch) has 2 filaments and every FILAMENT_LIMIT includes it, so on the test folder it never happens; with FILAMENT_LIMIT = 1 the cell runs. The first image without filaments is number 141 of 180 (viscell.json, reqs_notebook.json).

**Suggested fix:** Execute the final notebook once on all 180 images with the frozen files and commit that copy (the owner's own scratch/ws/main/run_nb.py does this). Add ipykernel, and nbformat / nbclient if the build and run scripts are meant to work, with exact versions to requirements.txt. Optional: 'if show:' around the figure code.

