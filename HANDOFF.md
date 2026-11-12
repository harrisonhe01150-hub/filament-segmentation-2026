# 交接：最后冲刺（2026-11-12，Harrison → Zichang、xianyang）

> 写给队友和队友的 Claude。先读这一页，再按需要读后面列出的文件。
> 写于 2026-11-12 14:40（美国中部时间 CST，= 20:40 UTC）。分支 `harrison/final-sprint`，PR #10。
> 时间一律写 UTC，括号里是美国中部时间（CST = UTC−6，11 月 1 日后已是冬令时）。

## 0. 一分钟版

- **本地五折折外分数从 0.4409 提到 0.4566**（官方自评口径，严格嵌套）。主要来自一个新的「学习型留/删规则」（K1），其次是逐条细化轮廓。两路独立复核都确认收益是真的。
- **线上还没验证。** 两份候选文件会在 **11/13 00:02 UTC（11/12 18:02 CST）** 从 Harrison 的电脑自动提交（先轻量版、再完整版，占当天 5 次里的 2 次）。目前线上最好是 0.39、第 331 名。
- **代码都在 PR #10 里**（包含 PR #9 的全部提交），还没合进 `main`。
- **权重文件（7 个，约 370 MB）还只在 Harrison 的电脑上**，没进仓库。最终 notebook 需要它们，见第 5 节。
- **剩下要做的**（第 3 节）：看今晚的线上结果 → 定最终流程 → 权重进仓库、跑一遍最终 notebook → 合并 PR → 报告 → 选两份最终提交 → 仓库公开 → 填表单。

## 1. 截止时间

| 事项 | 截止 | 负责（按 TEAM_ROLES.md） |
|---|---|---|
| Kaggle 提交 | 2026-11-15 06:00 UTC（11/15 00:00 CST） | Zichang |
| 在 Kaggle 上手动选 2 份最终提交 | 同上，**必须手动选** | Zichang |
| 仓库公开 | 计划 11/13（见第 6 节的风险提示） | Harrison 原计划 |
| Google 表单（报告 PDF + 仓库链接） | README 写的是 11/15 23:59 美国中部时间；计划 11/14 交 | Zichang |
| 每天提交额度 | 全队共 5 次，00:00 UTC（18:00 CST）重置 | — |

## 2. 现在的状态

### 2.1 本地分数（五折折外，707 张训练图，官方自评口径）

| 流程 | PQ | 说明 |
|---|---|---|
| v2 单模型，固定规则（面积 ≥ 300、平均概率 ≥ 0.85） | 0.4409 | 原基线，线上 0.39 |
| 两个主模型平均 + 弱连接分组 0.1 + 写出时收紧到 0.48 | 0.4456 | 线上 0.39，**第 331 名（目前最好）** |
| 上一行 + 逐条细化（固定规则） | 0.4490 | 细化的贴回设置也嵌套选时是 0.4483 |
| **学习型留/删 + 收紧 0.48（轻量版，不需要细化网络）** | **0.4537** | 阈值 0.25 时 0.4539 |
| **学习型留/删 + 逐条细化（完整版，最终流程）** | **0.4566** | 阈值 0.25 时 0.4563；细化设置也嵌套时约 0.455 |

- 「严格嵌套」：第 k 折用的留/删模型和阈值都没见过第 k 折（阈值由内层交叉拟合选）。
- 以往所有诚实方案线上 ≈ 本地 × 0.875。按这个比例，完整版约 0.399，**可能显示 0.40，也可能仍是 0.39**。公开榜只有 90 张图，单次读数波动约 ±0.004。
- 同分队伍按没显示出来的小数排，所以只要真实分数比 0.388 左右高，名次就会动。

### 2.2 线上提交记录

| 时间（UTC） | 文件 | 线上 | 名次 |
|---|---|---|---|
| 10/23 | v1 两折 / v2 两折 / v2 五折 | 0.37 / 0.39 / 0.38 | |
| 10/27 | 五折 min_mean 0.83；全量双模型 | 0.38；0.39 | |
| 11/07 | 队友：v2 两折 CPU 复现（commit 13a18c432efb，**这个 commit 不在本仓库**） | 0.39 | |
| 11/12 08:27 | 探测 min_mean 0.90（不是候选） | 0.36 | |
| 11/12 08:42 / 08:48 / 08:57 | 收紧 0.48 / 0.60 / 0.70 | 0.39 / 0.39 / 0.38 | 364→353→350 |
| 11/12 09:08 | `full2_stack_b01_d048.csv`（两模型 + 后处理组合） | 0.39 | **331** |
| **11/13 00:02（已排好）** | `K1_light_fp32.csv` → `K1_refine_fp32.csv` | 待出 | 待出 |

- 11/12 14:02 UTC 的榜：1500 队，0.39 档 146 队里我们排第 97，显示 ≥ 0.40 的有 234 队。
- 自动提交脚本在 Harrison 电脑上运行（`scratch/ws/main/submit_1113.py`），每交一份会下载榜单，把显示分、名次、在同分档里的位置写进 `scratch/sub_1113/SUBMIT_LOG.json`。**电脑关机或断网就不会提交**；11/13 00:40 UTC 后在 Kaggle 的 My Submissions 里看有没有这两份。取消办法：在那台电脑上建空文件 `scratch/sub_1113/STOP`。

### 2.3 代码和分支

| 分支 / PR | 内容 | 状态 |
|---|---|---|
| `main`（a4f153d） | v2 基线、PR #7 之前的一切 | |
| PR #9 `harrison/s1-self-training` | S1 自训练、R1 2048（都没通过）、全量模型记录 | 开着；被 #10 包含 |
| **PR #10 `harrison/final-sprint`** | 本次全部工作，最新 commit 见 PR | 开着，等合并 |
| PR #6、#8（smyll / codex） | A1 中心线（没通过）的记录 | 开着，没动 |

PR #10 里的关键文件：

| 文件 | 是什么 |
|---|---|
| `src/predict_final.py` | **最终流程一条命令**：主模型概率图 → 候选 → 留/删 → 细化形状 → CSV。`--no-refiner` 是轻量版，`--fp32` 是提交文件的生成方式 |
| `src/keep.py` | 学习型留/删规则（50 个描述量 + 梯度提升树，阈值 `TAU = 0.25`） |
| `src/refine.py` | 候选库、逐条细化网络、贴回 |
| `src/fastsweep.py`、`src/make_submission.py`、`src/predict.py` | 弱连接分组 `bridge`、写出阈值 `draw`（`predict.POST_STACK`） |
| `analysis/keep_bank.py` / `keep_fit.py` / `keep_eval.py` | 留/删规则的描述量表、交叉拟合评估（含严格嵌套）、真实标签图评估 |
| `analysis/refine_bank.py` / `refine_train.py` / `refine_eval.py` | 细化网络的候选库、训练、评估 |
| `notebooks/end_to_end_final.ipynb`（由 `scripts/build_notebook_final.py` 生成） | 最终流程的 notebook。**仓库里这份没有输出**，只在 3 张和 8 张图上试跑过 |
| `weights/README.md`、`weights/keep_table.npz` | 最终流程需要的文件清单和 SHA256；留/删模型的训练表（预测时现拟合，几秒） |
| `docs/FINAL_PIPELINE_zh-CN.md` | **大白话讲最终流程**（答辩用：主办方要求每人讲得清自己的代码） |
| `EXPERIMENTS.md` | 全部实验的预登记和结果，含今天的 K1、复核、M3、三路探索 |
| `results/evidence/final_sprint_1112/SUMMARY.json` | 折外分数阶梯（notebook 最后一格会打印） |
| `report/reproducibility.md` | 报告「可复现性」草稿，末尾有最终流程的补充 |
| `tests/test_keep_rule.py` 等 4 个 | CPU 自检，全部通过 |
| `docs/handoff_1112/` | 今天的复核、代码审查、探索报告原文，和 K2 的代码快照（见第 4 节） |

## 3. 剩下要做的（建议顺序）

1. **看今晚的线上结果**（11/13 00:40 UTC 后）。
   - 在 My Submissions 里读两份的显示分；下载榜单看名次（Kaggle CLI：`kaggle competitions leaderboard -c filament-segmentation-2026 --download`），和第 331 名比。
   - 写进 `EXPERIMENTS.md` 最后的提交记录和 `results/evidence/LIVE_SUBMISSIONS.json`。
2. **定最终流程**：完整版（K1 + 细化）还是轻量版（K1 + 收紧）。
   - 本地完整版高 0.003；代价是多 5 个细化网络（每个 41 MB）和多一段代码。
   - 建议：线上两份差不多时用完整版；轻量版明显更好再换。不要因为 ±0.004 以内的线上差别推翻本地结论。
3. **权重进仓库**（第 5 节）。放到 `weights/` 下，文件名和 SHA256 按 `weights/README.md`。用轻量版就只要两个主模型 + `keep_table.npz`。
4. **跑一遍最终 notebook 并提交带输出的版本**。
   - 在有权重的机器上跑 `notebooks/end_to_end_final.ipynb`（180 张图，`FP32 = True`，RTX 5070 Ti 上约 7 分钟，CPU 上约 70 分钟），保存带输出的 notebook 提交。
   - 用 `python analysis/compare_csv.py outputs/submission_final.csv <提交过的那份>` 逐条比，每条掩码 IoU ≥ 0.999 就算复现。
   - 用轻量版的话，把 notebook 的 `USE_REFINER` 改成 `False` 再跑。
5. **合并 PR #10 进 `main`**（会顺带合进 #9）。合之前跑一遍 `tests/` 下 4 个脚本和 `python scripts/update_manifest.py`（notebook 第一格会核对文件哈希）。
6. **报告**（4 页，官方 Overleaf 模板；分工见 `TEAM_ROLES.md` 第 7 节）。
   - 方法：照 `docs/FINAL_PIPELINE_zh-CN.md` 写（三步：主模型 + 候选 → 学习型留/删 → 逐条细化）。
   - 结果：折外分数阶梯用 2.1 节的表或 `SUMMARY.json`；按 1/2/3 人标注拆开的分数在 `EXPERIMENTS.md` K1 一节。
   - 评分里有 Dice / IoU 分布、拆分（1:n）和合并（n:1）次数。以前的图（`report/figures/`）是 v2 的；有时间就用最终流程重画（`analysis/diagnose.py`、`analysis/report_figs.py`）。
   - 可复现性：`report/reproducibility.md`（CPU 和 GPU 用 float32 结果一致：1159 条里 1158 条完全相同。这是在下午修正留/删描述量之前的版本上测的；修正不涉及跨设备的那部分，但修正后的文件没有再在 CPU 上跑过）。
   - **主动写清的局限**：K1 是事后补记的探索性实验（复核过）；对外收益按 +0.007 说，不按最初算的 +0.008；2018–2022 年的图上没有收益；细化的贴回设置是在全部五折上选的。
7. **在 Kaggle 上选 2 份最终提交**（11/15 06:00 UTC 前，手动）。
   - 私榜是另外 50% 的测试图，要按本地折外分选，不要按公开榜的小差别选。
   - 建议：一份选最终流程（完整版或轻量版里本地更高、线上没出问题的那份），一份选已验证的 `full2_stack_b01_d048.csv`（0.39，第 331 名）做保险。如果 K1 两份线上都明显好于它，也可以两份都选 K1。
8. **仓库公开**，然后**填 Google 表单**（报告 PDF + 仓库链接）。

## 4. 可选：K2（再 +0.002，已复核，没并入）

- **是什么**：四个预测取平均：K1 本身；同样的提升树用在另一套 66 个描述量上；5 个小神经网络用在 50 个描述量上；5 个用在 66 个描述量上。
- **效果**（严格嵌套）：原始掩码 0.4510 → 0.4528；另外 10 组分折平均 +0.0028（10/10 为正）。完整流程真实标签图上 0.4558 → 0.4571，复核方用自己的网络种子算是 +0.0020（+0.0011 … +0.0029）。
- **要注意**：
  - 在现在的阈值 0.25、原分折上只有 +0.0006；K2 的最优阈值在 0.22–0.23，要重新选。
  - 每套描述量用 10 个网络比 5 个稳。
  - **那 66 个描述量里 `ag_diff`、`ag_fk`、`ag_fc` 也取决于两个主模型的先后顺序**，和今天在 K1 里修掉的问题同一类，移植时必须改成对称写法并重测。
  - 神经网络在不同机器上现拟合会有细微差别，要么提交网络权重，要么提交分数。
- **代码**：`docs/handoff_1112/k2_snapshot/`（`candidate.py` 是整合好的类，`feats.py` 是 66 个描述量）。这是快照，**原样跑不起来**：它们引用 Harrison 电脑上的 `scratch/` 路径和缓存。
- **建议**：只在第 3 节都做完、还有一天以上时再做；做的话照 `EXPERIMENTS.md` 的规矩先预登记。

## 5. 只在 Harrison 电脑上的东西（不在 git 里）

| 东西 | 路径（仓库根目录下） | 大小 | 谁需要 |
|---|---|---|---|
| 最终流程的 7 个权重 | `weights/model_full_s2026.pt`、`model_full_s2027.pt`、`refiner_f0..4.pt` | 82 MB × 2、41 MB × 5 | **最终 notebook、复现** |
| 两份候选提交 CSV（各 1163 条） | `scratch/sub_1113/final/K1_refine_fp32.csv`、`K1_light_fp32.csv` | 各 0.3 MB | 自动提交失败时手动提交；有权重也可以用第 7 节的命令重新生成 |
| 已提交的组合版 CSV | `scratch/sub_1112/full2_stack_b01_d048.csv` | 0.3 MB | 不需要（Kaggle 上能直接选） |
| 测试集主模型概率图（float32） | `scratch/sub_1113/dense_fp32/dense_0.npz`、`dense_1.npz` | 小 | `predict_final.py --probs` 快速重算 |
| 全部折外概率图 | `runs/v2*`、`runs/v2b*`、`runs/v2_5070ti`、`runs/v2b_5070ti`、`runs/v2_s2028_5070ti` 等 | 2.4 GB | 重算折外分数、重建留/删训练表 |
| 候选库（含小图） | `scratch/ws/main/rbank/kc_bridge`（训练）、`test_full2_bridge`（测试） | 2 GB | 同上 |
| 折外细化预测 | `scratch/ws/main/kcb/pred/pred_f*.npz` | 小 | `keep_eval.py --shape refine` |
| 留/删描述量表和折外分数 | `scratch/keep/oof/`、`scratch/keep/test/` | 小 | `keep_fit.py`、`keep_eval.py` |
| 第三个全量模型（M3，没通过，不用） | `runs/v2_full_s2028_5070ti` | 82 MB | 不需要 |

**只做第 3 节的事，只需要第一行的权重。** 其余是重算本地分数用的。`scratch/bank/gt_index.pkl`（训练标注的像素索引）缺了会自动重建（约 2 分钟）。

## 6. 规矩和坑（队友的 Claude 务必遵守）

- **绝不使用测试图的标注**，包括 MAGFiLO 公开数据里的、别队的 notebook 输出和数据集。只用比赛的训练标注。
- 推理只用测试图本身。**任何描述量和决定都不能用文件名里的信息**（日期、观测站）；文件名只能用来分折和做分析。
- 每个实验先在 `EXPERIMENTS.md` 写好假设和通过标准再跑；通过标准是五折嵌套 ≥ +0.005、至少 4 折变好（今天的 K1 是事后补记，已注明并复核）。
- 不要把 Kaggle token、`.env` 写进任何文件或提交。
- **仓库公开前想清楚时机**：`PLAN.md` 写 11/13 公开，`README.md` 写「at competition close」。权重 + 代码一公开，别人在截止前就能直接复现我们的分数。先确认 Overview 的原话；如果允许，建议 11/15 06:00 UTC 提交截止后再公开、再填表。公开前查一遍提交历史里没有密钥。
- `weights/` 下的大文件现在是未跟踪状态，**不要 `git add -A`**；按文件名加。
- Windows 上跑 CPU 任务：设 `OPENBLAS_NUM_THREADS=1`（否则每个 numpy/scipy 进程先占约 1.5 GB 提交内存，多开几个会把同机的 GPU 训练挤死）；一次别开超过 6 个进程。
- 生成提交文件用 `--fp32`。CUDA 默认 float16，会让几个卡在阈值边上的候选进出。
- Windows 写出的 CSV 是 CRLF 换行，Kaggle 接受；比较两份 CSV 用 `analysis/compare_csv.py` 按掩码比，不要比文件哈希。

## 7. 常用命令

```bash
# 最终流程（完整版），从权重算到 CSV
python src/predict_final.py --dense weights/model_full_s2026.pt weights/model_full_s2027.pt --config configs/v2_full.json \
    --refiners weights/refiner_f0.pt weights/refiner_f1.pt weights/refiner_f2.pt weights/refiner_f3.pt weights/refiner_f4.pt \
    --keep-table weights/keep_table.npz --fp32 --out outputs/submission_final.csv

# 轻量版
python src/predict_final.py --dense weights/model_full_s2026.pt weights/model_full_s2027.pt --config configs/v2_full.json \
    --no-refiner --keep-table weights/keep_table.npz --fp32 --out outputs/submission_light.csv

# 自检（CPU）
python tests/test_keep_rule.py
python tests/test_cpu_portability.py
python tests/test_runtime_guards.py
python tests/test_small_filament_sampling.py

# 改了代码之后
python scripts/update_manifest.py
python scripts/build_notebook_final.py

# 重算本地分数（需要第 5 节的折外文件，只在 Harrison 电脑上）
python analysis/keep_fit.py --tables scratch/keep/oof --bank scratch/ws/main/rbank/kc_bridge
python analysis/keep_eval.py --scores scratch/keep/oof/scores.npz --bank scratch/ws/main/rbank/kc_bridge --shape refine --pred scratch/ws/main/kcb/pred
```

## 8. 今天试过、没用的（别再试）

细化网络取平均、多尺度推理、第三套主模型（M3：完整流程 0.4545 < 0.4566）、合并碎片的判断模型、降低阈值找新候选、用别的模型补候选、逐条选轮廓松紧（上限 +0.018，学不出来）、专门修 2018–2022 年的图、留/删模型的第二遍打分、细化输出或小图分类网络当描述量。数字和原因都在 `EXPERIMENTS.md`；报告原文在 `docs/handoff_1112/reports/`。

## 9. 参考资料

| 想知道 | 看哪里 |
|---|---|
| 最终流程每一步为什么这么做 | `docs/FINAL_PIPELINE_zh-CN.md` |
| 每个实验的假设、数字、结论 | `EXPERIMENTS.md`（K1 在后半部分） |
| K1 的两路独立复核 | `docs/handoff_1112/reports/verify_K1_audit.md`、`verify_K1_reimplementation.md` |
| 代码审查的 13 条和处理情况 | `docs/handoff_1112/reports/code_review_1112.md` |
| 三路探索 | `docs/handoff_1112/reports/explore_*.md`、`verify_keep_model_K2.md` |
| 权重清单 | `weights/README.md` |
| 团队分工、报告分工 | `TEAM_ROLES.md`、`PLAN.md` |
