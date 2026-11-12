# 交接：最后冲刺（2026-11-12，Harrison → Zichang、xianyang）

> 写给队友和队友的 Claude。先读这一页，再按需要读后面列出的文件。
> 写于 2026-11-12 14:40（美国中部时间 CST，= 20:40 UTC）。分支 `harrison/final-sprint`，PR #10。
> 时间一律写 UTC，括号里是美国中部时间（CST = UTC−6，11 月 1 日后已是冬令时）。
> 11/12 22:55 UTC 补：今晚两份测试集 CSV 的副本放进了 `results/test_submissions_1113/`；最终流程在训练图上的折外预测放进了 `results/oof_final/`（画报告图用，不需要 2 GB 候选库）。

## 0. 一分钟版

- **本地五折折外分数从 0.4409 提到 0.4566**（官方自评口径，严格嵌套）。主要来自一个新的「学习型留/删规则」（K1），其次是逐条细化轮廓。两路独立复核都确认收益是真的。
- **线上还没验证。** 两份候选文件会在 **11/13 00:02 UTC（11/12 18:02 CST）** 从 Harrison 的电脑自动提交（先轻量版、再完整版，占当天 5 次里的 2 次）。目前线上最好是 0.39、第 331 名。
- **代码都在 PR #10 里**（包含 PR #9 的全部提交），还没合进 `main`。
- **权重文件（7 个，约 370 MB）已提交进本分支的 `weights/`**，拉下分支就能跑最终 notebook。
- **文件在哪**：第 5 节（GitHub、仓库里、只在 Harrison 电脑上、Kaggle 上）。
- **剩下要做的**（第 3 节）：看今晚的线上结果 → 定最终流程 → 跑一遍最终 notebook → 合并 PR → 报告 → 选两份最终提交 → 仓库公开 → 填表单。

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
   - 写进 `EXPERIMENTS.md` 最后的提交记录，并补全 `results/evidence/LIVE_SUBMISSIONS.json`（目前只记到 10/23，11/12 的 5 份和今晚的 2 份都没记）。
2. **定最终流程**：完整版（K1 + 细化）还是轻量版（K1 + 收紧）。
   - 本地完整版高 0.003；代价是多 5 个细化网络（每个 41 MB）和多一段代码。
   - 建议：线上两份差不多时用完整版；轻量版明显更好再换。不要因为 ±0.004 以内的线上差别推翻本地结论。
3. **权重已在仓库里**（`weights/`，SHA256 见 `weights/README.md`；拉下后可用 `sha256sum weights/*` 核对）。最后定了轻量版的话，可以把 5 个 `refiner_f*.pt` 从 `main` 上删掉，也可以留着。
4. **跑一遍最终 notebook 并提交带输出的版本**。
   - 在有权重的机器上跑 `notebooks/end_to_end_final.ipynb`（180 张图，`FP32 = True`，RTX 5070 Ti 上约 7 分钟，CPU 上约 70 分钟），保存带输出的 notebook 提交。
   - 用 `python analysis/compare_csv.py outputs/submission_final.csv results/test_submissions_1113/K1_refine_fp32.csv`（轻量版比 `K1_light_fp32.csv`）逐条比，每条掩码 IoU ≥ 0.999 就算复现。
   - 用轻量版的话，把 notebook 的 `USE_REFINER` 改成 `False` 再跑。
5. **合并 PR #10 进 `main`**（会顺带合进 #9）。合之前跑一遍 `tests/` 下 4 个脚本和 `python scripts/update_manifest.py`（notebook 第一格会核对文件哈希）。
6. **报告**（4 页，官方 Overleaf 模板；分工见 `TEAM_ROLES.md` 第 7 节）。
   - 方法：照 `docs/FINAL_PIPELINE_zh-CN.md` 写（三步：主模型 + 候选 → 学习型留/删 → 逐条细化）。
   - 结果：折外分数阶梯用 2.1 节的表或 `SUMMARY.json`；按 1/2/3 人标注拆开的分数在 `EXPERIMENTS.md` K1 一节。
   - 评分里有 Dice / IoU 分布、拆分（1:n）和合并（n:1）次数。以前的图（`report/figures/`）是 v2 的；有时间就用最终流程重画。**数据已在仓库里**：`results/oof_final/`（最终流程在 707 张训练图上的折外预测，完整版和轻量版各一份，提交格式；每张图的 TP/FP/FN；每个候选的留/删分数、是否留下、和每份标注的最佳 IoU 及掩码）。说明见那里的 `README.md`。`analysis/report_figs.py` 现在读的是 v2 的候选表，要改成读这几个文件。
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

## 5. 所有文件在哪

### 5.1 GitHub

- 仓库：https://github.com/harrisonhe01150-hub/filament-segmentation-2026（**私有**）。
- 本次工作在分支 `harrison/final-sprint`，PR #10（含 PR #9）。`main` 还停在 a4f153d。
- 拉取：`git fetch origin && git checkout harrison/final-sprint`。权重在这个分支上，约 370 MB，第一次拉要几分钟。

### 5.2 仓库里（git 跟踪，拉下分支就有）

| 位置 | 内容 |
|---|---|
| `HANDOFF.md`、`CLAUDE.md` | 本交接；给 Claude 的入口 |
| `README.md`、`PLAN.md`、`TEAM_ROLES.md` | 比赛说明、计划、分工 |
| `EXPERIMENTS.md` | 全部实验的预登记、结果和提交记录 |
| `src/predict_final.py` | **最终流程**，一条命令从权重到 CSV |
| `src/keep.py` | 学习型留/删规则 |
| `src/refine.py` | 候选库、逐条细化网络、贴回 |
| `src/train_seg.py`、`src/predict.py`、`src/make_submission.py` | 主模型训练、v2 预测、CSV 写出（含后处理组合 `POST_STACK`） |
| `src/fastsweep.py`、`src/postproc.py`、`src/common.py` | 候选分组（含弱连接 `bridge`）、上采样和计分、官方口径 PQ |
| `src/treeselect.py`、`src/grouppost.py` | 没通过的候选树决策层、旧后处理实验（保留不用） |
| `analysis/keep_*.py` | 留/删规则：描述量表、交叉拟合评估、真实标签图评估 |
| `analysis/refine_*.py` | 细化网络：候选库、训练和应用、评估 |
| `analysis/tree_bank.py` | 候选树实验；也提供 `gt_index()`（训练标注的像素索引，其他脚本会自动调用） |
| `analysis/score_folds.py`、`compare_csv.py`、`diagnose.py`、`report_figs.py` 等 | 折外计分、两份 CSV 逐条比较、漏检诊断、报告图 |
| `scripts/build_notebook_final.py` → `notebooks/end_to_end_final.ipynb` | 最终流程的 notebook（**仓库里这份没有输出**） |
| `scripts/build_notebook.py` → `notebooks/end_to_end.ipynb` | v2 基线的 notebook |
| `notebooks/Solar_UNet_Team_Reproduction.ipynb` | 早期的团队复现 notebook |
| `scripts/update_manifest.py`、`SOURCE_MANIFEST.json` | 文件哈希清单（notebook 第一格核对） |
| `scripts/fetch_gong.py`、`make_pseudo.py`、`fetch_neighbors.py`、`build_lt_frames.py` | S1 自训练和 LT 相邻帧实验的数据脚本（都没通过） |
| `configs/` | 训练配置：`v2.json`（第 0、1 折）、`v2b.json`（第 2–4 折）、`v2_full.json`、`v2_full_s2027.json`（最终用的两个全量模型）、其余是实验 |
| `weights/` | **最终流程的全部文件**：`model_full_s2026.pt`、`model_full_s2027.pt`（主模型，各 82 MB）、`refiner_f0..4.pt`（细化网络，各 41 MB）、`keep_table.npz`（留/删训练表，1 MB）；`weights/README.md` 有来源和 SHA256 |
| `docs/FINAL_PIPELINE_zh-CN.md` | 最终流程的大白话说明（答辩用） |
| `docs/handoff_1112/reports/` | K1 两路复核、代码审查 13 条、三路探索、K2 复核的报告原文 |
| `docs/handoff_1112/k2_snapshot/` | K2 代码快照（原样跑不起来，见第 4 节） |
| `docs/BASELINE_AND_PROVENANCE.txt`、`docs/INTEGRATION_zh-CN.txt` | v2 基线的来源说明 |
| `report/reproducibility.md`、`report/figures/` | 报告「可复现性」草稿；v2 的诊断图 |
| `results/oof_final/` | **最终流程的折外预测**（707 张训练图，完整版 + 轻量版，提交格式），每张图和每个候选的明细；由 `analysis/export_oof_final.py` 生成。画报告图用这个，不需要 2 GB 的候选库 |
| `results/test_submissions_1113/` | **今晚自动提交的两份测试集 CSV 的副本**（`K1_refine_fp32.csv`、`K1_light_fp32.csv`）和 SHA256；拿重跑结果来比对用 |
| `results/evidence/` | 小型结果摘要：`final_sprint_1112/SUMMARY.json`（折外分数阶梯）、`full_v2_oof/`、`LIVE_SUBMISSIONS.json`（**只记到 10/23，需要补**）等 |
| `tests/` | 4 个 CPU 自检 |
| `requirements*.txt` | 固定版本的依赖（含 scikit-learn、notebook 内核） |

### 5.3 只在 Harrison 电脑上（被 `.gitignore` 忽略，不在 GitHub）

都在 `C:\Users\harri\code\filament-segmentation-2026\` 下面。做第 3 节的事**只可能用到前两行**；其余是重算本地分数、重做实验用的，需要时找 Harrison 要。

| 位置 | 大小 | 内容 | 什么时候需要 |
|---|---|---|---|
| `scratch/sub_1113/final/K1_light_fp32.csv`、`K1_refine_fp32.csv` | 各 0.3 MB | **今晚自动提交的两份**（各 1163 条）。**副本已放进仓库** `results/test_submissions_1113/` | 不需要找 Harrison 要了 |
| `scratch/sub_1113/SUBMIT_LOG.json`、`lb_before/`、`lb_after_*/` | 小 | 自动提交的日志和每次提交后下载的榜单 | 记提交结果时 |
| `scratch/sub_1112/` | 2 MB | 11/12 已交的 5 份（`full2_stack_b01_d048.csv` 是目前最好的 0.39） | 一般不需要（Kaggle 上能直接选） |
| `scratch/sub_1113/` 其余文件 | 11 MB | 中间版本的候选、`dense_fp32/`（测试集两个主模型的 float32 概率图）、CPU 复现日志 | 不需要 |
| `data/MAGFiLO_1.0_Kaggle_2026/` | 0.7 GB | 比赛数据（每人用自己的 Kaggle 账号下载） | 每人自己下 |
| `data/gong_unlabeled/`、`data/pseudo_s1/` | 1.2 GB、4.5 GB | S1 自训练用的无标注 GONG 图和伪标注（没通过） | 不需要 |
| `data/lt_frames/`、`data/lt_probe/` | 6.5 GB | LT 相邻帧实验的数据（没通过） | 不需要 |
| `runs/v2/`、`runs/v2b/` | 0.4 GB | 原 v2 的 5 个折模型和折外概率图（Kaggle 训练，来自 Zichang 的交接包） | 重算折外分数 |
| `runs/v2_5070ti/`、`runs/v2b_5070ti/` | 0.4 GB | 第二套 5 个折模型和折外概率图（5070 Ti，同配方） | 同上；两套平均是最终流程的折外依据 |
| `runs/v2_full_5070ti/`、`runs/v2_full_s2027_5070ti/` | 各 81 MB | 两个全量模型（= `weights/model_full_*.pt`）和它们的测试集概率图 | 已复制进 `weights/` |
| `runs/v2_s2028_5070ti/`、`runs/v2_full_s2028_5070ti/` | 0.5 GB | 第三套模型（M3，没通过） | 不需要 |
| `runs/S1_5070ti/`、`R1_2048_5070ti/`、`B1/B2/B3_5070ti/`、`LT_5070ti/`、`ms1344_k/`、`ms1728_k/` | 各约 0.1–0.2 GB | 没通过的实验的模型和折外概率图 | 不需要 |
| `runs/GPU_QUEUE.json`、`runs/*.log` | 小 | 显卡任务队列的记录 | 不需要 |
| `scratch/ws/main/rbank/kc_bridge/`、`test_full2_bridge/` | 2 GB | 最终流程的候选库（训练折外 6788 个候选、测试 1727 个，含小图） | 只有重算留/删或细化的折外分数才要。**画报告图不需要**：最终流程的折外预测已导出到仓库的 `results/oof_final/` |
| `scratch/ws/main/kcb/pred/` | 小 | 细化网络对折外候选的预测 | `keep_eval.py --shape refine` |
| `scratch/keep/oof/`、`scratch/keep/test/` | 小 | 留/删描述量表、交叉拟合分数、评估结果（`oof_v1/`、`test_v1/` 是修正前的版本，`oof_sym2/`、`oof_m3/` 是 M3 实验） | `keep_fit.py`、`keep_eval.py` |
| `scratch/ws/refine/run1/`、`run_kept/` | 1.7 GB（整个 `refine/`） | 细化网络的训练输出（`run1/refiner_f0..4.pt` = `weights/refiner_f*.pt`），`RESULTS.md` 是细化实验的详细记录 | 一般不需要 |
| `scratch/ws/main/` 其余 | 约 2 GB | Harrison 的实验脚本和中间结果（组合后处理、合并碎片、多尺度、M3 等）、自动提交脚本 `submit_1113.py` | 不需要 |
| `scratch/ws/k1_audit/`、`k1_reimpl/`、`x_*/` | 共约 0.2 GB | 复核、审查、探索各路的工作目录（报告已复制到 `docs/handoff_1112/`） | 做 K2 时可能需要 `x_keepmodel/`、`k1_reimpl/` |
| `scratch/bank/` | 95 MB | `gt_index.pkl`（训练标注像素索引，缺了会自动重建）和候选树实验的库 | 不需要 |
| 其他 `scratch/*` | — | 冒烟测试、旧的探测、临时目录 | 不需要 |

### 5.4 Kaggle 上

- 队伍 "Zichang Lu" 的全部提交在比赛页的 My Submissions 里（11/12 的 5 份和今晚的 2 份都在这里）。最终 2 份在那里手动勾选。
- 原 v2 的折模型最早在 Zichang 的私有 notebook `zichanglu/solar-seg-v2`、`solar-seg-v2b` 里（`runs/v2`、`runs/v2b` 就是从那里下的）。
- 11/07 那次「refactored notebook」提交引用的 commit 13a18c432efb 不在本仓库，应该在交的人自己那里。

## 6. 规矩和坑（队友的 Claude 务必遵守）

- **绝不使用测试图的标注**，包括 MAGFiLO 公开数据里的、别队的 notebook 输出和数据集。只用比赛的训练标注。
- 推理只用测试图本身。**任何描述量和决定都不能用文件名里的信息**（日期、观测站）；文件名只能用来分折和做分析。
- 每个实验先在 `EXPERIMENTS.md` 写好假设和通过标准再跑；通过标准是五折嵌套 ≥ +0.005、至少 4 折变好（今天的 K1 是事后补记，已注明并复核）。
- 不要把 Kaggle token、`.env` 写进任何文件或提交。
- **仓库公开前想清楚时机**：`PLAN.md` 写 11/13 公开，`README.md` 写「at competition close」。权重 + 代码一公开，别人在截止前就能直接复现我们的分数。先确认 Overview 的原话；如果允许，建议 11/15 06:00 UTC 提交截止后再公开、再填表。公开前查一遍提交历史里没有密钥。
- 仓库里的 `scratch/`、`runs/` 是忽略的，但不要用 `git add -A` 一把加，按文件名加。
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
