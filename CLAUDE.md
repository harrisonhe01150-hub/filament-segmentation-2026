# CLAUDE.md

Kaggle「Solar Filament Segmentation Challenge 2026」团队仓库（队伍 "Zichang Lu"：Zichang Lu、xianyang zeng、Harrison He）。

**先读 `HANDOFF.md`**：现状、剩下要做的事、截止时间；第 5 节列出所有文件在哪（仓库里、只在 Harrison 电脑上、Kaggle 上）。

## 硬规矩

- 绝不使用测试图的标注（MAGFiLO 公开数据里有；别队的 notebook 输出和数据集也不行）。只用比赛的训练标注。
- 推理只用测试图本身；描述量和决定不能用文件名里的日期、观测站。
- 实验先在 `EXPERIMENTS.md` 预登记假设和通过标准再跑；每次 Kaggle 提交都记进 `EXPERIMENTS.md`。全队每天 5 次，00:00 UTC 重置。
- Kaggle token、`.env` 不进任何文件和提交。
- `weights/` 里是最终流程的权重（已提交，SHA256 见 `weights/README.md`）；按文件名 `git add`，不要 `git add -A`。
- 改了 `src/`、`analysis/`、`scripts/`、`tests/`、`configs/` 后跑 `python scripts/update_manifest.py`（notebook 会核对哈希）。

## 地图

- 最终流程：`src/predict_final.py`（一条命令）、`notebooks/end_to_end_final.ipynb`（由 `scripts/build_notebook_final.py` 生成）、大白话说明 `docs/FINAL_PIPELINE_zh-CN.md`。
- v2 基线：`src/train_seg.py`、`src/predict.py`、`notebooks/end_to_end.ipynb`。
- 分数口径：五折折外、官方自评规则（`src/common.py`、`src/postproc.score_file`）。
