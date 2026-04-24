# TIDE Final Tables

This folder is the single official export location for paper tables and long-format results.

- Full result root: `D:\准备工作\论文\DIRT\projects\Dynamic-Cognitive-Diagnosis\ws\chapter3_runs`
- Base result root: `D:\准备工作\论文\DIRT\projects\Dynamic-Cognitive-Diagnosis\ws\chapter3_runs`
- Datasets: `assist2009, assist2012, kddcup`
- Seed: `2024`

Files:
- `table_3_1_dataset_statistics.csv`
- `table_3_2_training_config.csv`
- `table_3_3_main_results.csv`
- `table_3_4_ablation.csv`
- `table_3_4_ablation_detail.csv`
- `table_3_5_stage_comparison.csv`
- `long_results.csv`
- `tables.md`
- `tables.json`

Notes:
- `table_3_3_main_results.csv` uses only official TIDE and base DIRT result paths.
- `table_3_4_ablation.csv` reports ASSIST2009 ablations separately for `TIDE_1~4`.
- `table_3_4_ablation_detail.csv` adds best stage / epoch metadata for each ablation run.
- `table_3_5_stage_comparison.csv` reports `TIDE_3 / TIDE_4` stage-wise metrics.
- If workspaces are empty, result tables will be created with headers but no metrics.
