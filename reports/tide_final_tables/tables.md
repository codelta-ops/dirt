# TIDE Final Tables

## table_3_1_dataset_statistics

| Dataset | #Students | #Exercises | #Concepts | #Interactions | Correct Rate |
| --- | --- | --- | --- | --- | --- |
| ASSIST2009 | 3079 | 17671 | 123 | 271468 | 0.658295 |
| ASSIST2012 | 25270 | 32726 | 252 | 2536321 | 0.697891 |
| KDDCup | 7528 | 26964 | 373 | 1408029 | 0.837158 |

## table_3_2_training_config

| Setting | Value |
| --- | --- |
| cross_idx | 0 |
| seed | 2024 |
| batch size | 32 |
| epochs (stage1 / stage2) | 10 / 5 |
| learning rate | stage1=0.002, stage2=0.002 |
| optimizer | Adam |
| causal temporal enhancement | Yes |
| target-aware history aggregation | Yes |
| dynamic step weight | learnable |
| stage2 consistency | Yes |
| scheduler | ReduceLROnPlateau (monitor validation AUC, patience=1) |
| stage2 applies to | TIDE_3 / TIDE_4 only |

## table_3_3_main_results

| Model | ASSIST2009 AUC | ASSIST2009 ACC | ASSIST2012 AUC | ASSIST2012 ACC | KDDCup AUC | KDDCup ACC |
| --- | --- | --- | --- | --- | --- | --- |
| DIRT_1 |  |  |  |  |  |  |
| DIRT_2 |  |  |  |  |  |  |
| DIRT_3 |  |  |  |  |  |  |
| DIRT_4 |  |  |  |  |  |  |
| TIDE_1 |  |  |  |  |  |  |
| TIDE_2 |  |  |  |  |  |  |
| TIDE_3 |  |  |  |  |  |  |
| TIDE_4 |  |  |  |  |  |  |

## table_3_4_ablation

| Model | Variant | Dataset | Seed | AUC | ACC | RMSE |
| --- | --- | --- | --- | --- | --- | --- |

## table_3_4_ablation_detail

| Model | Variant | Dataset | Seed | AUC | ACC | RMSE | Best Stage | Best Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## table_3_5_stage_comparison

| Dataset | Model | Stage | Validation AUC | Validation ACC | Validation RMSE | Test AUC | Test ACC | Test RMSE | Best Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## long_results

| 模型 | 数据集 | 种子 | AUC | ACC |
| --- | --- | --- | --- | --- |
