## table1_main_results

| Model | AUC | ACC | RMSE |
| --- | --- | --- | --- |
| BKT | N/A | N/A | N/A |
| DKVMN | N/A | N/A | N/A |
| DKT_Q | N/A | N/A | N/A |
| DKT_KC | N/A | N/A | N/A |
| DKT_MLP | N/A | N/A | N/A |
| DIRT_1 | N/A | N/A | N/A |
| DIRT_2 | N/A | N/A | N/A |
| DIRT_3 | N/A | N/A | N/A |
| DIRT_4 | N/A | N/A | N/A |
| DNeuralCDM_1 | N/A | N/A | N/A |
| DNeuralCDM_2 | N/A | N/A | N/A |
| DNeuralCDM_3 | N/A | N/A | N/A |
| DNeuralCDM_4 | N/A | N/A | N/A |
| DIRT+_1 | 0.712096 | 0.694234 | 0.466178 |
| DIRT+_2 | 0.749549 | 0.729484 | 0.454461 |
| DIRT+_3 | 0.747248 | 0.733412 | 0.454641 |
| DIRT+_4 | 0.745832 | 0.732656 | 0.454899 |

## table2_ablation

| Variant | AUC | ACC | RMSE |
| --- | --- | --- | --- |
| Full Model | 0.746540 | 0.733034 | 0.454770 |
| w/o attention | 0.746290 | 0.729585 | 0.457251 |
| w/o query | 0.741246 | 0.720864 | 0.456605 |
| w/o weight | 0.746020 | 0.731909 | 0.457084 |
| w/o consistency | 0.745903 | 0.731876 | 0.455053 |

## table3_stage_comparison

| Model | Stage | Validation AUC | Validation ACC | Validation RMSE | Test AUC | Test ACC | Test RMSE | Best Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DIRT+_3 | stage1 | 0.750296 | 0.735740 | 0.454187 | N/A | N/A | N/A | 5 |
| DIRT+_3 | stage2 | 0.748869 | 0.736344 | 0.454147 | 0.747248 | 0.733412 | 0.454641 | 3 |
| DIRT+_4 | stage1 | 0.748176 | 0.732420 | 0.455061 | N/A | N/A | N/A | 5 |
| DIRT+_4 | stage2 | 0.746241 | 0.734068 | 0.454823 | 0.745832 | 0.732656 | 0.454899 | 0 |

## table4_dataset_statistics

| Dataset | #Students | #Exercises | #Concepts | #Interactions | Correct Rate |
| --- | --- | --- | --- | --- | --- |
| assist2009 | 3079 | 17671 | 123 | 271468 | 0.658295 |

## table5_training_config

| Setting | Value |
| --- | --- |
| batch size | 32 |
| epochs (stage1 / stage2) | 10 / 5 |
| learning rate | stage1=0.002, stage2=0.0006 |
| optimizer | Adam |
| seed | 2024 |
| attention | Yes |
| query | Yes |
| dynamic weight | learnable |
| consistency | Yes |
