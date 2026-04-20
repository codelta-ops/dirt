# Chapter 3 Tables

This export uses local files found in the current workspace snapshot. Missing experiment logs are filled as `N/A`.

## dataset_statistics

| Dataset | #Students | #Exercises | #Concepts | #Interactions | Correct Rate | Avg Length | Max Length |
| --- | --- | --- | --- | --- | --- | --- | --- |
| assist2009 | 3079 | 17671 | 123 | 271468 | 0.658295 | 88.167587 | 200 |

## main_results

| Model | AUC | ACC | RMSE |
| --- | --- | --- | --- |
| DIRT_1 | 0.7067 | 0.6941 | N/A |
| DIRT_2 | 0.7427 | 0.7223 | N/A |
| DIRT_3 | 0.7430 | 0.7225 | N/A |
| DIRT_4 | 0.7442 | 0.7238 | N/A |
| DIRT+_1 | 0.712096 | 0.694234 | 0.466178 |
| DIRT+_2 | 0.749549 | 0.729484 | 0.454461 |
| DIRT+_3 | 0.747248 | 0.733412 | 0.454641 |
| DIRT+_4 | 0.745832 | 0.732656 | 0.454899 |

## ablation_results

| Variant | AUC | ACC | RMSE |
| --- | --- | --- | --- |
| Full Model | 0.746540 | 0.733034 | 0.454770 |
| w/o attention | 0.746290 | 0.729585 | 0.457251 |
| w/o query | 0.741246 | 0.720864 | 0.456605 |
| w/o weight | 0.746020 | 0.731909 | 0.457084 |
| w/o consistency | 0.745903 | 0.731876 | 0.455053 |
| w/o adaptive fusion | N/A | N/A | N/A |
| w/o temporal bias | N/A | N/A | N/A |
| w/o exercise-aware decay | N/A | N/A | N/A |

## stage_comparison

| Model | Stage | Validation AUC | Validation ACC | Validation RMSE | Test AUC | Test ACC | Test RMSE | Best Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DIRT+_3 | Stage1 | 0.750296 | 0.735740 | 0.454187 | N/A | N/A | N/A | 5 |
| DIRT+_3 | Stage2 | 0.748869 | 0.736344 | 0.454147 | 0.747248 | 0.733412 | 0.454641 | 3 |
| DIRT+_4 | Stage1 | 0.748176 | 0.732420 | 0.455061 | N/A | N/A | N/A | 5 |
| DIRT+_4 | Stage2 | 0.746241 | 0.734068 | 0.454823 | 0.745832 | 0.732656 | 0.454899 | 0 |

## training_configuration

| Setting | Value |
| --- | --- |
| batch size | 32 |
| stage1 epochs | N |
| stage2 epochs | A / N/A |
| stage1 lr | 0.002 |
| stage2 lr | 0.0006 |
| optimizer | Adam |
| seed | 2024 |
| Attention | Yes |
| Query-guided aggregation | Yes |
| loss_weight_mode | learnable |
| use_confidence_consistency | Yes |
| grad clip | 5.000000 |
| scheduler | ReduceLROnPlateau (monitor validation AUC) |
| use_adaptive_fusion | 1 |
| use_temporal_bias | 1 |
| use_exercise_aware_decay | 1 |
| use_state_consistency | 0 |

## mechanism_analysis

| step_weight_mean | step_weight_std | step_weight_min | step_weight_max | teacher_confidence_mean | normalized_confidence_mean | consistency_loss | fusion_gate_rnn_mean | fusion_gate_attn_mean | fusion_gate_query_mean | decay_gamma_mean | decay_gamma_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## multi_seed_stability

| Model | AUC(mean±std) | ACC(mean±std) | RMSE(mean±std) |
| --- | --- | --- | --- |
| DIRT+ default configuration | N/A | N/A | N/A |

## Suggested Placement

- Dataset statistics: 3.1 Dataset and Preprocessing
- Main results: 3.3 Overall Performance Comparison
- Ablation results: 3.4 Ablation Study
- Stage comparison: 3.5 Stage-wise Analysis
- Training configuration: 3.2 Experimental Setup
- Mechanism analysis: 3.6 Mechanism Analysis / Interpretability
- Multi-seed stability: 3.7 Stability Analysis
