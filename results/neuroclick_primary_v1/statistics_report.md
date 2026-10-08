# NeuroClick-Hazard primary statistical analysis

Inferential unit: held-out participant (strict LOSOCV). Primary endpoint: first valid product-ROI visit. 
Tests: two-sided paired Wilcoxon signed-rank. Multiplicity: Holm correction across four 
pre-specified comparisons (two comparators × PR-AUC/MCC). Confidence intervals are 95% 
t-intervals over participant scores or participant-paired differences.

## Confirmatory paired comparisons

| Metric | Comparator | n | Candidate | Comparator | Difference (95% CI) | dz | W/T/L | p raw | p Holm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pr_auc | logreg_behavior | 42 | 0.3532 | 0.3052 | +0.0480 [+0.0251, +0.0708] | +0.653 | 32/0/10 | 0.000100382 | 0.000602293 |
| pr_auc | total_dwell | 42 | 0.3532 | 0.3053 | +0.0479 [+0.0266, +0.0691] | +0.702 | 30/0/12 | 7.8286e-05 | 0.000548002 |
| pr_auc | logreg_dwell_propensity | 42 | 0.3532 | 0.3557 | -0.0025 [-0.0146, +0.0096] | -0.064 | 22/0/20 | 0.75704 | 0.75704 |
| pr_auc | behavior_no_propensity | 42 | 0.3532 | 0.3008 | +0.0524 [+0.0321, +0.0727] | +0.805 | 31/0/11 | 1.09209e-05 | 8.73675e-05 |
| mcc | logreg_behavior | 42 | 0.2336 | 0.2112 | +0.0224 [-0.0090, +0.0539] | +0.223 | 24/0/18 | 0.213552 | 0.640657 |
| mcc | total_dwell | 42 | 0.2336 | 0.1991 | +0.0345 [+0.0058, +0.0632] | +0.375 | 27/2/13 | 0.0179976 | 0.0719902 |
| mcc | logreg_dwell_propensity | 42 | 0.2336 | 0.2198 | +0.0138 [-0.0102, +0.0379] | +0.179 | 23/0/19 | 0.358538 | 0.717076 |
| mcc | behavior_no_propensity | 42 | 0.2336 | 0.1938 | +0.0398 [+0.0136, +0.0660] | +0.473 | 27/1/14 | 0.00589345 | 0.0294673 |

Positive differences favor NeuroClick behavior. Statistical significance is assessed at 
Holm-adjusted p < 0.05. PR-AUC is the primary performance metric; MCC is confirmatory 
support for thresholded predictions.

## First-visit participant-level confidence intervals

| Model | Metric | n | Mean (95% CI) | SD |
|---|---|---:|---:|---:|
| behavior | pr_auc | 42 | 0.3532 [0.3160, 0.3904] | 0.1194 |
| behavior | mcc | 42 | 0.2336 [0.1993, 0.2679] | 0.1101 |
| behavior_eeg | pr_auc | 42 | 0.3340 [0.2985, 0.3696] | 0.1140 |
| behavior_eeg | mcc | 42 | 0.2095 [0.1750, 0.2439] | 0.1106 |
| behavior_eeg_et | pr_auc | 42 | 0.3277 [0.2905, 0.3649] | 0.1195 |
| behavior_eeg_et | mcc | 42 | 0.1951 [0.1542, 0.2360] | 0.1311 |
| behavior_et | pr_auc | 42 | 0.3569 [0.3164, 0.3974] | 0.1300 |
| behavior_et | mcc | 42 | 0.2260 [0.1818, 0.2703] | 0.1420 |
| behavior_no_propensity | pr_auc | 42 | 0.3008 [0.2642, 0.3373] | 0.1173 |
| behavior_no_propensity | mcc | 42 | 0.1938 [0.1557, 0.2319] | 0.1223 |
| logreg_behavior | pr_auc | 42 | 0.3052 [0.2679, 0.3425] | 0.1198 |
| logreg_behavior | mcc | 42 | 0.2112 [0.1706, 0.2518] | 0.1304 |
| logreg_dwell_propensity | pr_auc | 42 | 0.3557 [0.3211, 0.3902] | 0.1109 |
| logreg_dwell_propensity | mcc | 42 | 0.2198 [0.1898, 0.2498] | 0.0962 |
| total_dwell | pr_auc | 42 | 0.3053 [0.2674, 0.3432] | 0.1215 |
| total_dwell | mcc | 42 | 0.1991 [0.1549, 0.2433] | 0.1417 |
