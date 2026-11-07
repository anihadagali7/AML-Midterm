# Per-reason logistic regularization sweep

User-selected C values: 0.1,1,10,100,500,1000,2000. Vary one classifier at a time with the other four at C=1000, evaluating the complete five-reason sample F1. This is 35 reason/C evaluations and 31 unique multi-output configurations. Every classifier uses the same 150 embeddings/scaler, fixed DBSCAN targets/raw weights, normalized weight total 157 per reason, L2 penalty, class_weight=None, original solver settings, cutoff 0.5, and highest-probability fallback.

Reference sample F1 **0.910256**, macro F1 **0.811421**. Selected **baseline_C1000**: sample F1 **0.910256**, macro F1 **0.811421**. Selected C by reason: `{'Safety': 1000.0, 'Funding': 1000.0, 'Efficacy': 1000.0, 'Recruitment': 1000.0, 'Other': 1000.0}`.

## One-at-a-time sample F1

| Reason | C=0.1 | C=1 | C=10 | C=100 | C=500 | C=1000 | C=2000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Safety | 0.897436 | 0.901709 | 0.910256 | 0.910256 | 0.910256 | 0.910256 | 0.910256 |
| Funding | 0.897436 | 0.905983 | 0.905983 | 0.910256 | 0.910256 | 0.910256 | 0.910256 |
| Efficacy | 0.901709 | 0.884615 | 0.871795 | 0.893162 | 0.910256 | 0.910256 | 0.897436 |
| Recruitment | 0.897436 | 0.897436 | 0.897436 | 0.897436 | 0.910256 | 0.910256 | 0.897436 |
| Other | 0.884615 | 0.888889 | 0.901709 | 0.905983 | 0.910256 | 0.910256 | 0.910256 |

## Adaptive combinations

Shortlist: `{'Safety': 500.0, 'Funding': 500.0, 'Efficacy': 500.0, 'Recruitment': 500.0, 'Other': 500.0}`. For each reason, select its strongest nondefault C only when the individual score ties/improves the original baseline; ties favor values closest to C=1000, then smaller C. All subsets of at least two shortlisted changes were evaluated: **26** additional configurations; **57** unique configurations total. This is not an exhaustive 7^5 grid.

| Configuration | Sample F1 | Macro F1 | Change vs reference | Changed reasons |
|---|---:|---:|---:|---|
| baseline_C1000 | 0.910256 | 0.811421 | +0.000000 | none |
| solo_Efficacy_C500 | 0.910256 | 0.811421 | +0.000000 | Efficacy |
| solo_Funding_C2000 | 0.910256 | 0.811421 | +0.000000 | Funding |
| solo_Funding_C500 | 0.910256 | 0.811421 | +0.000000 | Funding |
| solo_Other_C2000 | 0.910256 | 0.811421 | +0.000000 | Other |
| solo_Other_C500 | 0.910256 | 0.811421 | +0.000000 | Other |
| solo_Recruitment_C500 | 0.910256 | 0.811421 | +0.000000 | Recruitment |
| solo_Safety_C2000 | 0.910256 | 0.811421 | +0.000000 | Safety |
| solo_Safety_C500 | 0.910256 | 0.811421 | +0.000000 | Safety |
| solo_Funding_C100 | 0.910256 | 0.811421 | +0.000000 | Funding |
| solo_Safety_C100 | 0.910256 | 0.811421 | +0.000000 | Safety |
| solo_Safety_C10 | 0.910256 | 0.811421 | +0.000000 | Safety |
| combined_EfficacyC500_OtherC500 | 0.910256 | 0.811421 | +0.000000 | Efficacy;Other |
| combined_EfficacyC500_RecruitmentC500 | 0.910256 | 0.811421 | +0.000000 | Efficacy;Recruitment |
| combined_FundingC500_EfficacyC500 | 0.910256 | 0.811421 | +0.000000 | Funding;Efficacy |
| combined_FundingC500_OtherC500 | 0.910256 | 0.811421 | +0.000000 | Funding;Other |
| combined_FundingC500_RecruitmentC500 | 0.910256 | 0.811421 | +0.000000 | Funding;Recruitment |
| combined_RecruitmentC500_OtherC500 | 0.910256 | 0.811421 | +0.000000 | Recruitment;Other |
| combined_SafetyC500_EfficacyC500 | 0.910256 | 0.811421 | +0.000000 | Safety;Efficacy |
| combined_SafetyC500_FundingC500 | 0.910256 | 0.811421 | +0.000000 | Safety;Funding |
| combined_SafetyC500_OtherC500 | 0.910256 | 0.811421 | +0.000000 | Safety;Other |
| combined_SafetyC500_RecruitmentC500 | 0.910256 | 0.811421 | +0.000000 | Safety;Recruitment |
| combined_EfficacyC500_RecruitmentC500_OtherC500 | 0.910256 | 0.811421 | +0.000000 | Efficacy;Recruitment;Other |
| combined_FundingC500_EfficacyC500_OtherC500 | 0.910256 | 0.811421 | +0.000000 | Funding;Efficacy;Other |
| combined_FundingC500_EfficacyC500_RecruitmentC500 | 0.910256 | 0.811421 | +0.000000 | Funding;Efficacy;Recruitment |
| combined_FundingC500_RecruitmentC500_OtherC500 | 0.910256 | 0.811421 | +0.000000 | Funding;Recruitment;Other |
| combined_SafetyC500_EfficacyC500_OtherC500 | 0.910256 | 0.811421 | +0.000000 | Safety;Efficacy;Other |
| combined_SafetyC500_EfficacyC500_RecruitmentC500 | 0.910256 | 0.811421 | +0.000000 | Safety;Efficacy;Recruitment |
| combined_SafetyC500_FundingC500_EfficacyC500 | 0.910256 | 0.811421 | +0.000000 | Safety;Funding;Efficacy |
| combined_SafetyC500_FundingC500_OtherC500 | 0.910256 | 0.811421 | +0.000000 | Safety;Funding;Other |
| combined_SafetyC500_FundingC500_RecruitmentC500 | 0.910256 | 0.811421 | +0.000000 | Safety;Funding;Recruitment |
| combined_SafetyC500_RecruitmentC500_OtherC500 | 0.910256 | 0.811421 | +0.000000 | Safety;Recruitment;Other |
| combined_FundingC500_EfficacyC500_RecruitmentC500_OtherC500 | 0.910256 | 0.811421 | +0.000000 | Funding;Efficacy;Recruitment;Other |
| combined_SafetyC500_EfficacyC500_RecruitmentC500_OtherC500 | 0.910256 | 0.811421 | +0.000000 | Safety;Efficacy;Recruitment;Other |
| combined_SafetyC500_FundingC500_EfficacyC500_OtherC500 | 0.910256 | 0.811421 | +0.000000 | Safety;Funding;Efficacy;Other |
| combined_SafetyC500_FundingC500_EfficacyC500_RecruitmentC500 | 0.910256 | 0.811421 | +0.000000 | Safety;Funding;Efficacy;Recruitment |
| combined_SafetyC500_FundingC500_RecruitmentC500_OtherC500 | 0.910256 | 0.811421 | +0.000000 | Safety;Funding;Recruitment;Other |
| combined_SafetyC500_FundingC500_EfficacyC500_RecruitmentC500_OtherC500 | 0.910256 | 0.811421 | +0.000000 | Safety;Funding;Efficacy;Recruitment;Other |
| solo_Other_C100 | 0.905983 | 0.809402 | -0.004274 | Other |
| solo_Funding_C10 | 0.905983 | 0.758088 | -0.004274 | Funding |
| solo_Funding_C1 | 0.905983 | 0.758088 | -0.004274 | Funding |
| solo_Safety_C1 | 0.901709 | 0.798944 | -0.008547 | Safety |
| solo_Efficacy_C0p1 | 0.901709 | 0.801027 | -0.008547 | Efficacy |
| solo_Other_C10 | 0.901709 | 0.803086 | -0.008547 | Other |
| solo_Efficacy_C2000 | 0.897436 | 0.803884 | -0.012821 | Efficacy |
| solo_Recruitment_C2000 | 0.897436 | 0.803884 | -0.012821 | Recruitment |
| solo_Recruitment_C100 | 0.897436 | 0.801106 | -0.012821 | Recruitment |
| solo_Recruitment_C10 | 0.897436 | 0.805022 | -0.012821 | Recruitment |
| solo_Recruitment_C1 | 0.897436 | 0.805022 | -0.012821 | Recruitment |
| solo_Funding_C0p1 | 0.897436 | 0.678088 | -0.012821 | Funding |
| solo_Recruitment_C0p1 | 0.897436 | 0.805022 | -0.012821 | Recruitment |
| solo_Safety_C0p1 | 0.897436 | 0.782633 | -0.012821 | Safety |
| solo_Efficacy_C100 | 0.893162 | 0.792111 | -0.017094 | Efficacy |
| solo_Other_C1 | 0.888889 | 0.795311 | -0.021368 | Other |
| solo_Efficacy_C1 | 0.884615 | 0.780426 | -0.025641 | Efficacy |
| solo_Other_C0p1 | 0.884615 | 0.786288 | -0.025641 | Other |
| solo_Efficacy_C10 | 0.871795 | 0.768961 | -0.038462 | Efficacy |

## Verification and artifacts

Each C=1000 reason fit reproduces its original probabilities. Every saved complete model reloads with matching probabilities, labels, and sample/macro F1. The selected model independently refits all five classifiers from the saved fixed data and normalized weight table with matching probabilities. Targets, raw weights, scaler, and source artifacts remain unchanged. Validation/test labels never enter fitting; no final validation-label refit was performed.

reason_models/ preserves all 35 binary fits. models/ and validation_predictions/ preserve every distinct complete configuration. datasets/ saves shared original data and exact normalized weights. fit_audit.csv records fitting rows/weights/convergence; individual_sweep.csv, best_standalone_by_reason.csv, combination_shortlist.csv, combination_results.csv, comparison.csv, and settings.json record settings, selections, results, and checksums.

The same repeatedly consulted 78 validation trials select settings, including the adaptive combination shortlist. These results do not establish independent test improvement. No Kaggle submission, commit, or tag was made.

No setting improves primary sample F1; retain the original all-C=1000 model and its existing prepared Kaggle file.
