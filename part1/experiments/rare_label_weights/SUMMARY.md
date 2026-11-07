# Funding and Other positive-class weighting

16 classifier settings cross positive multipliers 1/1.5/2/3 for Funding and Other, including the original 1/1 control. Source DBSCAN targets/raw weights stay unchanged. Only effective fitting weights for those two binary classifiers change: multiply positive weights, leave negative weights unchanged, then normalize each reason's total to 157. Safety, Efficacy, and Recruitment estimators are reused unchanged. All classifiers retain C=1000, original 150 embeddings/scaler, cutoff 0.5, and highest-probability fallback. No new class priors are assumed to be population rates.

Reference sample F1 **0.910256**, macro F1 **0.811421**. Selected **funding1_other1**: sample F1 **0.910256**, macro F1 **0.811421**.

| Funding positive weight | Other positive weight | Sample F1 | Macro F1 | Funding F1 | Other F1 | Change |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.910256 | 0.811421 | 0.666667 | 0.631579 | +0.000000 |
| 1 | 1.5 | 0.910256 | 0.811421 | 0.666667 | 0.631579 | +0.000000 |
| 1 | 2 | 0.910256 | 0.811421 | 0.666667 | 0.631579 | +0.000000 |
| 1 | 3 | 0.910256 | 0.811421 | 0.666667 | 0.631579 | +0.000000 |
| 1.5 | 1 | 0.910256 | 0.811421 | 0.666667 | 0.631579 | +0.000000 |
| 1.5 | 1.5 | 0.910256 | 0.811421 | 0.666667 | 0.631579 | +0.000000 |
| 1.5 | 2 | 0.910256 | 0.811421 | 0.666667 | 0.631579 | +0.000000 |
| 1.5 | 3 | 0.910256 | 0.811421 | 0.666667 | 0.631579 | +0.000000 |
| 2 | 1 | 0.910256 | 0.811421 | 0.666667 | 0.631579 | +0.000000 |
| 2 | 1.5 | 0.910256 | 0.811421 | 0.666667 | 0.631579 | +0.000000 |
| 2 | 2 | 0.910256 | 0.811421 | 0.666667 | 0.631579 | +0.000000 |
| 2 | 3 | 0.910256 | 0.811421 | 0.666667 | 0.631579 | +0.000000 |
| 3 | 1 | 0.910256 | 0.811421 | 0.666667 | 0.631579 | +0.000000 |
| 3 | 1.5 | 0.910256 | 0.811421 | 0.666667 | 0.631579 | +0.000000 |
| 3 | 2 | 0.910256 | 0.811421 | 0.666667 | 0.631579 | +0.000000 |
| 3 | 3 | 0.910256 | 0.811421 | 0.666667 | 0.631579 | +0.000000 |

The original control reproduces its probabilities. All saved models and prediction sets reload and rescore correctly; saved effective weights reproduce the declared multipliers and fixed total. The selected model independently refits from its saved source data and effective weights with matching probabilities. The untouched three classifiers and protected source files retain their parameters/checksums.

datasets/ saves the shared original data plus exact effective weights per configuration; models/ and validation_predictions/ preserve all fits and outputs. comparison.csv includes per-reason F1 and Funding/Other precision, recall, true positives, false positives, and false negatives; weight_audit.csv records effective class weight totals; settings.json preserves settings and hashes.

This last bounded screen uses the same repeatedly consulted 78-trial validation set and does not establish independent test improvement. No validation labels enter fitting, and no final refit, Kaggle submission, commit, or tag was performed.

No setting improves primary sample F1; keep the existing DBSCAN leader and its prepared Kaggle file.

## Weight-effect diagnostic

All 16 settings produce exactly the same validation label sets as the original model. The weights are active: at Funding=3 and Other=3, fitted coefficients change and the maximum absolute validation probability changes are approximately **0.02443 for Funding** and **0.35611 for Other**. None of these changes alters the final cutoff/fallback decisions. Safety, Efficacy, and Recruitment coefficients and probabilities remain exactly unchanged. `weight_effect_diagnostic.json` records these checks. Retain the original model and its existing prepared Kaggle submission.
