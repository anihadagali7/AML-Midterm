# Focused DBSCAN and logistic regression refinement

A fixed local grid evaluates 9 density settings and 3 classifier regularization strengths each, using only the same 150 embeddings. The previous DBSCAN + LR reference is sample-wise F1 0.9103 and macro F1 0.8114 on the same 78 validation trials.

Radius center: **0.00012142366141394104**. Radius factors: **0.5,1,2**. Min_samples: **3,5,8**. C: **100,1000,10000**.

| Min samples | Radius factor | C | Generated trials | Sample-wise F1 | Macro F1 | Change from reference |
|---|---:|---:|---:|---:|---:|---:|
| 5 | 1 | 1000 | 133 | 0.9103 | 0.8114 | +0.0000 |
| 3 | 1 | 1000 | 133 | 0.9103 | 0.8114 | +0.0000 |
| 8 | 1 | 1000 | 133 | 0.9103 | 0.8114 | +0.0000 |
| 5 | 0.5 | 1000 | 81 | 0.8974 | 0.8039 | -0.0128 |
| 5 | 2 | 1000 | 173 | 0.8974 | 0.8039 | -0.0128 |
| 5 | 1 | 10000 | 133 | 0.8974 | 0.8039 | -0.0128 |
| 5 | 0.5 | 10000 | 81 | 0.8974 | 0.8039 | -0.0128 |
| 5 | 2 | 10000 | 173 | 0.8974 | 0.8039 | -0.0128 |
| 3 | 0.5 | 1000 | 81 | 0.8974 | 0.8039 | -0.0128 |
| 3 | 2 | 1000 | 173 | 0.8974 | 0.8039 | -0.0128 |
| 3 | 1 | 10000 | 133 | 0.8974 | 0.8039 | -0.0128 |
| 8 | 0.5 | 1000 | 81 | 0.8974 | 0.8039 | -0.0128 |
| 8 | 2 | 1000 | 173 | 0.8974 | 0.8039 | -0.0128 |
| 3 | 0.5 | 10000 | 81 | 0.8974 | 0.8039 | -0.0128 |
| 3 | 2 | 10000 | 173 | 0.8974 | 0.8039 | -0.0128 |
| 8 | 1 | 10000 | 133 | 0.8974 | 0.8039 | -0.0128 |
| 8 | 0.5 | 10000 | 81 | 0.8974 | 0.8039 | -0.0128 |
| 8 | 2 | 10000 | 173 | 0.8974 | 0.8039 | -0.0128 |
| 5 | 1 | 100 | 133 | 0.8889 | 0.7901 | -0.0214 |
| 5 | 0.5 | 100 | 81 | 0.8889 | 0.7901 | -0.0214 |
| 5 | 2 | 100 | 173 | 0.8889 | 0.7939 | -0.0214 |
| 3 | 1 | 100 | 133 | 0.8889 | 0.7901 | -0.0214 |
| 3 | 0.5 | 100 | 81 | 0.8889 | 0.7901 | -0.0214 |
| 3 | 2 | 100 | 173 | 0.8889 | 0.7939 | -0.0214 |
| 8 | 1 | 100 | 133 | 0.8889 | 0.7901 | -0.0214 |
| 8 | 0.5 | 100 | 81 | 0.8889 | 0.7901 | -0.0214 |
| 8 | 2 | 100 | 173 | 0.8889 | 0.7939 | -0.0214 |

Best local configuration: **local_m5_r1p0__logistic_C1000**, sample-wise F1 **0.9103**, macro F1 **0.8114**.

All label confidence thresholds, per-reason budgets, training-only scalers, weight normalization, threshold 0.5, and nonempty fallback remain fixed. The central configuration reproduces the previous model's generated targets, weights, and probabilities. Original known labels are preserved, DBSCAN noise stays unknown for generated targets, and validation never enters fitting. The selected model is reloaded and independently refitted from its saved dataset with matching predictions.

This is a bounded local search selected using validation, not a proof of a global maximum or independent Kaggle improvement. Nearby density settings can produce identical label datasets. No Kaggle submission was used and the best-submitted tag remains unchanged.

`datasets/`, `label_audits/`, `clusters/`, `labeling_models/`, `classifier_models/`, and `validation_predictions/` preserve all inputs and outputs. `comparison.csv`, `density_comparison.csv`, `label_counts.csv`, and `settings.json` preserve the complete grid, diagnostics, settings, and checksums.

The grid did not improve the primary validation metric; retain the existing DBSCAN candidate rather than creating a new submission file.
