# Full versus tied covariance: controlled K=10 check

Only GMM covariance type changes. Both methods use 150 embeddings, the same training-only scalers, 3 initializations, covariance regularization 0.001, maximum 500 EM iterations, partial-label rules, logistic regression C=1000, threshold 0.5, and nonempty fallback.

| Covariance | Generated trials | Sample-wise F1 | Macro F1 | Covariance parameters |
|---|---:|---:|---:|---:|
| tied | 696 | 0.9060 | 0.8116 | 11325 |
| full | 922 | 0.8675 | 0.7624 | 113250 |

Full covariance converged after **2** iterations. Its validation sample-wise F1 difference from tied covariance is **-0.0385**.

## Per-reason F1

| Covariance | Safety | Funding | Efficacy | Recruitment | Other |
|---|---:|---:|---:|---:|---:|
| tied | 0.9412 | 0.6667 | 0.8387 | 0.9800 | 0.6316 |
| full | 0.9412 | 0.6667 | 0.6429 | 0.9615 | 0.6000 |

The extra covariance parameters can improve or harm the label geometry; the validation result is the evidence for this particular setting. It does not establish a general rule about full covariance. Neither covariance candidate has a measured Kaggle score. The current reported best remains 0.87393.

`datasets/`, `label_audits/`, and `clusters/` preserve the exact augmented data and estimates. `clustering_models/` and `classifier_models/` preserve the fitted models. `validation_predictions/` preserves the compared predictions. `covariance_diagnostics.csv` records eigenvalue ranges. `settings.json` records configuration and checksums. Refit and reload checks reproduce the selected full model; all previous input/model/submission files remain unchanged.

Full covariance did not improve the primary validation metric; retain the tied-covariance candidate for the remaining Kaggle submission.
