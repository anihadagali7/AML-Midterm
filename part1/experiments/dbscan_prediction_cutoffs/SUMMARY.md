# Current DBSCAN leader: prediction cutoff and fallback

Frozen model: `dbscan_m5_q25__logistic_C1000`. No training, feature changes, pseudo-label changes, or Kaggle submission. The same 78 validation trials were evaluated at final cutoffs 0.40, 0.45, 0.50, 0.55, 0.60, with and without fallback.

| Cutoff | Sample F1 with fallback | Sample F1 without | Macro F1 with | Macro F1 without | Empty before fallback |
|---:|---:|---:|---:|---:|---:|
| 0.40 | 0.910256 | 0.876068 | 0.811421 | 0.804787 | 4 |
| 0.45 | 0.910256 | 0.876068 | 0.811421 | 0.804787 | 4 |
| 0.50 | 0.910256 | 0.876068 | 0.811421 | 0.804787 | 4 |
| 0.55 | 0.910256 | 0.876068 | 0.811421 | 0.804787 | 4 |
| 0.60 | 0.910256 | 0.876068 | 0.811421 | 0.804787 | 4 |

Selected policy: cutoff **0.50**, fallback **enabled**. Sample F1 **0.910256**, macro F1 **0.811421**. Ties prefer the existing cutoff 0.50 with fallback.

## Fallback contribution

| Cutoff | Trials needing fallback | Helped trials | Unchanged trials | Sample F1 gain |
|---:|---:|---:|---:|---:|
| 0.40 | 4 | 3 | 1 | 0.034188 |
| 0.45 | 4 | 3 | 1 | 0.034188 |
| 0.50 | 4 | 3 | 1 | 0.034188 |
| 0.55 | 4 | 3 | 1 | 0.034188 |
| 0.60 | 4 | 3 | 1 | 0.034188 |

Without fallback, trials with no selected labels remain all-zero predictions. With sample-wise F1 and zero_division=0, selecting a reason for such a trial either leaves its F1 at zero or increases it if the selected reason is correct. Thus fallback cannot lower sample-wise F1 for these empty predictions under this scoring convention, although it can change per-label precision and macro F1.

## Saved artifacts and checks

`validation_probabilities.csv` freezes the single probability matrix used by all settings. `validation_predictions/` saves all ten label sets and per-trial scores. `comparison.csv` includes sample, macro, per-reason F1, empty predictions, and changes versus the current policy. `fallback_comparison.csv`, `fallback_cases.csv`, and `per_trial_comparison.csv` explain the changes. `selected_policy.json` references the original model and chosen postprocessing; no model was retrained.

The original cutoff/fallback reproduces the previous predictions and sample F1. All saved prediction sets were reloaded and independently rescored, threshold monotonicity and fallback invariants passed, and protected inputs and model artifacts retained their hashes.

Selection uses the repeatedly consulted validation set and does not establish independent test improvement. No Kaggle submission, commit, or tag was made.
