# Current model and Ani ensemble

Source Ani commit: `a5487215115e30afb68ab865bc5fd98be6b3e199`, notebook snapshot saved in ani_source/. Ani's validation-stage scores and 985 pseudo-labeled trials are reproduced, with only 157 verified training labels. Its original feature-only transductive scaler/PCA/GMM use train+validation+test embeddings; no validation or test labels enter supervised training. No final 235-label refit was performed. Our DBSCAN classifier stays unchanged.

## Decision-score alignment

Our five independent scores use cutoff 0.5; Ani's multinomial/neighbor scores use cutoff 0.3. Center each model's log-odds at its own cutoff, then blend those evidence scores by Ani weight alpha. An ensemble score sigmoid(evidence)>=0.5 selects a reason, with highest-score fallback when empty. This preserves native decisions at alpha=0 and alpha=1. The ensemble scores are decision scores, not calibrated marginal probabilities; no raw probability average or extra cutoff optimization was used.

## Results

| Ani weight | Sample F1 | Macro F1 | Change vs ours | Changed trials | Improved trials | Worsened trials |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.910256 | 0.811421 | +0.000000 | 0 | 0 | 0 |
| 0.1 | 0.897436 | 0.801106 | -0.012821 | 1 | 0 | 1 |
| 0.25 | 0.897436 | 0.801106 | -0.012821 | 1 | 0 | 1 |
| 0.5 | 0.888889 | 0.776906 | -0.021368 | 3 | 0 | 3 |
| 1 | 0.893162 | 0.713700 | -0.017094 | 10 | 2 | 5 |

Complementary errors: {'different_reason_sets': 10, 'ani_better_trials': 2, 'ours_better_trials': 5, 'equal_trial_f1': 71}. Best setting: **ani_weight0**, sample F1 **0.910256**, macro F1 **0.811421**.

## Verification and saved artifacts

Both native endpoint controls reproduce their validation scores and labels. Saved component models and policy files recreate all ensemble scores and label sets, independently rescoring to the recorded metrics. The final Ani base is independently refitted from its saved augmented inputs/targets with matching probabilities. Protected original models and CSVs retain their hashes.

models/ saves both native component models and every decision policy. The Ani component also preserves the scaler/PCA/GMM, final augmented fitting inputs/targets, and pseudo-label trial IDs. datasets/ani_pseudo_labels.csv and ani_round_audit.csv document its self-training. complementary_errors.csv records trial-level differences; validation_predictions/ saves component probabilities, centered log-odds, ensemble decision scores, and labels. comparison.csv and settings.json preserve the grid, scores, source checksums, and software versions.

This is exploratory model selection on the same 78 validation trials. No independent Kaggle improvement is established. The ensemble does not use the new self-training model; the two ideas are evaluated separately. No Kaggle submission, commit, or tag was made.

No mixture improves sample F1; retain our existing DBSCAN model.
