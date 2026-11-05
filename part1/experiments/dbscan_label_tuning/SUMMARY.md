# DBSCAN confidence and pseudo-label influence experiment

48 confidence/weight combinations plus a labeled-only control were evaluated on the same 78 validation trials. Clustering, 150 embeddings, training-only scalers, classifier C=1000, probability cutoff 0.5, and nonempty fallback remain fixed.

Positive confidence thresholds: 0.5, 0.6, 0.7, 0.8. Negative thresholds: 0.05, 0.1, 0.2. Influence multipliers: 0.25, 0.5, 1, 2. Multipliers scale both the per-example maximum weight (base 0.25) and the per-class added-weight budget (base 0.5 of the corresponding known class weight). Known labels keep raw weight 1; per-reason fitting weights are subsequently normalized to a total of 157. Both generated classes must exist; their total weights retain the known-label positive/negative ratio.

Reference sample F1: **0.910256**, macro F1: **0.811421**.
Best configuration: **pos0p6_neg0p1_weight1**; sample F1 **0.910256**, macro F1 **0.811421**. Change in sample F1: **+0.000000**.
The 49 configurations produced **5 distinct target/weight tables**.

## Confidence changes at the original influence

| Positive threshold | Negative threshold | Influence multiplier | Generated trials | Sample F1 | Macro F1 |
|---:|---:|---:|---:|---:|---:|
| 0.6 | 0.1 | 1 | 133 | 0.910256 | 0.811421 |
| 0.5 | 0.1 | 1 | 133 | 0.910256 | 0.811421 |
| 0.7 | 0.1 | 1 | 133 | 0.910256 | 0.811421 |
| 0.6 | 0.05 | 1 | 133 | 0.910256 | 0.811421 |
| 0.5 | 0.05 | 1 | 133 | 0.910256 | 0.811421 |
| 0.7 | 0.05 | 1 | 133 | 0.910256 | 0.811421 |
| 0.6 | 0.2 | 1 | 133 | 0.910256 | 0.811421 |
| 0.5 | 0.2 | 1 | 133 | 0.910256 | 0.811421 |
| 0.7 | 0.2 | 1 | 133 | 0.910256 | 0.811421 |
| 0.8 | 0.1 | 1 | 0 | 0.893162 | 0.804324 |
| 0.8 | 0.05 | 1 | 0 | 0.893162 | 0.804324 |
| 0.8 | 0.2 | 1 | 0 | 0.893162 | 0.804324 |

## Influence changes at the original confidence

| Positive threshold | Negative threshold | Influence multiplier | Generated trials | Sample F1 | Macro F1 |
|---:|---:|---:|---:|---:|---:|
| 0.6 | 0.1 | 1 | 133 | 0.910256 | 0.811421 |
| 0.6 | 0.1 | 0.25 | 133 | 0.901709 | 0.811626 |
| 0.6 | 0.1 | 0.5 | 133 | 0.897436 | 0.801106 |
| 0.6 | 0.1 | 2 | 133 | 0.897436 | 0.801106 |
| 0.6 | 0.1 | 0 | 0 | 0.893162 | 0.804324 |

## All combinations

| Positive threshold | Negative threshold | Influence multiplier | Generated trials | Sample F1 | Macro F1 |
|---:|---:|---:|---:|---:|---:|
| 0.6 | 0.1 | 1 | 133 | 0.910256 | 0.811421 |
| 0.5 | 0.1 | 1 | 133 | 0.910256 | 0.811421 |
| 0.7 | 0.1 | 1 | 133 | 0.910256 | 0.811421 |
| 0.6 | 0.05 | 1 | 133 | 0.910256 | 0.811421 |
| 0.5 | 0.05 | 1 | 133 | 0.910256 | 0.811421 |
| 0.7 | 0.05 | 1 | 133 | 0.910256 | 0.811421 |
| 0.6 | 0.2 | 1 | 133 | 0.910256 | 0.811421 |
| 0.5 | 0.2 | 1 | 133 | 0.910256 | 0.811421 |
| 0.7 | 0.2 | 1 | 133 | 0.910256 | 0.811421 |
| 0.6 | 0.1 | 0.25 | 133 | 0.901709 | 0.811626 |
| 0.5 | 0.1 | 0.25 | 133 | 0.901709 | 0.811626 |
| 0.6 | 0.05 | 0.25 | 133 | 0.901709 | 0.811626 |
| 0.7 | 0.1 | 0.25 | 133 | 0.901709 | 0.811626 |
| 0.5 | 0.05 | 0.25 | 133 | 0.901709 | 0.811626 |
| 0.6 | 0.2 | 0.25 | 133 | 0.901709 | 0.811626 |
| 0.7 | 0.05 | 0.25 | 133 | 0.901709 | 0.811626 |
| 0.5 | 0.2 | 0.25 | 133 | 0.901709 | 0.811626 |
| 0.7 | 0.2 | 0.25 | 133 | 0.901709 | 0.811626 |
| 0.6 | 0.1 | 0.5 | 133 | 0.897436 | 0.801106 |
| 0.6 | 0.1 | 2 | 133 | 0.897436 | 0.801106 |
| 0.5 | 0.1 | 0.5 | 133 | 0.897436 | 0.801106 |
| 0.5 | 0.1 | 2 | 133 | 0.897436 | 0.801106 |
| 0.7 | 0.1 | 0.5 | 133 | 0.897436 | 0.801106 |
| 0.7 | 0.1 | 2 | 133 | 0.897436 | 0.801106 |
| 0.6 | 0.05 | 0.5 | 133 | 0.897436 | 0.801106 |
| 0.6 | 0.05 | 2 | 133 | 0.897436 | 0.801106 |
| 0.5 | 0.05 | 0.5 | 133 | 0.897436 | 0.801106 |
| 0.5 | 0.05 | 2 | 133 | 0.897436 | 0.801106 |
| 0.6 | 0.2 | 0.5 | 133 | 0.897436 | 0.801106 |
| 0.6 | 0.2 | 2 | 133 | 0.897436 | 0.801106 |
| 0.7 | 0.05 | 0.5 | 133 | 0.897436 | 0.801106 |
| 0.7 | 0.05 | 2 | 133 | 0.897436 | 0.801106 |
| 0.5 | 0.2 | 0.5 | 133 | 0.897436 | 0.801106 |
| 0.5 | 0.2 | 2 | 133 | 0.897436 | 0.801106 |
| 0.7 | 0.2 | 0.5 | 133 | 0.897436 | 0.801106 |
| 0.7 | 0.2 | 2 | 133 | 0.897436 | 0.801106 |
| 0.8 | 0.1 | 1 | 0 | 0.893162 | 0.804324 |
| 0.8 | 0.05 | 1 | 0 | 0.893162 | 0.804324 |
| 0.8 | 0.1 | 0.5 | 0 | 0.893162 | 0.804324 |
| 0.8 | 0.1 | 2 | 0 | 0.893162 | 0.804324 |
| 0.8 | 0.05 | 0.5 | 0 | 0.893162 | 0.804324 |
| 0.8 | 0.05 | 2 | 0 | 0.893162 | 0.804324 |
| 0.8 | 0.1 | 0.25 | 0 | 0.893162 | 0.804324 |
| 0.8 | 0.2 | 1 | 0 | 0.893162 | 0.804324 |
| 0.8 | 0.05 | 0.25 | 0 | 0.893162 | 0.804324 |
| 0.8 | 0.2 | 0.5 | 0 | 0.893162 | 0.804324 |
| 0.8 | 0.2 | 2 | 0 | 0.893162 | 0.804324 |
| 0.8 | 0.2 | 0.25 | 0 | 0.893162 | 0.804324 |
| 0.6 | 0.1 | 0 | 0 | 0.893162 | 0.804324 |

## Selected generated labels

| Reason | Generated positives | Generated negatives | Positive raw weight total | Negative raw weight total |
|---|---:|---:|---:|---:|
| Safety | 0 | 0 | 0.000000 | 0.000000 |
| Funding | 0 | 0 | 0.000000 | 0.000000 |
| Efficacy | 8 | 125 | 2.000000 | 16.470588 |
| Recruitment | 0 | 0 | 0.000000 | 0.000000 |
| Other | 0 | 0 | 0.000000 | 0.000000 |

## Reproducibility and interpretation

Original targets and weights remain unchanged for the 157 known trials, DBSCAN noise never receives generated labels, and the reference setting reproduces its exact targets and weights and matching probabilities. The zero-influence control reproduces the supervised baseline. Influence changes preserve accepted targets and scale raw pseudo weights exactly. The selected model was reloaded and independently refitted from its saved dataset, with matching validation probabilities. Protected previous inputs and artifacts retain their hashes.

Every dataset, label audit, classifier, validation prediction, configuration, and checksum is saved locally. Selection uses sample-wise F1; ties favor settings closest to the reference. Repeated tuning on this small validation set is exploratory and does not establish Kaggle improvement. No Kaggle submission was used; no commit or tag was made.

The primary validation metric did not improve; keep the existing DBSCAN candidate.
