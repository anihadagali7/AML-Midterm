# Labeling and classifier comparison: 150 embeddings

No year, additional PCA, text, or drug features were used. All validation scores use the same 78 trials and the same 0.5 threshold with nonempty fallback.

## Frozen submitted references

| Model | Sample-wise F1 | Macro F1 | Reported Kaggle |
|---|---:|---:|---:|
| submitted_supervised | 0.8932 | 0.8043 | 0.87072 |
| submitted_k02 | 0.8568 | 0.6808 | 0.87393 |

## Best new classifier configurations

| Configuration | Sample-wise F1 | Macro F1 | Generated trials |
|---|---:|---:|---:|
| gmm_tied_k10__logistic_C1000 | 0.9060 | 0.8116 | 696 |
| kmeans_k08__logistic_C1000 | 0.9017 | 0.8091 | 803 |
| gmm_diag_k02__logistic_C10000 | 0.8974 | 0.8039 | 0 |
| known_only__logistic_C10000 | 0.8974 | 0.8039 | 0 |
| kmeans_k08__logistic_C10000 | 0.8974 | 0.8013 | 803 |
| gmm_diag_k02__logistic_C1000 | 0.8932 | 0.8043 | 0 |
| known_only__logistic_C1000 | 0.8932 | 0.8043 | 0 |
| kmeans_k08__logistic_C100 | 0.8932 | 0.7801 | 803 |
| gmm_diag_k02__logistic_C100 | 0.8889 | 0.7939 | 0 |
| known_only__logistic_C100 | 0.8889 | 0.7939 | 0 |
| gmm_tied_k10__logistic_C100 | 0.8889 | 0.7901 | 696 |
| gmm_tied_k10__logistic_C10000 | 0.8889 | 0.7937 | 696 |

Best new configuration: **gmm_tied_k10__logistic_C1000**, validation sample-wise F1 **0.9060**.
Best augmented configuration: **gmm_tied_k10__logistic_C1000**, validation sample-wise F1 **0.9060**.

## Clustering and generated labels

| Family | Best K | Sample-wise F1 with C=1000 | Macro F1 | Generated trials |
|---|---:|---:|---:|---:|
| kmeans | 8 | 0.9017 | 0.8091 | 803 |
| gmm_diag | 2 | 0.8932 | 0.8043 | 0 |
| gmm_tied | 10 | 0.9060 | 0.8116 | 696 |

For the selected augmented dataset, generated-label counts are:

| Reason | Generated positives | Generated negatives | Positive weight | Negative weight |
|---|---:|---:|---:|---:|
| Safety | 0 | 0 | 0.000 | 0.000 |
| Funding | 0 | 0 | 0.000 | 0.000 |
| Efficacy | 0 | 0 | 0.000 | 0.000 |
| Recruitment | 608 | 88 | 39.679 | 22.000 |
| Other | 0 | 0 | 0.000 | 0.000 |

## Method and limits

Stage 1 compares 15 clustering settings with the same partial-label rule and C=1000 classifier. Stage 2 shortlists one dataset per clustering family plus known-only and legacy K=2 labels, then evaluates seven classifier settings each. Some low-component labelers abstain entirely and reproduce the known-only dataset; those are not augmentation improvements. The result is the best of this staged search, not a global optimum over every method or parameter combination.

Each reason is assigned independently. Confident positives need estimated prevalence >=0.6; confident negatives need <=0.1; at least 3 effective labeled members are required. Component estimates use prior strength 2. Generated positive and negative weights are added in the same fraction of their respective known classes, capped at 0.5, with per-example weight <=0.25. If either generated class is absent, neither is used for that reason. This prevents generated negatives alone from drowning rare positive classes, without forcing equal class frequencies. Preserving known class balance is an influence-control choice, not an assertion about population prevalence.

All new classifiers use the same frozen scaler fitted on 157 known-label trials. Logistic fitting weights are normalized to a total of 157 per reason to hold effective regularization comparable. Weighted nearest-neighbor votes include both inverse distance and sample weights. Exact-distance matches vote alone. The frozen submitted K=2 model used its original scaler and weight total; its controlled re-evaluation is a separate candidate.

Unknown targets stay -100 with weight zero and are excluded independently per reason. Original known labels are preserved. No validation trial participates in fitting. Validation has been used to select among many candidates, and the new Kaggle scores remain unmeasured; small validation gains should not be presented as confirmed test improvements.

## Artifacts and verification

`datasets/` saves each exact feature/target/weight table. `label_audits/`, `clusters/`, and `label_counts.csv` expose generated labels and budgets. `clustering_models/` and `classifier_models/` save fitted models. `validation_predictions/` saves probabilities and labels. `clustering_comparison.csv`, `classifier_comparison.csv`, and `comparison.csv` retain every evaluated configuration. `settings.json` records settings, selection, versions, and checksums.

Checks reproduced baseline probabilities, matched weighted-neighbor predictions to scikit-learn when all weights equal one, preserved known labels and source artifacts, enforced partial-label/weight masks and budgets, and replayed saved selected models and data.

## Candidate Kaggle files

These are local files for manual upload; neither candidate has a measured Kaggle score. Compare against **0.87393**.

| Candidate | Validation F1 | Submission file | Changed test trials from original K=2 |
|---|---:|---|---:|
| gmm_tied_k10__logistic_C1000 | 0.9060 | `submissions/part1_gmm_tied_k10_logistic_C1000_submission.csv` | 54 |
| kmeans_k08__logistic_C1000 | 0.9017 | `submissions/part1_kmeans_k08_logistic_C1000_submission.csv` | 58 |
