# Additional classical ML methods

All models use only the same 150 embeddings, the same 78 validation trials, threshold 0.5, and nonempty fallback. No neural networks, extra PCA, year, text, or drug features were used. No Kaggle submission was made.

Current validation reference: **0.9060**. Best new model: **dbscan_m5_q25__logistic_C1000**, sample-wise F1 **0.9103**. Leading model after this round: **dbscan_m5_q25__logistic_C1000**.

## Best result by method

| Method | Configuration | Sample-wise F1 | Macro F1 | Change from 0.9060 reference |
|---|---|---:|---:|---:|
| rbf_svm | frozen_gmm_tied_k10__rbf_svm_C10p0_gammascale | 0.8675 | 0.7206 | -0.0385 |
| gaussian_nb | graph_n20_a0p2__crossover_gaussian_nb | 0.8295 | 0.7200 | -0.0765 |
| logistic | dbscan_m5_q25__logistic_C1000 | 0.9103 | 0.8114 | +0.0043 |
| graph_direct | graph_n10_a0p2__direct | 0.8205 | 0.5402 | -0.0855 |
| weighted_knn | dbscan_m5_q25__weighted_knn_k1 | 0.8547 | 0.6690 | -0.0513 |

## Leading new configurations

| Configuration | Sample-wise F1 | Macro F1 |
|---|---:|---:|
| dbscan_m5_q25__logistic_C1000 | 0.9103 | 0.8114 |
| dbscan_m10_q25__logistic_C1000 | 0.8974 | 0.8039 |
| dbscan_m10_q50__logistic_C1000 | 0.8932 | 0.7987 |
| dbscan_m10_q75__logistic_C1000 | 0.8932 | 0.8043 |
| dbscan_m5_q75__logistic_C1000 | 0.8932 | 0.8043 |
| dbscan_m5_q50__logistic_C1000 | 0.8889 | 0.7896 |
| hybrid_gmm_dbscan__logistic_C1000 | 0.8889 | 0.7937 |
| frozen_gmm_tied_k10__rbf_svm_C10p0_gammascale | 0.8675 | 0.7206 |
| graph_n20_a0p2__logistic_C1000 | 0.8632 | 0.7199 |
| graph_n5_a0p8__logistic_C1000 | 0.8632 | 0.6710 |
| frozen_gmm_tied_k10__rbf_svm_C1p0_gammascale | 0.8590 | 0.7128 |
| known_only__rbf_svm_C1p0_gammascale | 0.8590 | 0.7128 |

## Method details and limits

RBF SVM compares C=0.1,1,10 and gamma=0.001,scale,0.03 on known-only and frozen GMM labels. Its probabilities are calibrated with sigmoid calibration and three stratified training-only folds; validation is excluded. GaussianNB compares smoothing 1e-9,0.001,0.1 on the same two datasets. It accepts continuous signed features; MultinomialNB is not appropriate for these embeddings.

DBSCAN compares min_samples=5,10 and radii taken from the 25th/50th/75th percentiles of training neighbor distances. Noise is excluded from generated targets. LabelSpreading compares 5,10,20 neighbors and alpha=0.2,0.8, one graph per reason. Missing labels are -1 internally for sklearn and remain -100 in saved uncertain targets. For augmentation, graph components need at least 3 known-label members and nonzero propagated mass. Both labelers reuse the fixed confidence thresholds, original-label preservation, and generated-influence budgets. Their augmented datasets train C=1000 logistic regression.

Direct graph prediction is also evaluated from each LabelSpreading fit. It sums class distributions over training neighbors and uses known-only logistic probabilities only if propagated mass is zero. It is separate from generating labels and fitting LR. The direct graph model uses the training-only graph scaler and does not use augmentation weight budgets.

A small crossover refits the best SVM and GaussianNB parameter settings on the best DBSCAN and LabelSpreading datasets. One adaptive follow-up combines GMM-generated Recruitment labels with DBSCAN-generated Efficacy labels, retaining known-only labels for the other reasons and preserving all source weights. This staged search does not establish a global maximum. All new scores are validation selections, not independently measured Kaggle scores. The reported best submitted Kaggle model remains K=2 without year or Safety boost at 0.87393.

## Artifacts and controls

`datasets/` and `label_audits/` preserve each training table and target/weight mask; `clusters/` records density or graph components. `labeling_models/` and `classifier_models/` save fitted models. `validation_predictions/` preserves probabilities and labels. `comparison.csv`, `new_comparison.csv`, `labeler_comparison.csv`, and `label_counts.csv` retain all results. `settings.json` records grids, selected models, source/data checksums, and software versions.

Controls reproduced the old baseline, preserved all known labels and protected artifacts, excluded noise and unreachable graph nodes from generated targets, matched direct graph predictions to sklearn where defined, reloaded selected models with matching predictions, and refitted selected classifiers from their saved datasets.


## Complementary-label hybrid

The one additional GMM-Recruitment/DBSCAN-Efficacy hybrid scored sample-wise F1 **0.8889** and macro F1 **0.7937**, below both leading individual label-generation approaches. The source-probability controls confirmed that each reason's probabilities matched its selected source classifier. Sample-wise F1 and the nonempty fallback couple the five outputs, so changes that help separately need not help when combined. Retain this negative result; do not use the hybrid as the leading candidate.

## KNN follow-up on the new label sets

The earlier KNN results used older datasets. A separate follow-up tests k=1,3,7,15 on the best DBSCAN and LabelSpreading datasets and the complementary-label hybrid, using the same frozen classifier scaler and distance/sample weights.

| Dataset | Best neighbor count | Sample-wise F1 | Macro F1 | Difference from DBSCAN + LR |
|---|---:|---:|---:|---:|
| dbscan_m5_q25 | 1 | 0.8547 | 0.6690 | -0.0556 |
| graph_n20_a0p2 | 15 | 0.8291 | 0.6159 | -0.0812 |
| hybrid_gmm_dbscan | 3 | 0.8526 | 0.5864 | -0.0577 |

All twelve follow-up configurations are saved in `knn_comparison.csv`. The results only compare these tested settings; earlier losses alone did not establish that KNN could not benefit from new labels.

## Prepared candidate

`submissions/part1_dbscan_m5_q25_logistic_C1000_submission.csv` contains 312 predictions from **dbscan_m5_q25__logistic_C1000**. Validation sample-wise F1: **0.9103**; macro F1: **0.8114**. The file is prepared locally, not submitted. No Kaggle slot was used, and the submitted-best tag is unchanged. The model and file checksums are recorded in `submission_manifest.json`.
