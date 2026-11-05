# PCA before GMM clustering

Four controlled settings: no PCA, PCA20, PCA30, PCA50 before the same tied-covariance K=10 GMM. PCA uses full SVD without whitening and is fitted on the 1,171 training trials only. The classifier always receives all 150 original standardized embedding dimensions.

All GMM hyperparameters, partial-label rules, raw weight budgets, classifier C=1000, fitting-weight total 157, cutoff 0.5, and highest-probability fallback remain fixed. PCA can change generated labels and their computed weights because cluster assignments change, but the rule for accepting/weighting them is unchanged.

| Clustering input | Variance retained | Generated trials | Sample F1 | Macro F1 | Change vs original GMM | Change vs DBSCAN |
|---|---:|---:|---:|---:|---:|---:|
| Original 150 dimensions | 100.00% | 696 | 0.905983 | 0.811626 | +0.000000 | -0.004274 |
| PCA 20 | 69.03% | 927 | 0.876068 | 0.787722 | -0.029915 | -0.034188 |
| PCA 30 | 77.06% | 936 | 0.897436 | 0.807576 | -0.008547 | -0.012821 |
| PCA 50 | 86.20% | 862 | 0.841880 | 0.732313 | -0.064103 | -0.068376 |

Best GMM setting: **gmm_tied_k10_no_pca**, sample F1 **0.905983**, macro F1 **0.811626**. Current DBSCAN reference: sample F1 **0.910256**, macro F1 **0.811421**.

## Generated-label counts

| Setting | Reason | Generated positives | Generated negatives | Positive raw weight total | Negative raw weight total |
|---|---|---:|---:|---:|---:|
| gmm_tied_k10_no_pca | Safety | 0 | 0 | 0.000000 | 0.000000 |
| gmm_tied_k10_no_pca | Funding | 0 | 0 | 0.000000 | 0.000000 |
| gmm_tied_k10_no_pca | Efficacy | 0 | 0 | 0.000000 | 0.000000 |
| gmm_tied_k10_no_pca | Recruitment | 608 | 88 | 39.678571 | 22.000000 |
| gmm_tied_k10_no_pca | Other | 0 | 0 | 0.000000 | 0.000000 |
| gmm_tied_k10_pca20 | Safety | 0 | 0 | 0.000000 | 0.000000 |
| gmm_tied_k10_pca20 | Funding | 0 | 0 | 0.000000 | 0.000000 |
| gmm_tied_k10_pca20 | Efficacy | 78 | 839 | 8.500000 | 70.000000 |
| gmm_tied_k10_pca20 | Recruitment | 706 | 136 | 50.500000 | 28.000000 |
| gmm_tied_k10_pca20 | Other | 0 | 0 | 0.000000 | 0.000000 |
| gmm_tied_k10_pca30 | Safety | 0 | 0 | 0.000000 | 0.000000 |
| gmm_tied_k10_pca30 | Funding | 0 | 0 | 0.000000 | 0.000000 |
| gmm_tied_k10_pca30 | Efficacy | 0 | 0 | 0.000000 | 0.000000 |
| gmm_tied_k10_pca30 | Recruitment | 707 | 229 | 50.500000 | 28.000000 |
| gmm_tied_k10_pca30 | Other | 0 | 0 | 0.000000 | 0.000000 |
| gmm_tied_k10_pca50 | Safety | 0 | 0 | 0.000000 | 0.000000 |
| gmm_tied_k10_pca50 | Funding | 0 | 0 | 0.000000 | 0.000000 |
| gmm_tied_k10_pca50 | Efficacy | 28 | 834 | 7.000000 | 57.647059 |
| gmm_tied_k10_pca50 | Recruitment | 706 | 28 | 12.625000 | 7.000000 |
| gmm_tied_k10_pca50 | Other | 0 | 0 | 0.000000 | 0.000000 |

## Verification and interpretation

The no-PCA fit reproduces the old GMM targets, weights, and validation probabilities. Each saved scaler/PCA/GMM recreates its archived coordinates, responsibilities, and targets/weights. All classifiers reload and rescore; the selected classifier independently refits from its saved dataset with matching predictions. Known labels and weights are preserved; no validation or test trial enters fitting. Original artifacts retain their checksums.

This experiment isolates PCA at K=10 with tied covariance. It does not test Ani's K=30 diagonal GMM, clustering with validation/test features, iterative self-training, or final train+validation refitting. Scores are selections on the repeatedly consulted 78 validation trials, not independent test estimates.

datasets/, label_audits/, clusters/, labeling_models/, classifier_models/, clustering_inputs/, validation_predictions/, comparison.csv, label_counts.csv, pca_variance.csv, and settings.json preserve all models, generated targets/weights, exact inputs, results, and checksums. No Kaggle submission, commit, or tag was made.

No setting improves the primary metric over DBSCAN; keep the current DBSCAN leader.
