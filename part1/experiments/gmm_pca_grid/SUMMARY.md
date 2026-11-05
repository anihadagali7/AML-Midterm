# PCA30 GMM cluster-count and covariance sweep

Six settings cross K=10/20/30 with tied/diagonal covariance. The exact previously fitted training-only PCA30 and clustering scaler are reused. All other GMM parameters and label-generation rules stay fixed. The classifier continues to use all 150 original embeddings, its frozen scaler, C=1000, normalized weight total 157 per reason, cutoff 0.5, and highest-probability fallback.

References: previous PCA30 tied/K10 sample F1 **0.897436**; original no-PCA GMM **0.905983**; current DBSCAN leader **0.910256**.

| Clusters | Covariance | Generated trials | Sample F1 | Macro F1 | Change vs original GMM | Change vs DBSCAN |
|---:|---|---:|---:|---:|---:|---:|
| 10 | tied | 936 | 0.897436 | 0.807576 | -0.008547 | -0.012821 |
| 20 | tied | 804 | 0.841880 | 0.708348 | -0.064103 | -0.068376 |
| 30 | tied | 647 | 0.893162 | 0.780069 | -0.012821 | -0.017094 |
| 10 | diag | 687 | 0.871795 | 0.768873 | -0.034188 | -0.038462 |
| 20 | diag | 812 | 0.822650 | 0.699040 | -0.083333 | -0.087607 |
| 30 | diag | 703 | 0.833333 | 0.707118 | -0.072650 | -0.076923 |

Best grid setting: **gmm_pca30_tied_k10**, sample F1 **0.897436**, macro F1 **0.807576**.

## Generated-label counts

| Setting | Reason | Generated positives | Generated negatives | Positive raw weight total | Negative raw weight total |
|---|---|---:|---:|---:|---:|
| gmm_pca30_tied_k10 | Safety | 0 | 0 | 0.000000 | 0.000000 |
| gmm_pca30_tied_k10 | Funding | 0 | 0 | 0.000000 | 0.000000 |
| gmm_pca30_tied_k10 | Efficacy | 0 | 0 | 0.000000 | 0.000000 |
| gmm_pca30_tied_k10 | Recruitment | 707 | 229 | 50.500000 | 28.000000 |
| gmm_pca30_tied_k10 | Other | 0 | 0 | 0.000000 | 0.000000 |
| gmm_pca30_tied_k20 | Safety | 55 | 731 | 8.000000 | 70.500000 |
| gmm_pca30_tied_k20 | Funding | 0 | 0 | 0.000000 | 0.000000 |
| gmm_pca30_tied_k20 | Efficacy | 17 | 650 | 4.250000 | 35.000000 |
| gmm_pca30_tied_k20 | Recruitment | 0 | 0 | 0.000000 | 0.000000 |
| gmm_pca30_tied_k20 | Other | 0 | 0 | 0.000000 | 0.000000 |
| gmm_pca30_tied_k30 | Safety | 58 | 589 | 8.000000 | 70.500000 |
| gmm_pca30_tied_k30 | Funding | 0 | 0 | 0.000000 | 0.000000 |
| gmm_pca30_tied_k30 | Efficacy | 66 | 530 | 8.500000 | 70.000000 |
| gmm_pca30_tied_k30 | Recruitment | 0 | 0 | 0.000000 | 0.000000 |
| gmm_pca30_tied_k30 | Other | 8 | 433 | 2.000000 | 10.560000 |
| gmm_pca30_diag_k10 | Safety | 0 | 0 | 0.000000 | 0.000000 |
| gmm_pca30_diag_k10 | Funding | 0 | 0 | 0.000000 | 0.000000 |
| gmm_pca30_diag_k10 | Efficacy | 18 | 669 | 4.500000 | 37.058824 |
| gmm_pca30_diag_k10 | Recruitment | 0 | 0 | 0.000000 | 0.000000 |
| gmm_pca30_diag_k10 | Other | 0 | 0 | 0.000000 | 0.000000 |
| gmm_pca30_diag_k20 | Safety | 9 | 695 | 2.250000 | 19.828125 |
| gmm_pca30_diag_k20 | Funding | 0 | 0 | 0.000000 | 0.000000 |
| gmm_pca30_diag_k20 | Efficacy | 34 | 773 | 8.500000 | 70.000000 |
| gmm_pca30_diag_k20 | Recruitment | 656 | 39 | 17.584821 | 9.750000 |
| gmm_pca30_diag_k20 | Other | 0 | 0 | 0.000000 | 0.000000 |
| gmm_pca30_diag_k30 | Safety | 9 | 568 | 2.250000 | 19.828125 |
| gmm_pca30_diag_k30 | Funding | 0 | 0 | 0.000000 | 0.000000 |
| gmm_pca30_diag_k30 | Efficacy | 20 | 680 | 5.000000 | 41.176471 |
| gmm_pca30_diag_k30 | Recruitment | 549 | 40 | 18.035714 | 10.000000 |
| gmm_pca30_diag_k30 | Other | 0 | 0 | 0.000000 | 0.000000 |

## Verification and interpretation

The PCA30 tied/K10 control reproduces its prior targets, weights, and validation probabilities. All persisted scaler/PCA/GMM packets replay their exact coordinates, responsibilities, and generated labels/weights. All classifiers reload and rescore, and the selected classifier independently refits from its saved dataset with matching probabilities. Known labels and raw weights are preserved. Validation/test never enter fitting. Protected original artifacts retain their checksums.

The larger diagonal GMM is inspired by Ani's setup but retains our training-only preprocessing, seed, covariance regularization, label-generation rules, and classifier. It is not Ani's full transductive self-training ensemble. The same repeatedly consulted 78 validation trials are used for selection; no independent test improvement is established.

Exact generated datasets, label audits, cluster statistics, native labeling/classifier models, training coordinates and responsibilities, validation predictions, comparison.csv, label_counts.csv, and settings.json are saved locally. No new features, final validation-label refit, Kaggle submission, commit, or tag was used.

No grid setting improves the current DBSCAN leader; keep the existing leader.
