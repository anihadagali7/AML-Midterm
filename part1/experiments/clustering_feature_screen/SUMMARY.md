# Added features in GMM and DBSCAN clustering

Seven clustering representations cross embeddings-only with disease or criteria blocks at relative scales 0.10/0.25/1. GMM uses the original K=10 tied-covariance configuration (7 fits). DBSCAN uses min_samples=5 and epsilon at the 25th/50th/75th percentiles of each representation's training fifth-neighbor distances, including self as in the original implementation (21 fits). Total 28 settings. All fit on the 1,171 training trials only.

## Input preprocessing and controls

Disease input reuses the 71-indicator diseases_min5 encoding. Criteria reuses the existing 2,000-term criteria_min3 TF-IDF vocabulary/IDF, with a new 20-component training-only TruncatedSVD (seed 42, n_iter=7) for manageable GMM covariances. Added blocks are centered using their training means and normalized to match the aggregate training RMS of the original standardized embedding block before applying the relative scale. Scales 0.10/0.25/1 add aggregate variance equal to 1%/6.25%/100% of the embedding variance. They control input geometry and covariance regularization, not literal GMM likelihood feature-importance weights.

Every final classifier still uses only the original 150 embeddings, the same frozen 157-row scaler, C=1000, normalized fitting-weight total 157 per reason, cutoff 0.5, and nonempty fallback. Generated labels/weights can change with clusters, but known labels and the label acceptance/weight-budget rules are fixed. Validation/test never enter preprocessing or clustering. At inference only the original embeddings are needed by the classifier.

Reference sample F1: original GMM **0.905983**, current DBSCAN **0.910256**. Best setting: **dbscan_embeddings_only_q25**, sample F1 **0.910256**, macro F1 **0.811421**.

## Best by representation

| Family | Representation | Radius quantile | Sample F1 | Macro F1 | Generated trials | Change vs leader |
|---|---|---:|---:|---:|---:|---:|
| dbscan | embeddings_only | 0.25 | 0.910256 | 0.811421 | 133 | +0.000000 |
| dbscan | criteria_scale0p25 | 0.5 | 0.910256 | 0.815391 | 469 | +0.000000 |
| dbscan | diseases_scale0p25 | 0.5 | 0.910256 | 0.815391 | 515 | +0.000000 |
| gmm | embeddings_only | — | 0.905983 | 0.811626 | 696 | -0.004274 |
| dbscan | criteria_scale1 | 0.25 | 0.901709 | 0.803884 | 268 | -0.008547 |
| dbscan | diseases_scale0p1 | 0.25 | 0.897436 | 0.803884 | 168 | -0.012821 |
| dbscan | criteria_scale0p1 | 0.25 | 0.893162 | 0.792111 | 171 | -0.017094 |
| gmm | diseases_scale0p1 | — | 0.893162 | 0.807626 | 871 | -0.017094 |
| dbscan | diseases_scale1 | 0.25 | 0.893162 | 0.804324 | 0 | -0.017094 |
| gmm | criteria_scale0p1 | — | 0.888889 | 0.802384 | 876 | -0.021368 |
| gmm | diseases_scale1 | — | 0.884615 | 0.768865 | 449 | -0.025641 |
| gmm | criteria_scale1 | — | 0.871795 | 0.788407 | 637 | -0.038462 |
| gmm | diseases_scale0p25 | — | 0.846154 | 0.741988 | 854 | -0.064103 |
| gmm | criteria_scale0p25 | — | 0.824786 | 0.695511 | 1013 | -0.085470 |

## All configurations

| Family | Representation | Radius quantile | Sample F1 | Macro F1 | Generated trials | Change vs leader |
|---|---|---:|---:|---:|---:|---:|
| gmm | embeddings_only | — | 0.905983 | 0.811626 | 696 | -0.004274 |
| dbscan | embeddings_only | 0.25 | 0.910256 | 0.811421 | 133 | +0.000000 |
| dbscan | embeddings_only | 0.5 | 0.888889 | 0.789590 | 483 | -0.021368 |
| dbscan | embeddings_only | 0.75 | 0.893162 | 0.804324 | 0 | -0.017094 |
| gmm | diseases_scale0p1 | — | 0.893162 | 0.807626 | 871 | -0.017094 |
| gmm | diseases_scale0p25 | — | 0.846154 | 0.741988 | 854 | -0.064103 |
| gmm | diseases_scale1 | — | 0.884615 | 0.768865 | 449 | -0.025641 |
| gmm | criteria_scale0p1 | — | 0.888889 | 0.802384 | 876 | -0.021368 |
| gmm | criteria_scale0p25 | — | 0.824786 | 0.695511 | 1013 | -0.085470 |
| gmm | criteria_scale1 | — | 0.871795 | 0.788407 | 637 | -0.038462 |
| dbscan | diseases_scale0p1 | 0.25 | 0.897436 | 0.803884 | 168 | -0.012821 |
| dbscan | diseases_scale0p1 | 0.5 | 0.888889 | 0.789590 | 479 | -0.021368 |
| dbscan | diseases_scale0p1 | 0.75 | 0.893162 | 0.804324 | 0 | -0.017094 |
| dbscan | diseases_scale0p25 | 0.25 | 0.884615 | 0.793668 | 169 | -0.025641 |
| dbscan | diseases_scale0p25 | 0.5 | 0.910256 | 0.815391 | 515 | +0.000000 |
| dbscan | diseases_scale0p25 | 0.75 | 0.893162 | 0.804324 | 0 | -0.017094 |
| dbscan | diseases_scale1 | 0.25 | 0.893162 | 0.804324 | 0 | -0.017094 |
| dbscan | diseases_scale1 | 0.5 | 0.893162 | 0.804324 | 0 | -0.017094 |
| dbscan | diseases_scale1 | 0.75 | 0.893162 | 0.804324 | 0 | -0.017094 |
| dbscan | criteria_scale0p1 | 0.25 | 0.893162 | 0.792111 | 171 | -0.017094 |
| dbscan | criteria_scale0p1 | 0.5 | 0.888889 | 0.789590 | 481 | -0.021368 |
| dbscan | criteria_scale0p1 | 0.75 | 0.893162 | 0.804324 | 0 | -0.017094 |
| dbscan | criteria_scale0p25 | 0.25 | 0.893162 | 0.792111 | 171 | -0.017094 |
| dbscan | criteria_scale0p25 | 0.5 | 0.910256 | 0.815391 | 469 | +0.000000 |
| dbscan | criteria_scale0p25 | 0.75 | 0.893162 | 0.804324 | 0 | -0.017094 |
| dbscan | criteria_scale1 | 0.25 | 0.901709 | 0.803884 | 268 | -0.008547 |
| dbscan | criteria_scale1 | 0.5 | 0.893162 | 0.804324 | 0 | -0.017094 |
| dbscan | criteria_scale1 | 0.75 | 0.893162 | 0.804324 | 0 | -0.017094 |

## Artifacts and verification

Both original family controls reproduce their targets, weights, and validation probabilities. Every saved labeling packet recreates its exact clustering coordinates and targets/weights; DBSCAN is independently refitted with matching cluster assignments. All saved classifiers reload and rescore. The selected classifier independently refits from its saved dataset with matching probabilities; all classifier estimators have exactly 150 input features. Protected source data and models retain their checksums.

representations/, profiles/, and clustering_inputs/ preserve encoders, SVD, normalization, geometry, training IDs, coordinates, and neighbor distances. labeling_models/, clusters/, clustering_outputs/, datasets/, label_audits/, classifier_models/, and validation_predictions/ preserve all fits and generated labels/weights. comparison.csv includes matched family and matched radius-quantile controls; dbscan_radius_audit.csv documents each recomputed radius. best_by_family_group.csv, best_by_representation.csv, label_counts.csv, and settings.json record results and checksums.

This bounded screen evaluates these encodings, scales, and density settings on the repeatedly consulted 78-trial validation set; it does not exhaust all feature representations or establish independent test improvement. No self-training, final validation-label refit, Kaggle submission, commit, or tag was performed.

No setting beats the current leader; retain the original embeddings-only DBSCAN classifier.

## Follow-up attribution checks

The two DBSCAN scale-0.25, median-radius candidates tie sample F1 0.910256 and increase macro F1 to 0.815391. Both change only trial TR11226 from predicted Efficacy to Recruitment; its actual reason is Other. Thus the macro change reallocates one false positive across classes and does not correct an additional trial. The original embeddings-only model remains the selected reference under the predeclared simpler-model tie preference.

Two adaptive absolute-radius controls remove the added features while retaining those exact radii. Both score sample F1 0.893162 and macro F1 0.804324, so simply using the larger radii on embeddings alone does not reproduce the tied feature models. These two controls are separate from the original 28-setting grid, for 30 fits in total.

See `absolute_radius_controls/` and `part1/clustering_radius_controls.ipynb` for exact code, models, targets, predictions, and results.
