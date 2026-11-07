# Classifier feature additions with frozen DBSCAN pseudo-labels

Reference: sample F1 **0.910256**, macro F1 **0.811421**. Each feature group is added separately to the same 150 embeddings. No clustering, target, or generated-weight changes. The same 157 known-label trials and 133 trials with generated Efficacy labels are used; each reason retains its own targets/weights.

## Training-only frequency inspection

Diseases: 879 distinct normalized names; minimum counts 2/3/5 retain 249/145/71 indicators. Counts are numbers of distinct training trials, not repeated mentions.

Pathways: 29 broad level-0 categories. Count thresholds 1/5/30 retain 29/27/26 categories; each adds a missing-annotation indicator. That indicator is also evaluated alone. Missing annotation does not imply no drug or no biological activity.

Criteria: 104482 unigram/bigram terms, including 67965 occurring in one trial. With a 2,000-term cap, low cutoffs such as 3/5/10 would select nearly identical vocabularies. Minimum document counts 3/50/100 were chosen to test progressively stronger filtering. All criteria vocabularies use maximum DF 95%, sublinear TF, row L2 normalization, and preserved negation words. No stop-word list is used.

## Validation results

| Configuration | Added features | Sample F1 | Macro F1 | Change in sample F1 | Fallback trials |
|---|---:|---:|---:|---:|---:|
| embeddings_only | 0 | 0.910256 | 0.811421 | +0.000000 | 4 |
| diseases_min2 | 249 | 0.910256 | 0.811421 | +0.000000 | 4 |
| diseases_min3 | 145 | 0.910256 | 0.811421 | +0.000000 | 4 |
| diseases_min5 | 71 | 0.910256 | 0.811421 | +0.000000 | 4 |
| pathway_missing_only | 1 | 0.905983 | 0.805106 | -0.004274 | 4 |
| pathways_min1 | 30 | 0.873932 | 0.760207 | -0.036325 | 8 |
| pathways_min5 | 28 | 0.873932 | 0.760207 | -0.036325 | 8 |
| pathways_min30 | 27 | 0.873932 | 0.760207 | -0.036325 | 8 |
| criteria_min3 | 2000 | 0.910256 | 0.811421 | +0.000000 | 4 |
| criteria_min50 | 1533 | 0.897436 | 0.801106 | -0.012821 | 4 |
| criteria_min100 | 776 | 0.893162 | 0.792111 | -0.017094 | 4 |

Selected: **embeddings_only**, sample F1 **0.910256**, macro F1 **0.811421**.

## Controls and interpretation

The embedding scaler stays frozen from the 157 originally labeled trials. New feature vocabularies and IDF are fit only on all 1,171 training trials, using no validation/test documents or labels. Binary features stay 0/1; TF-IDF uses L2 row normalization. New blocks have weight 1. C=1000, normalized fitting-weight total 157, final cutoff 0.5, and highest-probability fallback stay fixed. No feature combinations are evaluated in this round.

All saved sparse feature matrices reproduce their preprocessing. Saved models reload with matching predictions, the embeddings-only control reproduces the original probabilities, and the selected model independently refits from its persisted matrix and targets/weights with matching probabilities. Protected original artifacts retain their hashes.

Performance differences reflect these specific encodings/filtering settings at fixed C and feature-block scale; a negative result does not establish that a raw feature can never help. This is exploratory selection on the same 78 validation trials, not an independent test estimate.

`profiles/` preserves frequency distributions and counts; `vocabularies/` records accepted terms/categories. `datasets/` saves each exact sparse classifier matrix with ordered feature names, and the shared trial IDs/targets/weights. `models/` saves native sklearn preprocessing and estimators; `validation_predictions/` saves per-trial outputs. `comparison.csv`, `best_by_group.csv`, and `settings.json` preserve scores, settings, and checksums.

No Kaggle submission, commit, or tag was made.

No addition improved the primary metric; retain the embeddings-only DBSCAN model as the leader.
