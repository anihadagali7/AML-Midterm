# Combined classifier features with fixed DBSCAN pseudo-labels

Four combinations of diseases, criteria, and broad pathways are tested with logistic regression at C=100 and C=1000, with an embeddings-only control at each C: ten configurations. Each matrix includes the 150 embeddings exactly once.

Frozen representations: diseases_min5 (71 binary columns), criteria_min3 (2,000 TF-IDF columns), and pathways_min30 (26 broad pathway columns plus one missing-annotation flag). Disease and pathway choices are the smallest tied representations from the initial feature comparison; criteria_min3 is the strongest previous text representation. Every feature-block weight is 1. No vocabularies, IDF, scalers, targets, or pseudo-label weights are refitted. Cutoff 0.5, highest-probability fallback, and fitting-weight total 157 per reason remain fixed.

Reference sample F1 **0.910256**, macro F1 **0.811421**. Selected **embeddings_only_weight1_C1000**: sample F1 **0.910256**, macro F1 **0.811421**.

## Results

| Representation | C | Added features | Sample F1 | Macro F1 | Change vs leader | Change vs same-C embeddings |
|---|---:|---:|---:|---:|---:|---:|
| embeddings_only | 100 | 0 | 0.888889 | 0.790092 | -0.021368 | +0.000000 |
| embeddings_only | 1000 | 0 | 0.910256 | 0.811421 | +0.000000 | +0.000000 |
| diseases_criteria | 100 | 2071 | 0.888889 | 0.797110 | -0.021368 | -0.000000 |
| diseases_criteria | 1000 | 2071 | 0.893162 | 0.794790 | -0.017094 | -0.017094 |
| diseases_pathways | 100 | 98 | 0.871795 | 0.772471 | -0.038462 | -0.017094 |
| diseases_pathways | 1000 | 98 | 0.865385 | 0.745498 | -0.044872 | -0.044872 |
| criteria_pathways | 100 | 2027 | 0.863248 | 0.764743 | -0.047009 | -0.025641 |
| criteria_pathways | 1000 | 2027 | 0.865385 | 0.745498 | -0.044872 | -0.044872 |
| diseases_criteria_pathways | 100 | 2098 | 0.854701 | 0.746244 | -0.055556 | -0.034188 |
| diseases_criteria_pathways | 1000 | 2098 | 0.873932 | 0.760207 | -0.036325 | -0.036325 |

## Verification and interpretation

Every saved model reloads with matching probabilities, and its preprocessing exactly reproduces its persisted combined matrix. All ten saved prediction sets independently reproduce their sample and macro F1. The original embeddings-only leader reproduces its previous probabilities, and the selected model independently refits from the saved matrix and shared targets/weights with matching probabilities. Protected source artifacts retain their hashes.

Matched-C controls separate feature-combination changes from changing regularization. Logistic regression adds feature contributions; these combinations introduce no explicit interaction terms. The same repeatedly consulted 78 validation trials are used for selection, so the results do not establish independent test improvement.

datasets/ preserves each exact combined sparse matrix, ordered columns, and shared trial IDs/targets/weights. models/ persists all native sklearn preprocessing components and fitted classifiers, so these same combined inputs can be reused in a subsequent nonlinear-model experiment. comparison.csv, best_by_representation.csv, embedding_controls.csv, settings.json, and validation_predictions/ preserve results and checksums.

No nonlinear models, Kaggle submissions, commits, or tags were used in this round.

No configuration improves the primary validation metric; retain the current embeddings-only leader.
