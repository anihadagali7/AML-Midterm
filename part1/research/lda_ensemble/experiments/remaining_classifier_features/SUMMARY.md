# Remaining classifier feature screen

30 configurations: five categorical feature fields at minimum training counts 3/5 and C=100/1000, plus standardized year, molecular descriptors, two missing-annotation controls, and embeddings-only at both C settings. Each group is added independently to the same 150 embeddings with fixed DBSCAN targets and raw weights.

Reference sample F1 **0.910256**, macro F1 **0.811421**. Selected **embeddings_only_C1000**: sample F1 **0.910256**, macro F1 **0.811421**.

## Best by feature group

| Representation | C | Added features | Sample F1 | Macro F1 | Change vs leader | Change vs same-C embeddings |
|---|---:|---:|---:|---:|---:|---:|
| embeddings_only | 1000 | 0 | 0.910256 | 0.811421 | +0.000000 | +0.000000 |
| matched_drugs_min5 | 1000 | 51 | 0.910256 | 0.811421 | +0.000000 | +0.000000 |
| drug_actions_missing_only | 1000 | 1 | 0.905983 | 0.805106 | -0.004274 | -0.004274 |
| drug_targets_min5 | 1000 | 170 | 0.901709 | 0.803884 | -0.008547 | -0.008547 |
| drugs_min5 | 1000 | 82 | 0.897436 | 0.803884 | -0.012821 | -0.012821 |
| icdcodes_min5 | 1000 | 186 | 0.895299 | 0.800687 | -0.014957 | -0.014957 |
| drug_actions_min5 | 100 | 18 | 0.888889 | 0.800905 | -0.021368 | -0.000000 |
| smiles | 1000 | 19 | 0.884615 | 0.783190 | -0.025641 | -0.025641 |
| year | 100 | 1 | 0.826923 | 0.710959 | -0.083333 | -0.061966 |

## Preprocessing

Category counts measure distinct training trials. Drug names/actions are lowercased with normalized whitespace; protein target symbols are uppercased; ICD codes are safely unpacked from nested lists and reduced to three-character category prefixes. Categories below the selected training frequency are omitted. Multi-hot columns remain 0/1. Each matched-drug, target, and action block includes its annotation-missing flag. The matched-drug missing indicator covers the same training rows as missing targets/SMILES; actions have a slightly different pattern. Both patterns are tested alone so missingness can be separated from biological content.

Eight RDKit molecular descriptors (molecular weight, logP, polar surface area, hydrogen donors/acceptors, rotatable bonds, ring count, fraction sp3 carbons) are aggregated by mean and maximum over valid distinct matched SMILES. Valid-structure count is added; absent descriptors are imputed with training medians. These 17 numeric columns are standardized using all training trials; no-valid-structure and invalid-structure flags stay binary. No fingerprints or chemical text bag-of-words is used in this initial screen. Parsing failures are saved in the molecule audit. Start year is standardized on all 1,171 training trials; the original embedding scaler remains frozen from 157 known-label trials.

All new preprocessing uses training rows only. Validation/test never supplies vocabulary, imputation, scaling, or fit labels. The classifier keeps normalized fitting weight 157 per reason, final cutoff 0.5, highest-probability fallback, and block weight 1.

## All configurations

| Representation | C | Added features | Sample F1 | Macro F1 | Change vs leader | Change vs same-C embeddings |
|---|---:|---:|---:|---:|---:|---:|
| embeddings_only | 100 | 0 | 0.888889 | 0.790092 | -0.021368 | +0.000000 |
| embeddings_only | 1000 | 0 | 0.910256 | 0.811421 | +0.000000 | +0.000000 |
| drug_actions_min3 | 100 | 22 | 0.888889 | 0.800905 | -0.021368 | -0.000000 |
| drug_actions_min3 | 1000 | 22 | 0.880342 | 0.778485 | -0.029915 | -0.029915 |
| drug_actions_min5 | 100 | 18 | 0.888889 | 0.800905 | -0.021368 | -0.000000 |
| drug_actions_min5 | 1000 | 18 | 0.880342 | 0.778485 | -0.029915 | -0.029915 |
| icdcodes_min3 | 100 | 266 | 0.893162 | 0.797110 | -0.017094 | +0.004274 |
| icdcodes_min3 | 1000 | 266 | 0.891026 | 0.793319 | -0.019231 | -0.019231 |
| icdcodes_min5 | 100 | 186 | 0.893162 | 0.797110 | -0.017094 | +0.004274 |
| icdcodes_min5 | 1000 | 186 | 0.895299 | 0.800687 | -0.014957 | -0.014957 |
| drug_targets_min3 | 100 | 231 | 0.897436 | 0.811145 | -0.012821 | +0.008547 |
| drug_targets_min3 | 1000 | 231 | 0.901709 | 0.803884 | -0.008547 | -0.008547 |
| drug_targets_min5 | 100 | 170 | 0.893162 | 0.757811 | -0.017094 | +0.004274 |
| drug_targets_min5 | 1000 | 170 | 0.901709 | 0.803884 | -0.008547 | -0.008547 |
| drugs_min3 | 100 | 124 | 0.893162 | 0.792111 | -0.017094 | +0.004274 |
| drugs_min3 | 1000 | 124 | 0.897436 | 0.803884 | -0.012821 | -0.012821 |
| drugs_min5 | 100 | 82 | 0.893162 | 0.792111 | -0.017094 | +0.004274 |
| drugs_min5 | 1000 | 82 | 0.897436 | 0.803884 | -0.012821 | -0.012821 |
| matched_drugs_min3 | 100 | 71 | 0.888889 | 0.750443 | -0.021368 | -0.000000 |
| matched_drugs_min3 | 1000 | 71 | 0.910256 | 0.811421 | +0.000000 | +0.000000 |
| matched_drugs_min5 | 100 | 51 | 0.888889 | 0.750443 | -0.021368 | -0.000000 |
| matched_drugs_min5 | 1000 | 51 | 0.910256 | 0.811421 | +0.000000 | +0.000000 |
| matched_drugs_missing_only | 100 | 1 | 0.888889 | 0.750443 | -0.021368 | -0.000000 |
| matched_drugs_missing_only | 1000 | 1 | 0.905983 | 0.805106 | -0.004274 | -0.004274 |
| drug_actions_missing_only | 100 | 1 | 0.888889 | 0.750443 | -0.021368 | -0.000000 |
| drug_actions_missing_only | 1000 | 1 | 0.905983 | 0.805106 | -0.004274 | -0.004274 |
| year | 100 | 1 | 0.826923 | 0.710959 | -0.083333 | -0.061966 |
| year | 1000 | 1 | 0.750000 | 0.585077 | -0.160256 | -0.160256 |
| smiles | 100 | 19 | 0.884615 | 0.783430 | -0.025641 | -0.004274 |
| smiles | 1000 | 19 | 0.884615 | 0.783190 | -0.025641 | -0.025641 |

## Verification and limits

Every saved model reloads with matching probabilities; all label sets rescore to the recorded sample and macro F1. Each model recreates its exact saved sparse matrix. The C=1000 embedding control reproduces the original leader, and the selected model independently refits from its persisted matrix and targets/weights with matching probabilities. Protected sources retain their checksums.

This is a bounded screen of specific encodings on the repeatedly consulted 78-trial validation set. It does not establish independent test improvement or exhaust all encodings, feature combinations, fingerprints, or regularization settings.

profiles/ and vocabularies/ preserve training frequency distributions, filters, and molecular parsing audits; datasets/ saves exact matrices, ordered feature names, and shared trial IDs/targets/weights; models/ and validation_predictions/ preserve all fits and outputs. comparison.csv, best_by_group.csv, embedding_controls.csv, and settings.json preserve results, settings, and checksums. RDKit was installed in the workspace virtual environment.

No Kaggle submission, commit, or tag was made.

No configuration improves the primary validation metric; retain the current embeddings-only leader.
