# Regularization and added-feature weighting

Grid: C=1,10,100,1000; added-feature multipliers=0.25,1,4. Six existing representations (three diseases, three criteria) each have 12 settings, plus four embeddings-only controls: 76 configurations. No pathway or feature-combination tests in this round.

All embeddings, vocabularies, IDF, generated targets and weights are frozen from the previous experiment. Only classifier C and the added-column multiplier vary. Smaller C means stronger regularization. The multiplier changes relative regularization cost, not a prescribed multiplier of the feature's prediction effect. Cutoff 0.5, highest-probability fallback, and fitting-weight total 157 per reason stay fixed.

Reference sample F1 **0.910256**, macro F1 **0.811421**. Selected **embeddings_only_weight1_C1000**: sample F1 **0.910256**, macro F1 **0.811421**.

## Best by representation

| Representation | C | Block weight | Added features | Sample F1 | Macro F1 | Change vs leader | Change vs same-C embeddings |
|---|---:|---:|---:|---:|---:|---:|---:|
| embeddings_only | 1000 | 1 | 0 | 0.910256 | 0.811421 | +0.000000 | +0.000000 |
| diseases_min5 | 1000 | 1 | 71 | 0.910256 | 0.811421 | +0.000000 | +0.000000 |
| diseases_min3 | 1000 | 1 | 145 | 0.910256 | 0.811421 | +0.000000 | +0.000000 |
| diseases_min2 | 1000 | 1 | 249 | 0.910256 | 0.811421 | +0.000000 | +0.000000 |
| criteria_min3 | 1000 | 1 | 2000 | 0.910256 | 0.811421 | +0.000000 | +0.000000 |
| criteria_min100 | 1000 | 0.25 | 776 | 0.897436 | 0.803884 | -0.012821 | -0.012821 |
| criteria_min50 | 1000 | 1 | 1533 | 0.897436 | 0.801106 | -0.012821 | -0.012821 |

## Matched embeddings-only controls

| Representation | C | Block weight | Added features | Sample F1 | Macro F1 | Change vs leader | Change vs same-C embeddings |
|---|---:|---:|---:|---:|---:|---:|---:|
| embeddings_only | 1 | 1 | 0 | 0.858974 | 0.695743 | -0.051282 | +0.000000 |
| embeddings_only | 10 | 1 | 0 | 0.867521 | 0.718678 | -0.042735 | +0.000000 |
| embeddings_only | 100 | 1 | 0 | 0.888889 | 0.790092 | -0.021368 | +0.000000 |
| embeddings_only | 1000 | 1 | 0 | 0.910256 | 0.811421 | +0.000000 | +0.000000 |

## All configurations

| Representation | C | Block weight | Added features | Sample F1 | Macro F1 | Change vs leader | Change vs same-C embeddings |
|---|---:|---:|---:|---:|---:|---:|---:|
| embeddings_only | 1000 | 1 | 0 | 0.910256 | 0.811421 | +0.000000 | +0.000000 |
| diseases_min5 | 1000 | 1 | 71 | 0.910256 | 0.811421 | +0.000000 | +0.000000 |
| diseases_min3 | 1000 | 1 | 145 | 0.910256 | 0.811421 | +0.000000 | +0.000000 |
| diseases_min2 | 1000 | 1 | 249 | 0.910256 | 0.811421 | +0.000000 | +0.000000 |
| criteria_min3 | 1000 | 1 | 2000 | 0.910256 | 0.811421 | +0.000000 | +0.000000 |
| diseases_min5 | 1000 | 0.25 | 71 | 0.897436 | 0.803884 | -0.012821 | -0.012821 |
| diseases_min3 | 1000 | 0.25 | 145 | 0.897436 | 0.803884 | -0.012821 | -0.012821 |
| diseases_min2 | 1000 | 0.25 | 249 | 0.897436 | 0.803884 | -0.012821 | -0.012821 |
| criteria_min100 | 1000 | 0.25 | 776 | 0.897436 | 0.803884 | -0.012821 | -0.012821 |
| criteria_min50 | 1000 | 1 | 1533 | 0.897436 | 0.801106 | -0.012821 | -0.012821 |
| criteria_min50 | 1000 | 0.25 | 1533 | 0.897436 | 0.803884 | -0.012821 | -0.012821 |
| criteria_min3 | 1000 | 0.25 | 2000 | 0.897436 | 0.803884 | -0.012821 | -0.012821 |
| diseases_min5 | 100 | 0.25 | 71 | 0.893162 | 0.803776 | -0.017094 | +0.004274 |
| diseases_min3 | 100 | 0.25 | 145 | 0.893162 | 0.803776 | -0.017094 | +0.004274 |
| diseases_min2 | 100 | 0.25 | 249 | 0.893162 | 0.803776 | -0.017094 | +0.004274 |
| criteria_min100 | 1000 | 1 | 776 | 0.893162 | 0.792111 | -0.017094 | -0.017094 |
| criteria_min100 | 100 | 0.25 | 776 | 0.893162 | 0.795905 | -0.017094 | +0.004274 |
| criteria_min100 | 100 | 4 | 776 | 0.893162 | 0.783442 | -0.017094 | +0.004274 |
| criteria_min3 | 100 | 0.25 | 2000 | 0.893162 | 0.807572 | -0.017094 | +0.004274 |
| criteria_min3 | 1000 | 4 | 2000 | 0.893162 | 0.792111 | -0.017094 | -0.017094 |
| diseases_min3 | 1000 | 4 | 145 | 0.891026 | 0.792957 | -0.019231 | -0.019231 |
| diseases_min2 | 1000 | 4 | 249 | 0.891026 | 0.792957 | -0.019231 | -0.019231 |
| embeddings_only | 100 | 1 | 0 | 0.888889 | 0.790092 | -0.021368 | +0.000000 |
| diseases_min2 | 100 | 1 | 249 | 0.888889 | 0.790092 | -0.021368 | +0.000000 |
| criteria_min100 | 100 | 1 | 776 | 0.888889 | 0.790092 | -0.021368 | +0.000000 |
| criteria_min50 | 100 | 1 | 1533 | 0.888889 | 0.790092 | -0.021368 | +0.000000 |
| criteria_min50 | 100 | 0.25 | 1533 | 0.888889 | 0.793887 | -0.021368 | +0.000000 |
| criteria_min3 | 100 | 1 | 2000 | 0.888889 | 0.790092 | -0.021368 | +0.000000 |
| criteria_min50 | 1000 | 4 | 1533 | 0.888889 | 0.776906 | -0.021368 | -0.021368 |
| criteria_min50 | 100 | 4 | 1533 | 0.888889 | 0.776906 | -0.021368 | -0.000000 |
| diseases_min5 | 1000 | 4 | 71 | 0.886752 | 0.783962 | -0.023504 | -0.023504 |
| diseases_min5 | 100 | 1 | 71 | 0.884615 | 0.783776 | -0.025641 | -0.004274 |
| diseases_min3 | 100 | 1 | 145 | 0.884615 | 0.783776 | -0.025641 | -0.004274 |
| diseases_min2 | 100 | 4 | 249 | 0.884615 | 0.776857 | -0.025641 | -0.004274 |
| criteria_min100 | 1000 | 4 | 776 | 0.884615 | 0.778834 | -0.025641 | -0.025641 |
| diseases_min5 | 100 | 4 | 71 | 0.882479 | 0.766563 | -0.027778 | -0.006410 |
| criteria_min3 | 100 | 4 | 2000 | 0.880342 | 0.776109 | -0.029915 | -0.008547 |
| diseases_min3 | 100 | 4 | 145 | 0.871795 | 0.760854 | -0.038462 | -0.017094 |
| criteria_min100 | 1 | 4 | 776 | 0.867521 | 0.706804 | -0.042735 | +0.008547 |
| embeddings_only | 10 | 1 | 0 | 0.867521 | 0.718678 | -0.042735 | +0.000000 |
| diseases_min5 | 10 | 1 | 71 | 0.867521 | 0.718678 | -0.042735 | +0.000000 |
| diseases_min5 | 10 | 0.25 | 71 | 0.867521 | 0.718678 | -0.042735 | +0.000000 |
| diseases_min3 | 10 | 1 | 145 | 0.867521 | 0.718678 | -0.042735 | +0.000000 |
| diseases_min3 | 10 | 0.25 | 145 | 0.867521 | 0.718678 | -0.042735 | +0.000000 |
| diseases_min2 | 10 | 1 | 249 | 0.867521 | 0.718678 | -0.042735 | +0.000000 |
| diseases_min2 | 10 | 0.25 | 249 | 0.867521 | 0.718678 | -0.042735 | +0.000000 |
| criteria_min100 | 10 | 1 | 776 | 0.867521 | 0.718678 | -0.042735 | +0.000000 |
| criteria_min100 | 10 | 0.25 | 776 | 0.867521 | 0.718678 | -0.042735 | +0.000000 |
| criteria_min50 | 10 | 1 | 1533 | 0.867521 | 0.718678 | -0.042735 | +0.000000 |
| criteria_min50 | 10 | 0.25 | 1533 | 0.867521 | 0.718678 | -0.042735 | +0.000000 |
| criteria_min3 | 10 | 1 | 2000 | 0.867521 | 0.718678 | -0.042735 | +0.000000 |
| criteria_min3 | 10 | 0.25 | 2000 | 0.867521 | 0.718678 | -0.042735 | +0.000000 |
| diseases_min5 | 10 | 4 | 71 | 0.863248 | 0.695299 | -0.047009 | -0.004274 |
| criteria_min100 | 10 | 4 | 776 | 0.863248 | 0.713725 | -0.047009 | -0.004274 |
| criteria_min50 | 10 | 4 | 1533 | 0.863248 | 0.713725 | -0.047009 | -0.004274 |
| criteria_min3 | 10 | 4 | 2000 | 0.863248 | 0.713725 | -0.047009 | -0.004274 |
| criteria_min50 | 1 | 4 | 1533 | 0.863248 | 0.701540 | -0.047009 | +0.004274 |
| criteria_min3 | 1 | 4 | 2000 | 0.863248 | 0.701540 | -0.047009 | +0.004274 |
| embeddings_only | 1 | 1 | 0 | 0.858974 | 0.695743 | -0.051282 | +0.000000 |
| diseases_min5 | 1 | 1 | 71 | 0.858974 | 0.695743 | -0.051282 | +0.000000 |
| diseases_min5 | 1 | 0.25 | 71 | 0.858974 | 0.695743 | -0.051282 | +0.000000 |
| diseases_min3 | 1 | 1 | 145 | 0.858974 | 0.695743 | -0.051282 | +0.000000 |
| diseases_min3 | 1 | 0.25 | 145 | 0.858974 | 0.695743 | -0.051282 | +0.000000 |
| diseases_min2 | 1 | 1 | 249 | 0.858974 | 0.695743 | -0.051282 | +0.000000 |
| diseases_min2 | 1 | 0.25 | 249 | 0.858974 | 0.695743 | -0.051282 | +0.000000 |
| criteria_min100 | 1 | 1 | 776 | 0.858974 | 0.695743 | -0.051282 | +0.000000 |
| criteria_min100 | 1 | 0.25 | 776 | 0.858974 | 0.695743 | -0.051282 | +0.000000 |
| criteria_min50 | 1 | 1 | 1533 | 0.858974 | 0.695743 | -0.051282 | +0.000000 |
| criteria_min50 | 1 | 0.25 | 1533 | 0.858974 | 0.695743 | -0.051282 | +0.000000 |
| criteria_min3 | 1 | 1 | 2000 | 0.858974 | 0.695743 | -0.051282 | +0.000000 |
| criteria_min3 | 1 | 0.25 | 2000 | 0.858974 | 0.695743 | -0.051282 | +0.000000 |
| diseases_min5 | 1 | 4 | 71 | 0.854701 | 0.684632 | -0.055556 | -0.004274 |
| diseases_min3 | 10 | 4 | 145 | 0.854701 | 0.684632 | -0.055556 | -0.012821 |
| diseases_min3 | 1 | 4 | 145 | 0.854701 | 0.684632 | -0.055556 | -0.004274 |
| diseases_min2 | 10 | 4 | 249 | 0.854701 | 0.684632 | -0.055556 | -0.012821 |
| diseases_min2 | 1 | 4 | 249 | 0.854701 | 0.684632 | -0.055556 | -0.004274 |

## Verification and interpretation

All 76 saved prediction sets were reloaded and rescored. Every C=1000, weight=1 setting reproduces its previous probabilities. The winner's preprocessing recreates its exact saved feature matrix, and independent refitting from that matrix and the fixed targets/weights gives matching validation probabilities. Protected earlier artifacts retain their hashes.

For every added-feature setting, delta_vs_same_C_embeddings distinguishes an improvement due to added information from one due to changing regularization alone. Selection uses the same repeatedly consulted 78 validation trials; it does not establish independent test improvement. Models remain alternatives using one new feature group at a time.

Exact weighted sparse matrices and ordered feature names are saved once per representation/weight in datasets/, with the shared trial IDs, targets, and raw weights. Each model references its matrix and persists the original native sklearn preprocessing plus the new block weight. comparison.csv, best_by_representation.csv, best_by_group.csv, embedding_controls.csv, and settings.json preserve results and checksums.

No Kaggle submission, commit, or tag was made.

No configuration improves the primary validation metric; retain the current embeddings-only leader.

## Matched-control interpretation

| C | Embeddings only | Best disease addition | Best criteria addition |
|---:|---:|---:|---:|
| 1 | 0.858974 | 0.858974 | 0.867521 |
| 10 | 0.867521 | 0.867521 | 0.867521 |
| 100 | 0.888889 | 0.893162 | 0.893162 |
| 1000 | 0.910256 | 0.910256 | 0.910256 |

Nine added-feature configurations improve the embeddings-only control at the same C. The largest matched-control gain is criteria_min100 at C=1 and block weight=4: sample F1 0.867521 versus 0.858974 (+0.008547). At C=100, some disease and criteria additions improve 0.888889 to 0.893162. All remain below the existing C=1000 embeddings-only leader (0.910256). Thus these results do show modest gains over some more strongly regularized controls, but no improvement over the best current model.
