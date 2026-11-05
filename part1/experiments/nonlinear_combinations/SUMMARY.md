# RBF SVM on frozen feature combinations

45 settings: C=1,10,100; gamma=0.001,scale,0.03; embeddings alone and all four saved pair/triple combinations. Inputs, vocabularies, IDF, embedding scaler, DBSCAN targets, and raw weights are copied unchanged from the logistic experiment. Each reason uses normalized fitting-weight total 157. Embeddings-only uses dense input as in the earlier SVM check; combinations use sparse matrices.

Probabilities use sigmoid calibration with three stratified training-only folds (seed 42, ensemble=False). The final estimator is trained on all included training rows for each reason. Calibration audit records support-vector counts, resolved gamma, training-row counts, and hashes of each held-out training-fold row mapping. The 78 external validation trials never enter fitting or calibration. Final cutoff=0.5, highest-probability fallback enabled.

Current logistic leader: sample F1 **0.910256**, macro F1 **0.811421**. Best SVM: **embeddings_only_rbf_C100_gamma0p001**, sample F1 **0.867521**, macro F1 **0.651287**.

## Best SVM by representation

| Representation | C | Gamma | Sample F1 | Macro F1 | Change vs logistic leader | Change vs matched embedding SVM |
|---|---:|---|---:|---:|---:|---:|
| embeddings_only | 100 | 0.001 | 0.867521 | 0.651287 | -0.042735 | +0.000000 |
| diseases_pathways | 1 | scale | 0.858974 | 0.712807 | -0.051282 | +0.000000 |
| criteria_pathways | 1 | scale | 0.858974 | 0.712807 | -0.051282 | +0.000000 |
| diseases_criteria | 1 | scale | 0.858974 | 0.712807 | -0.051282 | +0.000000 |
| diseases_criteria_pathways | 1 | scale | 0.858974 | 0.712807 | -0.051282 | +0.000000 |

## All configurations

| Representation | C | Gamma | Sample F1 | Macro F1 | Change vs logistic leader | Change vs matched embedding SVM |
|---|---:|---|---:|---:|---:|---:|
| embeddings_only | 100 | 0.001 | 0.867521 | 0.651287 | -0.042735 | +0.000000 |
| embeddings_only | 1 | scale | 0.858974 | 0.712807 | -0.051282 | +0.000000 |
| diseases_pathways | 1 | scale | 0.858974 | 0.712807 | -0.051282 | +0.000000 |
| criteria_pathways | 1 | scale | 0.858974 | 0.712807 | -0.051282 | +0.000000 |
| diseases_criteria | 1 | scale | 0.858974 | 0.712807 | -0.051282 | +0.000000 |
| diseases_criteria_pathways | 1 | scale | 0.858974 | 0.712807 | -0.051282 | +0.000000 |
| embeddings_only | 10 | 0.001 | 0.854701 | 0.615299 | -0.055556 | +0.000000 |
| diseases_pathways | 100 | 0.001 | 0.854701 | 0.629655 | -0.055556 | -0.012821 |
| criteria_pathways | 100 | 0.001 | 0.854701 | 0.629655 | -0.055556 | -0.012821 |
| diseases_criteria | 10 | 0.001 | 0.854701 | 0.615299 | -0.055556 | +0.000000 |
| diseases_criteria | 100 | 0.001 | 0.854701 | 0.630753 | -0.055556 | -0.012821 |
| diseases_criteria_pathways | 100 | 0.001 | 0.854701 | 0.629655 | -0.055556 | -0.012821 |
| embeddings_only | 1 | 0.001 | 0.850427 | 0.693129 | -0.059829 | +0.000000 |
| diseases_pathways | 1 | 0.001 | 0.850427 | 0.693129 | -0.059829 | +0.000000 |
| criteria_pathways | 1 | 0.001 | 0.850427 | 0.693129 | -0.059829 | +0.000000 |
| diseases_criteria | 1 | 0.001 | 0.850427 | 0.693129 | -0.059829 | +0.000000 |
| diseases_criteria_pathways | 1 | 0.001 | 0.850427 | 0.693129 | -0.059829 | +0.000000 |
| diseases_pathways | 10 | scale | 0.841880 | 0.680314 | -0.068376 | +0.004274 |
| diseases_pathways | 100 | scale | 0.841880 | 0.680314 | -0.068376 | +0.004274 |
| criteria_pathways | 10 | scale | 0.841880 | 0.680314 | -0.068376 | +0.004274 |
| criteria_pathways | 100 | scale | 0.841880 | 0.680314 | -0.068376 | +0.004274 |
| diseases_criteria_pathways | 10 | scale | 0.841880 | 0.680314 | -0.068376 | +0.004274 |
| diseases_criteria_pathways | 100 | scale | 0.841880 | 0.680314 | -0.068376 | +0.004274 |
| embeddings_only | 10 | scale | 0.837607 | 0.665577 | -0.072650 | +0.000000 |
| embeddings_only | 100 | scale | 0.837607 | 0.665577 | -0.072650 | +0.000000 |
| diseases_pathways | 10 | 0.001 | 0.837607 | 0.595518 | -0.072650 | -0.017094 |
| criteria_pathways | 10 | 0.001 | 0.837607 | 0.595518 | -0.072650 | -0.017094 |
| diseases_criteria | 10 | scale | 0.837607 | 0.665577 | -0.072650 | +0.000000 |
| diseases_criteria | 100 | scale | 0.837607 | 0.665577 | -0.072650 | +0.000000 |
| diseases_criteria_pathways | 10 | 0.001 | 0.837607 | 0.595518 | -0.072650 | -0.017094 |
| diseases_pathways | 1 | 0.03 | 0.786325 | 0.557421 | -0.123932 | +0.021368 |
| criteria_pathways | 1 | 0.03 | 0.782051 | 0.477421 | -0.128205 | +0.017094 |
| diseases_criteria | 1 | 0.03 | 0.777778 | 0.532009 | -0.132479 | +0.012821 |
| diseases_criteria_pathways | 1 | 0.03 | 0.769231 | 0.482707 | -0.141026 | +0.004274 |
| embeddings_only | 1 | 0.03 | 0.764957 | 0.523843 | -0.145299 | +0.000000 |
| embeddings_only | 10 | 0.03 | 0.756410 | 0.484627 | -0.153846 | +0.000000 |
| embeddings_only | 100 | 0.03 | 0.756410 | 0.484627 | -0.153846 | +0.000000 |
| diseases_pathways | 10 | 0.03 | 0.756410 | 0.484627 | -0.153846 | +0.000000 |
| diseases_pathways | 100 | 0.03 | 0.756410 | 0.484627 | -0.153846 | +0.000000 |
| criteria_pathways | 10 | 0.03 | 0.756410 | 0.484627 | -0.153846 | +0.000000 |
| criteria_pathways | 100 | 0.03 | 0.756410 | 0.484627 | -0.153846 | +0.000000 |
| diseases_criteria_pathways | 10 | 0.03 | 0.756410 | 0.484627 | -0.153846 | +0.000000 |
| diseases_criteria_pathways | 100 | 0.03 | 0.756410 | 0.484627 | -0.153846 | +0.000000 |
| diseases_criteria | 10 | 0.03 | 0.747863 | 0.468128 | -0.162393 | -0.008547 |
| diseases_criteria | 100 | 0.03 | 0.747863 | 0.468128 | -0.162393 | -0.008547 |

## Verification and interpretation

All 45 saved models were reloaded and their probabilities/predictions independently checked and rescored. The earlier DBSCAN SVM control is reproduced when C/gamma match. The best SVM independently refits from the saved matrix and fixed targets/weights with matching calibrated probabilities; protected original artifacts retain their hashes.

comparison.csv reports changes against the current logistic leader, the C=1000 logistic model on the same input, and the embeddings-only SVM at matching C/gamma. Thus gains from new input groups can be distinguished from a classifier change. These are validation-selected alternatives, not independent test estimates or a proof that every nonlinear model will behave similarly.

datasets/ preserves exact reused matrices and ordered columns with the shared fixed targets/weights; models/ saves native sklearn preprocessing and calibrated classifiers. validation_predictions/, comparison.csv, best_by_representation.csv, embedding_controls.csv, calibration_audit.csv, and settings.json preserve results and checksums.

No Kaggle submission, commit, or tag was made after this experiment. The preceding logistic feature work was committed before this run.

No SVM configuration improves the primary validation metric; retain the current logistic leader.
