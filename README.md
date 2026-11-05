# AML-Midterm

## Part 1 experiment checkpoint

The current validation leader is **DBSCAN pseudo-labeling + one-vs-rest logistic regression**, with sample-wise F1 **0.910256** and macro F1 **0.811421** on the 78 labeled validation trials. Its Kaggle score is unknown. The best reported Kaggle submission remains the earlier K-means K=2 model, scoring **0.87393**, tagged `current-best-kaggle`.

The current model uses the original 150 embedding features. DBSCAN uses `eps=0.00012142366141394104` and `min_samples=5`; logistic regression uses `C=1000`. Generated labels are accepted at estimated probability >=0.60 for positives and <=0.10 for negatives, with at least three known supporting trials, and only when both generated classes exist for a reason. Noise remains unlabeled. Generated raw weights are capped at 0.25 per example and at 50% of the corresponding known class weight, preserving the known positive/negative weight ratio. Original labels keep raw weight 1; fitting weights are normalized to total 157 per reason.

Only **Efficacy** receives generated labels in this configuration: 8 positives and 125 negatives. The other four classifiers use the original 157 labeled trials. The classifier scaler is frozen from those 157 trials; the clustering scaler uses all 1,171 training trials. Validation is used for model selection, never for fitting.

Final predictions select every reason with probability >=0.50. If none qualifies, they select the single highest-probability reason. Keep this policy for the next feature comparisons.

| Completed experiment | Configurations | Result |
|---|---:|---|
| [Additional methods](part1/experiments/additional_methods/SUMMARY.md) | 59 | DBSCAN + logistic regression leads the tested SVM, GaussianNB, graph, hybrid, and KNN alternatives |
| [DBSCAN density and regularization](part1/experiments/dbscan_refinement/SUMMARY.md) | 27 | No improvement over the current leader |
| [Confidence and influence](part1/experiments/dbscan_label_tuning/SUMMARY.md) | 49 including control | No improvement; many thresholds produce identical labels |
| [Fine weight-cap and class-budget tuning](part1/experiments/dbscan_weight_fine_tuning/SUMMARY.md) | 73 including control | Per-label cap 0.25 remains best; tested class budgets do not bind |
| [Prediction cutoffs and fallback](part1/experiments/dbscan_prediction_cutoffs/SUMMARY.md) | 10 | Cutoffs 0.40–0.60 give identical predictions; fallback raises sample F1 from 0.876068 to 0.910256 |
| [Classifier feature additions](part1/experiments/classifier_features/SUMMARY.md) | 11 including reference | Diseases and the 2,000-term criteria TF-IDF tie the reference; broad pathways and stronger text filtering reduce F1 |
| [Feature weighting and regularization](part1/experiments/classifier_feature_tuning/SUMMARY.md) | 76 including four controls | Some additions improve matched controls at stronger regularization, but none beat the C=1000 embeddings-only leader |
| [Combined classifier features](part1/experiments/classifier_combinations/SUMMARY.md) | 10 including two controls | All four combinations stay below the leader; diseases + criteria reaches 0.893162 at C=1000 |

These are comparisons on the same repeatedly consulted validation set, not independent estimates of test performance. The nearest weight settings differ from the leader on just one validation trial, so the precise winning weight has limited supporting evidence.

### Reproduce or extend

Executed notebooks are saved in `part1/`: `additional_methods.ipynb`, `dbscan_refinement.ipynb`, `dbscan_label_tuning.ipynb`, `dbscan_weight_fine_tuning.ipynb`, and `dbscan_prediction_cutoffs.ipynb`. These new notebooks accept supplied CSVs either directly in `part1/` or in `part1/data/`. Each experiment folder preserves its settings, comparison tables, validation predictions, and relevant training datasets, label audits, and models.

`part1/classifier_features.ipynb` tests disease multi-hot indicators, broad pathway indicators with a missing-annotation flag, and eligibility-text TF-IDF independently, keeping DBSCAN targets and weights fixed. Training-only frequency profiles determine the tested filtering settings. Exact sparse classifier matrices, ordered feature names, shared targets/weights, preprocessing objects, models, and validation predictions are saved in `part1/experiments/classifier_features/`. No addition improves sample-wise F1; retain the embeddings-only reference. Feature combinations and new classifier regularization or feature-block weights have not been tested in this round.

`part1/classifier_feature_tuning.ipynb` follows up with C=1/10/100/1000 and added-block multipliers 0.25/1/4 for all three disease and all three criteria representations. It reuses the exact preprocessing, generated labels, and weights, with an embeddings-only control at each C. The 76 settings do not improve the current leader. Exact weighted matrices, models, predictions, and matched-control comparisons are preserved in `part1/experiments/classifier_feature_tuning/`. No pathway or combined-feature tuning is included.

`part1/classifier_combinations.ipynb` evaluates the four pairs/triple of diseases, criteria, and pathways with logistic regression at C=100/1000, plus embeddings-only controls. It freezes the existing diseases_min5, criteria_min3, and pathways_min30 representations and all pseudo-label settings. No combination beats the embeddings-only leader. Exact combined matrices and native preprocessing objects in `part1/experiments/classifier_combinations/` are available for a later nonlinear-model comparison; no nonlinear model is trained in this round.

- [Current trained model](part1/experiments/additional_methods/classifier_models/dbscan_m5_q25__logistic_C1000.joblib)
- [Exact current training targets and raw weights](part1/experiments/additional_methods/datasets/dbscan_m5_q25.csv)
- [Selected prediction policy and source-model checksum](part1/experiments/dbscan_prediction_cutoffs/selected_policy.json)
- [Prepared, unsubmitted Kaggle candidate](part1/experiments/additional_methods/submissions/part1_dbscan_m5_q25_logistic_C1000_submission.csv)

All ten cutoff/fallback prediction sets were independently rescored from disk. Training experiments also check reference reproduction and refit the selected model from its saved dataset. Feature and combination followups retain this embeddings-only configuration as the reference, with combined matrices saved for a later nonlinear comparison. These experiments introduce no new submission tag.
