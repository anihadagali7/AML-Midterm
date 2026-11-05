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

These are comparisons on the same repeatedly consulted validation set, not independent estimates of test performance. The nearest weight settings differ from the leader on just one validation trial, so the precise winning weight has limited supporting evidence.

### Reproduce or extend

Executed notebooks are saved in `part1/`: `additional_methods.ipynb`, `dbscan_refinement.ipynb`, `dbscan_label_tuning.ipynb`, `dbscan_weight_fine_tuning.ipynb`, and `dbscan_prediction_cutoffs.ipynb`. These new notebooks accept supplied CSVs either directly in `part1/` or in `part1/data/`. Each experiment folder preserves its settings, comparison tables, validation predictions, and relevant training datasets, label audits, and models.

- [Current trained model](part1/experiments/additional_methods/classifier_models/dbscan_m5_q25__logistic_C1000.joblib)
- [Exact current training targets and raw weights](part1/experiments/additional_methods/datasets/dbscan_m5_q25.csv)
- [Selected prediction policy and source-model checksum](part1/experiments/dbscan_prediction_cutoffs/selected_policy.json)
- [Prepared, unsubmitted Kaggle candidate](part1/experiments/additional_methods/submissions/part1_dbscan_m5_q25_logistic_C1000_submission.csv)

All ten cutoff/fallback prediction sets were independently rescored from disk. Earlier training experiments also check reference reproduction and refit the selected model from its saved dataset. The next comparison should add one feature group at a time while retaining this embeddings-only configuration as the reference. This checkpoint introduces no new submission tag.
