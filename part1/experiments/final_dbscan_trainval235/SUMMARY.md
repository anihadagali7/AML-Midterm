# Final DBSCAN fit: 235 verified labels

Submitted manually; reported Kaggle score **0.86452**. Final CSV: `part1/submissions/archive/submitted/dbscan_logistic_trainval235__kaggle_0.86452.csv`. The score was reported by Michael; no upload was performed by the agent.

## Frozen recipe and final training

The current validation-selected DBSCAN recipe is refitted using all 235 verified labels (157 original training + 78 former validation). The original 1,014 unlabeled training trials remain available. Test trials never enter preprocessing, clustering, pseudo-label generation, or classifier fitting.

Classifier StandardScaler is fitted on 235 verified rows. Cluster StandardScaler and DBSCAN are fitted on the combined 1,249 training/validation rows. DBSCAN retains min_samples=5 and the training-derived q25 fifth-neighbor-radius rule (floor 1e-6); its numerical radius is relearned as 0.00014595414084762. Labels/weights are regenerated from the 235 known labels using unchanged support, confidence, smoothing, cap, and class-budget rules. Five L2 logistic classifiers retain C=1000, solver=lbfgs, tol=1e-4, max_iter=5000, class_weight=None, random_state=42. Fitting weights sum to 235 per reason. Threshold 0.5 and highest-probability fallback stay fixed.

## Generated labels

| Reason | Verified positives | Verified negatives | Generated positives | Generated negatives | Fitting rows |
|---|---:|---:|---:|---:|---:|
| Safety | 25 | 210 | 0 | 0 | 235 |
| Funding | 12 | 223 | 0 | 0 | 235 |
| Efficacy | 31 | 204 | 8 | 165 | 408 |
| Recruitment | 151 | 84 | 0 | 0 | 235 |
| Other | 35 | 200 | 0 | 0 | 235 |

## Submission checks and evaluation provenance

312 unique test IDs in sample-submission order; columns ID/Predicted; valid nonempty label strings. Fallback on 17 trials. Predictions change on 34 test trials relative to the earlier DBSCAN model fitted with 157 verified labels. Saved models reload and the five classifiers independently refit from the persisted data/weights with matching probabilities. All protected original model, data, and submission checksums remain unchanged.

The historical pre-refit validation sample F1 is 0.910256, macro F1 0.811421. These scores describe the earlier 157-label model. The new final model has trained on the former validation labels and has no independent local validation score; its reported Kaggle sample F1 is **0.86452**. No new settings were chosen during this refit.

Submission SHA-256: `19f29c7bcbda974f8ae9646d3ced83bad2953ead45412e88731e8c12e55d2661`.

Reproduce using part1/final_dbscan_trainval235.ipynb. datasets/ preserves exact raw embeddings, combined verified labels, generated targets, raw/normalized weights, test features, and source audits. models/ stores the final labeler and classifiers. predictions/ stores test probabilities, labels, and fallback flags. submission_manifest.json records provenance, recipe, checksums, and manual upload status.
