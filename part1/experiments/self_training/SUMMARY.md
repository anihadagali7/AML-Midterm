# Conservative self-training from the DBSCAN seed

Eight settings cross confidence 0.90/0.95, added-example caps 0.05/0.10, and one/two rounds. Only original seed-unknown reason slots in unlabeled training rows can receive new targets. Known and DBSCAN seed targets/weights remain unchanged. Both new pseudo classes are required; their total added fractions preserve the known class ratio, capped at 0.25 per class and total seed+self influence 0.50. Each round refreshes its additions from the original seed using the previous classifier, allowing uncertain self-generated targets to be dropped.

Seed sample F1 **0.910256**, macro F1 **0.811421**. Best self-training setting **confidence0p95_cap0p1_round2**: sample F1 **0.901709**, macro F1 **0.812040**.

| Confidence | Added weight cap | Round | Self-labeled trials | Self-labeled reason slots | Sample F1 | Macro F1 | Change |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9 | 0.05 | 1 | 1014 | 4717 | 0.897436 | 0.805022 | -0.012821 |
| 0.9 | 0.05 | 2 | 1014 | 4797 | 0.897436 | 0.801106 | -0.012821 |
| 0.9 | 0.1 | 1 | 1014 | 4717 | 0.897436 | 0.805022 | -0.012821 |
| 0.9 | 0.1 | 2 | 1014 | 4802 | 0.897436 | 0.801106 | -0.012821 |
| 0.95 | 0.05 | 1 | 1014 | 4649 | 0.897436 | 0.805022 | -0.012821 |
| 0.95 | 0.05 | 2 | 1014 | 4723 | 0.897436 | 0.805022 | -0.012821 |
| 0.95 | 0.1 | 1 | 1014 | 4649 | 0.897436 | 0.805022 | -0.012821 |
| 0.95 | 0.1 | 2 | 1014 | 4734 | 0.901709 | 0.812040 | -0.008547 |

Classifier/scaler, 150 embeddings, C=1000, fitting-weight total 157, cutoff 0.5, and fallback stay fixed. Validation and test do not participate in training. No final refit with validation labels was performed. These are exploratory selections on the same 78 validation trials; no independent test improvement is established.

Exact datasets/weights, source audits distinguishing original/DBSCAN/self-training targets, teacher probabilities, classifiers, validation predictions, per-reason acceptance counts, settings, and checksums are saved. The seed reproduces the current leader. Every saved model rescored correctly and preserves all seed targets/weights; the selected model independently refits from its saved data with matching probabilities. Protected sources retain their hashes.

No Kaggle submission, commit, or tag was made.

No self-training setting improves the current leader; keep the seed model.
