# Part 1 submission records

The selected report winner is **anchored80/Cand2_20 at 0.89935**, user-reported. Its separate score commit is `cde3d500`, with annotated tag `part1-anchored80-cand2-inductive20-trainval235-kaggle-0.89935`. The [report handoff](../REPORT_HANDOFF.md) identifies the exact CSV, source checkout and frozen recipe at that immutable checkpoint. The latest [anchored80/diverse-kernel20 result](results/anchored80-diverse-kernel20-trainval235.json) scored .89316 and did not replace it. Model exploration is closed for this report phase.

The previous consolidated winner, **Anchored75/LDA25: 0.89850**, improved anchored's 0.89636 by .00214. Its exact [scored CSV](ready/part1_anchored75_lda25_trainval235.csv) combines the two existing verified full-235 fits and changes five of 312 label sets. Both components used the 1,014 originally masked training rows with inductive access; test inputs entered only prediction. SHA-256: `6331d245a73148ab8d155623059f60371e28a4b25312bd12c8e23fb926c20c8b`. The CSV remains at its preparation path to preserve frozen references; its [result record](results/anchored75-lda25-trainval235.json) carries the submitted status.

## Scored checkpoints

These records identify manually submitted CSVs, exact hashes, frozen recipes, fitting access, and reported scores. Older records were backfilled; the blend checkpoint records the newly reported manual result. Recording a result does not upload a file.

| Recipe | Reported Kaggle F1 | Immutable annotated tag |
|---|---:|---|
| [Selected anchored80/Cand2_20, 235 labels](../REPORT_HANDOFF.md) | **0.89935** | `part1-anchored80-cand2-inductive20-trainval235-kaggle-0.89935` |
| [Anchored80/diverse-kernel20, 235 labels](results/anchored80-diverse-kernel20-trainval235.json) | 0.89316 | `part1-anchored80-diverse-kernel20-trainval235-kaggle-0.89316` |
| [Anchored75/LDA25, 235 labels](results/anchored75-lda25-trainval235.json) | 0.89850 | `part1-anchored75-lda25-trainval235-kaggle-0.89850` |
| [Anchored, 235 labels](results/anchored-trainval235.json) | 0.89636 | `part1-anchored-trainval235-kaggle-0.89636` |
| [80% anchored/LDA blend +20% PySR, 235 labels](results/current80-pysr20-trainval235.json) | 0.89423 | `part1-current80-pysr20-trainval235-kaggle-0.89423` |
| [Original Ani](results/ani-original.json) | 0.89423 | `part1-ani-original-kaggle-0.89423` |
| [LDA 70/15/15 ensemble, 235 labels](results/lda-ensemble-trainval235.json) | 0.89401 | `part1-lda-ensemble-trainval235-kaggle-0.89401` |
| [Previous 70/15/15 ensemble, 235 labels](results/three-way-trainval235.json) | 0.89401 | `part1-three-way-trainval235-kaggle-0.89401` |
| [Tri-training, 235 labels](results/tri-training-trainval235.json) | 0.88012 | `part1-tri-training-trainval235-kaggle-0.88012` |
| [Cand2 graph/features, inductive five-seed, 235 labels](results/cand2-cs-features-inductive5-trainval235.json) | 0.89102 | `part1-cand2-cs-features-inductive5-trainval235-kaggle-0.89102` |

Scores come from user/branch reports; uploaded bytes and leaderboard partition were not independently inspected. Original Ani's .89423 identifies the original `a548721` transductive artifact. The later `8a8c3fc` best-model source and CSV changed; no score is established for that changed CSV. See [Ani provenance](../research/lda_ensemble/research/ani_snapshots/README.md).

The [historical commit index](../research/consolidation_v1/commit_index.json) maps the earlier consolidation checkpoints to commits and tags; later result tags are listed above. Never move a scored tag or transfer a score to a different CSV. A tied score gets a distinct recipe name. Preparation-time manifests retain their original status; use the later `results/*.json` or experiment `kaggle_result.json` for the reported outcome.

## Earlier logistic and clustering submissions

| Model | Fitting gold labels | Reported Kaggle F1 | Artifact |
|---|---:|---:|---|
| DBSCAN + logistic, original fit | 157 | 0.87500 | [CSV](archive/submitted/dbscan_logistic_train157__kaggle_0.87500.csv) |
| DBSCAN + logistic, refit | 235 | 0.86452 | [CSV](archive/submitted/dbscan_logistic_trainval235__kaggle_0.86452.csv) |
| Supervised logistic baseline | 157 | 0.87072 | [CSV](archive/submitted/logistic_baseline__kaggle_0.87072.csv) |
| K-means k=2 + logistic | 157 | 0.87393 | [CSV](archive/submitted/kmeans_k2__kaggle_0.87393.csv) |
| K-means k=2 + logistic + year | 157 | 0.85149 | [CSV](archive/submitted/kmeans_k2_year__kaggle_0.85149.csv) |
| K-means + Safety boost .05, variant assumed | 157 | 0.86111 | [CSV](archive/submitted/kmeans_k2_safety_boost005__reported_0.86111.csv) |

The Safety-boost row retains uncertainty about which variant was uploaded. [index.csv](index.csv) and [archive_manifest.json](archive_manifest.json) preserve earlier submitted and unsubmitted artifacts, hashes, and notes. They are historical ledgers; the scored checkpoints above carry the later consolidated result provenance. Historical `current-best-kaggle` tags were not moved and should not be read as a live leaderboard.

The original DBSCAN validation F1 .910256 belongs to its 157-label fit. The 235-label refit used the former validation labels and has no independent validation score. Its lower reported Kaggle result does not isolate which part of refitting caused the difference. [Original-fit result](../experiments/additional_methods/kaggle_result_original157.json), [refit record](../experiments/final_dbscan_trainval235/submission_manifest.json), and [historical comparison](next_submission.json).

## Future submissions

The user selected the **0.89935 anchored80/Cand2_20** result for report writing. Both that winner's frozen preparation record and the latest .89316 exploratory preparation remain unchanged; their later score records supersede preparation-time unsubmitted statuses. No additional submission is queued.

Use the [current testing policy](../TESTING_POLICY.md) to compare recipes and [AGENTS.md](../AGENTS.md) for result commits and immutable tags. A final 235-label refit is a separate artifact from CV and original validation. Preserve its exact CSV, hash, fitting population, input access, recipe/source, and reported score provenance.

Uploads remain manual. An archived CSV or preparation manifest does not queue an upload. Earlier statements about remaining daily slots are historical and do not establish a current budget.
