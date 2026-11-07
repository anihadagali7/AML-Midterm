# Part 1 submission records

The highest recorded, user-reported Kaggle score is **anchored: 0.89636**. Its exact [scored CSV](archive/submitted/anchored_reason_trainval235__kaggle_0.89636.csv) was fitted on **235 verified labels** with **757 accepted generated rows** from the **1,014 originally masked training rows**. Kaggle test inputs were used only for prediction. The SHA-256 is `1db6433dc5a5b900d440db16e5ef8375eb51dcf02d3f270217527d8b174319eb`.

## Scored checkpoints

These records identify earlier manually submitted CSVs, exact hashes, frozen recipes, fitting access, and reported scores. The dedicated commits and tags are backfilled checkpoints, not new uploads.

| Recipe | Reported Kaggle F1 | Immutable annotated tag |
|---|---:|---|
| [Anchored, 235 labels](results/anchored-trainval235.json) | **0.89636** | `part1-anchored-trainval235-kaggle-0.89636` |
| [Original Ani](results/ani-original.json) | 0.89423 | `part1-ani-original-kaggle-0.89423` |
| [LDA 70/15/15 ensemble, 235 labels](results/lda-ensemble-trainval235.json) | 0.89401 | `part1-lda-ensemble-trainval235-kaggle-0.89401` |
| [Previous 70/15/15 ensemble, 235 labels](results/three-way-trainval235.json) | 0.89401 | `part1-three-way-trainval235-kaggle-0.89401` |
| [Tri-training, 235 labels](results/tri-training-trainval235.json) | 0.88012 | `part1-tri-training-trainval235-kaggle-0.88012` |

Scores come from user/branch reports; uploaded bytes and leaderboard partition were not independently inspected. Original Ani's .89423 identifies the original `a548721` transductive artifact. The later `8a8c3fc` best-model source and CSV changed; no score is established for that changed CSV. See [Ani provenance](../research/lda_ensemble/research/ani_snapshots/README.md).

The [commit index](../research/consolidation_v1/commit_index.json) maps these checkpoints to commits and tags. Never move a scored tag or transfer a score to a different CSV. A tied score gets a distinct recipe name. Preparation-time manifests retain their original status; use the later `results/*.json` or experiment `kaggle_result.json` for the reported outcome.

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

The [Cand2 strict-inductive five-seed CSV](candidates/cand2_cs_features_inductive5.csv) is prepared as a user-requested exploratory candidate. It changes 49/312 predictions from anchored and has weaker local benchmark scores; it is not a promoted leader or a recorded upload. [Recipe, evidence and exact hash](../experiments/cand2_submission_v1/README.md).

Use the [current testing policy](../TESTING_POLICY.md) to compare recipes and [AGENTS.md](../AGENTS.md) for result commits and immutable tags. A final 235-label refit is a separate artifact from CV and original validation. Preserve its exact CSV, hash, fitting population, input access, recipe/source, and reported score provenance.

Uploads remain manual. An archived CSV or preparation manifest does not queue an upload. Earlier statements about remaining daily slots are historical and do not establish a current budget.
