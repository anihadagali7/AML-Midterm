# Consolidated submission status

Highest reported score: **Ani original, 0.89423** (235 labels, transductive). Latest reported submission: **70/15/15 ensemble, 0.89401** (235 labels, inductive). Classical tri-training scored **0.88012**. The exact three CSVs were added to `archive/submitted`; originals, SHA-256 hashes, fitting counts, and score provenance are in [index.csv](index.csv) and `archive_manifest.json`. Integration performed no upload.

The newer Ani `8a8c3fc` best CSV is separately preserved without an assigned score. The original .89423 CSV matches that commit’s `part1_first_submission.csv`. See [the current handoff](../RESEARCH_HANDOFF.md) and [testing policy](../TESTING_POLICY.md). No new file is being nominated in `ready/`; two submissions remained in the user’s last reported budget.

The guide below is the **historical 7425345 checkpoint**; its “latest” statements and ready-state manifests predate the consolidated records above. Historical tags and model snapshots remain unchanged.

---

# Part 1 Kaggle submissions

## Latest reported result

[Original 157-label DBSCAN — Kaggle **0.87500**](archive/submitted/dbscan_logistic_train157__kaggle_0.87500.csv)

Michael submitted the exact original predictions, with no retraining or validation labels used for fitting. It scores **0.87500**, versus **0.86452** for the 235-label refit: a **0.01048** gain for the original fit on this Kaggle evaluation. Ani remains the highest reported score at **0.89423**.

The ready folder is empty. All uploads are manual. No new submission is queued.

## Submitted and prepared archives

| Model | Verified labels | Status | Kaggle F1 | CSV |
|---|---:|---|---:|---|
| DBSCAN + logistic; original 157-label fit (no validation refit) | 157 | submitted manually; score reported | 0.87500 | [CSV](archive/submitted/dbscan_logistic_train157__kaggle_0.87500.csv) |
| DBSCAN + logistic; final train+validation fit | 235 | submitted manually; score reported | 0.86452 | [CSV](archive/submitted/dbscan_logistic_trainval235__kaggle_0.86452.csv) |
| Supervised logistic baseline | 157 | submitted | 0.87072 | [CSV](archive/submitted/logistic_baseline__kaggle_0.87072.csv) |
| K-means k=2 + logistic | 157 | submitted | 0.87393 | [CSV](archive/submitted/kmeans_k2__kaggle_0.87393.csv) |
| K-means k=2 + logistic + year | 157 | submitted | 0.85149 | [CSV](archive/submitted/kmeans_k2_year__kaggle_0.85149.csv) |
| K-means k=2 + Safety year boost 0.05 | 157 | reported; variant assumed | 0.86111 | [CSV](archive/submitted/kmeans_k2_safety_boost005__reported_0.86111.csv) |
| K-means k=2 + Safety year boost 0.10 | 157 | unsubmitted | — | [CSV](archive/unsubmitted/kmeans_k2_safety_boost010__157labels.csv) |
| GMM tied k=10 + logistic; earlier 157-label fit | 157 | unsubmitted | — | [CSV](archive/unsubmitted/gmm_tied_k10_logistic__157labels.csv) |
| K-means k=8 + logistic; earlier 157-label fit | 157 | unsubmitted | — | [CSV](archive/unsubmitted/kmeans_k8_logistic__157labels.csv) |

## Evaluation provenance

The original DBSCAN validation F1 **0.910256** was measured on 78 trials with 157 known training labels. The 235-label refit used those former validation labels and has no independent local validation score. The Kaggle comparison directly compares these two fitted pipelines; it does not isolate which refit component caused the difference.

[Original-fit result](../experiments/additional_methods/kaggle_result_original157.json) · [Refit result](../experiments/final_dbscan_trainval235/submission_manifest.json) · [Comparison manifest](next_submission.json)

`index.csv` and `archive_manifest.json` record paths, status, reported scores, notes, and hashes. The duplicate unsubmitted 157-label archive and ready copy were consolidated into the scored archive without changing any bytes. The experiment-local original and fitted model are unchanged. Its preparation-time submission_manifest.json is retained as a historical snapshot; the later kaggle_result_original157.json records the actual score.

The +0.05 Safety-boost score retains the earlier uncertainty about which boost variant was submitted. Ani’s CSV belongs to a separate branch. Existing Git tags remain unchanged; the historical current-best-kaggle tag still points to K-means rather than the newly reported DBSCAN result.
