# Part 1 submission archives

Prediction CSVs are archived by submission status. Reported scores identify manual Kaggle submissions. The sample-submission template is an input in part1/data/. Experiment-local source CSVs are retained.

| Model | Verified labels | Status | Kaggle F1 | CSV |
|---|---:|---|---:|---|
| Supervised logistic baseline | 157 | submitted | 0.87072 | [CSV](archive/submitted/logistic_baseline__kaggle_0.87072.csv) |
| K-means k=2 + logistic | 157 | submitted | 0.87393 | [CSV](archive/submitted/kmeans_k2__kaggle_0.87393.csv) |
| K-means k=2 + logistic + year | 157 | submitted | 0.85149 | [CSV](archive/submitted/kmeans_k2_year__kaggle_0.85149.csv) |
| K-means k=2 + Safety year boost 0.05 | 157 | reported; variant assumed | 0.86111 | [CSV](archive/submitted/kmeans_k2_safety_boost005__reported_0.86111.csv) |
| K-means k=2 + Safety year boost 0.10 | 157 | unsubmitted | — | [CSV](archive/unsubmitted/kmeans_k2_safety_boost010__157labels.csv) |
| GMM tied k=10 + logistic; earlier 157-label fit | 157 | unsubmitted | — | [CSV](archive/unsubmitted/gmm_tied_k10_logistic__157labels.csv) |
| K-means k=8 + logistic; earlier 157-label fit | 157 | unsubmitted | — | [CSV](archive/unsubmitted/kmeans_k8_logistic__157labels.csv) |

index.csv and archive_manifest.json record paths, original locations, hashes, known scores, and uncertain Safety-boost variant assignment. Original prediction bytes are unchanged.
