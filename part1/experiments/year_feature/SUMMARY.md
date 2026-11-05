# Year as a classifier feature

Controlled comparison on the same 78 validation trials. Add only standardized start_year to the final classifier; use the same training rows, labels, weights, C=1000, threshold 0.5, and nonempty fallback.

| Model | Features | Sample-wise F1 | Macro F1 | Safety F1 | Effect of year | Kaggle |
|---|---:|---:|---:|---:|---:|---:|
| baseline_embeddings | 150 | 0.8932 | 0.8043 | 0.9412 | +0.0000 | 0.87072 |
| baseline_embeddings_year | 151 | 0.8974 | 0.8166 | 0.9412 | +0.0043 | unsubmitted |
| k02_embeddings | 150 | 0.8568 | 0.6808 | 0.8571 | +0.0000 | 0.87393 |
| k02_embeddings_year | 151 | 0.8611 | 0.6937 | 0.8571 | +0.0043 | 0.85149 |

## Results

Adding year improved baseline validation sample-wise F1 from 0.8932 to 0.8974 (+0.0043). Safety F1 changed from 0.9412 to 0.9412.
Adding year improved k02 validation sample-wise F1 from 0.8568 to 0.8611 (+0.0043). Safety F1 changed from 0.8571 to 0.8571.

## Per-reason validation F1

| Model | Safety | Funding | Efficacy | Recruitment | Other |
|---|---:|---:|---:|---:|---:|
| baseline_embeddings | 0.9412 | 0.6667 | 0.8125 | 0.9697 | 0.6316 |
| baseline_embeddings_year | 0.9412 | 0.6667 | 0.8387 | 0.9697 | 0.6667 |
| k02_embeddings | 0.8571 | 0.4000 | 0.7500 | 0.9524 | 0.4444 |
| k02_embeddings_year | 0.8571 | 0.4000 | 0.7500 | 0.9615 | 0.5000 |

## Interpretation and artifacts

K-means still used 150 embeddings. The K=2 training set remains exactly 157 original-label trials plus 388 generated-label trials with generated weight 0.25. No year-based Safety probability or label was manually imposed. Year's influence was learned through the same regularized classifiers. The supplied labeled subset was deliberately enriched for rare reasons, so year coefficients are associations in the selected training data and must not be interpreted as unbiased population trends.

`year_coefficients.csv` records each fitted year coefficient per one training standard deviation of year, holding the embedding features fixed. Positive coefficients increase that reason's predicted log-odds; negative coefficients decrease them. The scaling mean and standard deviation are recorded for each model.

`datasets/` saves the exact selected features, labels, and weights for all four configurations. `models/` saves fitted pipelines with their feature lists; `validation_predictions/` saves per-trial results. `comparison.csv` contains metrics and `settings.json` contains settings and checksums. Saved datasets reproduce the year models' predictions, loaded models reproduce their probabilities, and the source datasets, label audit, clusters, K=2 model, and both existing submission CSVs remain unchanged.

Existing Kaggle scores are user-reported. The K=2 year-feature submission is exported in the following section. K=2 plus year scored 0.85149 after manual submission, below K=2 without year at 0.87393. The supervised baseline plus year remains unsubmitted. These validation results use the same small set that informed earlier experiments.
## Reported Kaggle result: adding year to K=2

Michael reported **0.85149** after submitting `part1_kmeans_K2_year_C1000_submission.csv`.

| K=2 classifier | Validation sample-wise F1 | Kaggle score |
|---|---:|---:|
| Embeddings only | 0.8568 | **0.87393** |
| Embeddings plus year | **0.8611** | 0.85149 |

Adding year improved validation by approximately 0.00427 but reduced the reported Kaggle score by **0.02244**, or **2.244 F1 points**. Retain K=2 without year as the best observed Kaggle model among the submitted experiments. This year-feature change did not transfer successfully to the scored test trials.

A plausible explanation is that year relationships learned from the selected/enriched labels and earlier trials do not transfer to later trials. The aggregate Kaggle score alone cannot establish that cause. This result applies to year as a feature in the K=2 augmented classifier; the supervised model plus year remains unsubmitted, and year has not been added to clustering. Preserve both models and record this negative result.

## K=2 plus year Kaggle submission

Prepared `part1_kmeans_K2_year_C1000_submission.csv` with 312 predictions from the saved 151-feature classifier. Test trials using the nonempty fallback: 21. Predictions differ from K=2 without year on 19 trials. **Kaggle result: 0.85149 (reported by Michael after manual submission).** Compare with K=2 without year at **0.87393**. The model and submission checksums are recorded in `k02_year_submission.json`.
