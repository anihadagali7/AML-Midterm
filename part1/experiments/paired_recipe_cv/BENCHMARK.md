# Fixed benchmark for subsequent Part 1 recipes

**Active policy: training-only feature access**, chosen by the user after reviewing the four-variant experiment. Future fitting must exclude each fold's held-out inputs and all Kaggle test inputs. The all-embeddings results are retained as historical diagnostics; they are not the active selection reference. This changes the policy for future work, not the original predeclared experiment in protocol.json/PROTOCOL.md.

Reuse `part1/paired_recipe_cv.ipynb` and the saved `splits/repeat0.csv` through `repeat7.csv`. Do not regenerate different folds after inspecting results. Exact input features/visible labels are saved in `datasets/shared_raw_features_labels.csv`, with their checksum in settings.json.

## Evaluation contract

- Keep the existing pool of 157 verified training trials and 78 former validation trials: 235 known labels. Here, training-only means the training partition of each CV fold. It does not mean silently returning to the original 157-label pool; the fold's 47 held-out trials replace the fixed 78-trial holdout for this benchmark.
- Use the saved eight repeats of five folds, seeds 100–107, with 188 known fitting labels and 47 held-out labels per fit.
- Keep each held-out trial's labels out of classifier fitting, cluster-to-label mapping, confidence estimation, and generated training targets. Held-out known rows must not become pseudo-labeled training rows.
- Fit preprocessing, clustering, label generation, and classifiers only with the fold's 188 verified training trials and the 1,014 originally unlabeled training trials. Exclude all 47 held-out trials and all 312 Kaggle test trials from every fitting stage, including scaling, PCA, clustering, and pseudo-labeling.
- Preserve each candidate's declared training procedure and decision rule across every fold. Refit learned preprocessing inside each fold. Apply learned transformations to held-out inputs only when predicting them.
- Pool all 235 held-out predictions within each repeat and compute sample-wise F1. Average the eight repeat scores for the primary result. Also report repeat SD, paired differences/wins, macro F1, and each reason's F1.
- Compare future candidates against the training-only Ani reference below, retaining training-only DBSCAN as a secondary baseline. Do not use the higher transductive reference as if it were a training-only result.

## Frozen reference results

| Recipe | Feature access | Mean sample F1 | Mean macro F1 |
|---|---|---:|---:|
| Ani recipe with training-only access | Training-only | **0.908865** | 0.822043 |
| DBSCAN original recipe | Training-only | 0.888316 | 0.807876 |

Ani wins this active comparison in **31/40 folds**, ties two, loses seven, and wins on the average of **all eight repeats**. Its mean advantage is **0.020550 sample F1**. See active_benchmark.json and training_only_reference.csv for the machine-readable policy and reference results.

## Additional candidates evaluated on the same saved folds

The [leading-recipe follow-up](../leading_recipes_cv/SUMMARY.md) adds three fixed recipes, with preprocessing and pseudo-label generation rebuilt inside folds and the same training-only exclusions:

| Candidate | Mean sample F1 | Change vs Ani | Change vs original DBSCAN |
|---|---:|---:|---:|
| K-means k=8 + logistic C1000 | 0.895089 | -0.013777 | +0.006773 |
| Normalized DBSCAN + logistic C100 | 0.892624 | -0.016241 | +0.004309 |
| GMM tied k=10 + logistic C1000 | 0.887553 | -0.021312 | -0.000762 |

Ani remains the frozen reference. It beats K-means and normalized DBSCAN in every repeat average. K-means improves on original DBSCAN in seven of eight repeats; normalization improves in five. These are descriptive paired results on the same development folds, without a new Kaggle comparison.

## Historical all-embeddings diagnostics

| Recipe | Feature access | Mean sample F1 | Mean macro F1 |
|---|---|---:|---:|
| Ani original recipe | Transductive | 0.914273 | 0.838763 |
| DBSCAN recipe with transductive access | Transductive | 0.887606 | 0.806349 |

Compared with training-only access, Ani's mean improves by 0.005408 while DBSCAN's falls by 0.000709. These are observed CV differences; they do not establish a general benefit or harm. The diagnostic changes both feature-only preprocessing access and eligibility to pseudo-label test rows, so it does not isolate those effects individually.

## Interpretation

The recipe ordering agrees with the known Kaggle results: Ani 0.89423; original 157-label DBSCAN 0.87500; 235-label DBSCAN refit 0.86452. Ani's submitted model used transductive access; **the training-only Ani variant has no known Kaggle score**. Two recipes do not independently validate this benchmark's predictive accuracy for every future change.

The 235 labels already influenced recipe choices. These are reusable development folds, not an untouched final test. Repeated tuning can overfit them. The latest-year Kaggle trials can differ from random folds, and each CV model has 188 fitting labels rather than 157 or 235. Keep all candidate results, including negative results, and use occasional manual Kaggle checks for shortlisted changes.

No new submission or full-data refit is implied by a benchmark win. Save a separate candidate and its exact fitting-data/feature-access policy before manual upload. Historical model snapshots, original CSVs, and Git tags remain unchanged.
