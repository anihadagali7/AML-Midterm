# Part 1 handoff

Written from a Claude Code session on 2026-11-02. The project is due **Saturday, November 7 at 11:59 PM ET** on Gradescope and Kaggle.

## Where things stand

- **Part 1 has a working model and generated predictions.** The predictions haven't been submitted to Kaggle.
- **No model clearly beats a tuned true-labels baseline.** The top scores are one or two validation rows apart.
- **Part 2 hasn't been started.** `part2/part2.ipynb` only has setup cells.
- **`part1/part1.ipynb` needs a full rerun.** Its PCA cell changed and its last cells have no outputs yet. Older outputs in the notebook may be stale.

## Current model

The final cell of `part1/part1.ipynb` (`32fa32c3`) implements this:

1. **Cluster:** fit a diagonal GMM with `k=2` on 20 PCA dimensions of the standardized embeddings.
2. **Generate labels:** estimate each cluster's label rates from the 157 labeled train rows, smoothed with `strength=5`. An unlabeled row gets a 1 at probability 0.7 or above, a 0 at half the label's overall rate or below, and no label in between.
3. **Train:** fit one balanced logistic regression (`C=1`) per label on all 150 standardized dimensions.
4. **Predict:** assign each label at probability 0.5 or above. A row with no predicted reason gets its most likely one, because every known trial has at least one reason.
5. **Final fit:** after scoring validation, refit with the 78 validation rows added.

Validation score, before the refit: **sample F1 0.870, macro F1 0.762**. Per-label F1 is Safety 0.842, Funding 0.667, Efficacy 0.720, Recruitment 0.970 and Other 0.609.

GMM `k=2` was kept because it meets the unsupervised requirement, gives the same result on every seed, and is simple. **Its labels add no measurable gain over the baseline**, and the report should say so. `k=2` is a cautious labeler that finds one group, not a model of the five reasons.

## Decisions still open

1. **Are kNN labels allowed as the labeling method?** Ask on Ed, since they rely on labeled neighbors and may not count as unsupervised. If they are allowed, switch to kNN labels with `C=0.1`. If not, keep GMM `k=2`.
2. **Is Other over-predicted?** It's on 29.5% of test rows, versus 13–16% in known rows. Test trials are later, so some shift is expected, but there are no test labels to check against.

## To do

1. Rerun `part1/part1.ipynb` from the top and confirm it matches the validation score above.
2. Move the experiment code from `.claude/part1-experiments/` into a notebook, for example `part1/part1_experiments.ipynb`. The assignment requires code for all experiments and figures in `.ipynb` files.
3. Get the Ed answer on kNN labels, then submit `part1/part1_submission.csv` to Kaggle.
4. Build Part 2, using `part1/part1_reason_labels.csv` for the error analysis.
5. Write the report, limited to 5 pages excluding references. It must cover cluster initialization and EM stopping settings.
6. Reword or remove notebook text tagged `<!-- AI description -->`, and clear out cell `669a75e2` ("First steps") and the old cutoff cells if they aren't needed.

## Results

These are the numbers to use. They include the fallback and the PCA fix, and `final_comparison_with_fallback_cell.py` reproduces them.

| Model | Val sample F1 | Val macro F1 | CV sample change | CV macro change |
|---|---:|---:|---:|---:|
| Baseline: true labels, `C=1` | 0.870 | 0.762 | n/a | n/a |
| True labels, `C=0.01` | 0.859 | 0.746 | -0.002 | -0.008 |
| **GMM `k=2`, `C=1` (current)** | 0.870 | 0.762 | -0.005 | -0.005 |
| GMM `k=2`, `C=10` | 0.876 | 0.785 | -0.011 | -0.009 |
| DBSCAN, `C=1` | 0.874 | 0.775 | -0.003 | +0.002 |
| Soft GMM `k=4`, weight 0.02, `C=0.1` | 0.890 | 0.745 | -0.018 | -0.041 |
| kNN labels, `C=0.1` | 0.878 | 0.744 | +0.004 | +0.006 |
| kNN labels + `criteria`, `C=0.01` | 0.893 | 0.767 | -0.006 | -0.019 |
| Ensemble: LR `C=0.01` + kNN `k=10` | 0.863 | 0.691 | +0.004 | -0.014 |

**How to read them:**

- **Validation** is the 78 rows in `part1_val.csv`, which the assignment says to use for selection. Sample F1 is the main metric. GMM rows average 5 seeds.
- **CV** is 5 repeats of 5-fold cross-validation over the 235 rows with true labels, shown as the change from the baseline. It's a stability check. Differences typically vary by about ±0.01–0.02 between repeats.
- **Noise:** one validation row can noticeably shift a score. CV also mixes train and validation rows, so it doesn't capture the shift to later test years.

**Setup definitions:**

- **kNN labels:** 10 distance-weighted neighbors with Euclidean distance, on all 150 standardized dimensions, using the same 0.7 and half-overall-rate cutoffs.
- **DBSCAN:** cosine distance, `min_samples=3`, `eps≈0.234`. That gives 37 clusters, and 48% of rows count as noise and get no labels.
- **Soft labels:** each unlabeled row appears twice, as a positive weighted `p` and a negative weighted `1 - p`, with both scaled by the listed weight.

## What we learned

**Labeling and clustering**

- **Clusters mainly isolate Recruitment.** Extra clusters mostly split Recruitment further, and Safety and Efficacy stay mixed even at 8 GMM clusters. K-Means at `k=15` found clusters that are 53% Safety and 63% Efficacy, but with few labeled rows.
- **Rare labels get almost no generated 1s.** At the 0.7 cutoff, hard-label clustering generates about 200–700 Recruitment 1s, none for Safety, Funding or Other, and at most about 29 for Efficacy.
- **The signal exists, but clustering doesn't find it.** A supervised model gets Safety F1 0.824 from the same embeddings, so don't blame the features.
- **More clusters don't help.** BIC and silhouette keep improving through 40 clusters, by which point some have no labeled rows. Higher `k` improves AUC (0.657 at `k=2`, about 0.84 at `k=4`), but there aren't enough labeled rows to turn that into reliable labels.
- **Hand-picked cutoffs failed.** At 10 GMM clusters with a 0.25 cutoff, only 8 of 21 generated Safety 1s were right on validation, and its F1 fell from 0.824 to 0.560.
- **Seeding the GMM doesn't help.** With `k=5` started at each label's mean, clusters drift from their starting reasons and vary by seed.
- **Soft labels only work at very low weight.** Results are also sensitive: raising the weight from 0.02 to 0.1 drops validation sample F1 from about 0.89 to 0.81.
- **Agreement and duplicates add little.** Requiring all three clusterers to agree behaves like GMM `k=2`. Exact-duplicate copying reaches only 9 unlabeled rows.

**Models and features**

- **The fallback is the biggest single gain.** It takes the baseline from 0.848 to 0.870 on validation sample F1, and it can't lower sample F1 because an empty prediction already scores 0. Comparisons made before it was added overstate GMM `k=2`'s edge.
- **`C` interacts strongly with the labels.** Before the fallback, GMM `k=2` scored 0.786, 0.853, 0.870 and 0.876 at `C` = 0.01, 0.1, 1 and 10.
- **kNN alone misses rare labels.** At a 0.5 threshold, Funding F1 falls to 0 at `k=3` and above on validation, likely because only 8 labeled train rows have Funding.
- **Cluster on embeddings only.** Adding other fields wrecks the clusters: at `k=4`, mean AUC falls from 0.843 to 0.514 with all fields. On their own, those fields are near chance. The embeddings come from the stopping-reason text, while drug and disease fields describe the trial.
- **Added fields don't reliably help the classifier.** Logistic regression stays near the baseline and kNN gets worse. This helps answer the assignment's question about why including everything can fail.
- **Second PCA:** it isn't redundant for clustering, because the provided embeddings were randomly rotated and their dimensions are correlated. PCA to 20 was best at `k=4` and `k=8` (AUC 0.840 and 0.822). The classifier should keep all 150 dimensions.
- **Standardizing:** this mainly changes what `C` means. Raw embeddings scored 0.682 at `C=1` but 0.857 at `C=100`, before the fallback.

**Data**

- **Duplicate embeddings:** 374 of 1,171 train rows share an embedding with another train row, as do 21 of 78 validation rows and 85 of 312 test rows with a train row. Only 18 known-label rows are in duplicate groups, in 5 groups with no conflicts.
- **Skewed labels:** the 157 labeled train rows are the earliest trials plus 40 chosen for rare reasons, so their label rates don't reflect true frequencies. Test trials start in 2010–2016, and the assignment notes that Safety becomes more common over time.

## Assignment constraints

- **No deep learning.** The assignment bans it for this project.
- **Don't reverse engineer test labels.** Don't fill in test predictions by matching test rows to train duplicates, and don't look for the fully labeled source data. The penalty is severe.
- **At least one unsupervised method** must be used to add the missing labels in Part 1.
- **Part 2:** reason embeddings and reason labels are for error analysis only, not predictive inputs.
- **Report the required metrics:** Part 1 needs sample F1, per-label F1 and macro F1.

## Files

| Path | Contents |
|---|---|
| `part1/part1.ipynb` | Main notebook, ending with the final model |
| `part1/part1_submission.csv` | Test predictions in Kaggle format, 312 rows |
| `part1/part1_reason_labels.csv` | Labels for all 1,561 trials, with `split` and `source` columns |
| `Midterm_Project.pdf` | Full assignment and rubric |
| `.claude/part1-experiments/` | Experiment scripts, saved outputs and result tables |

The two CSVs came from running the final cell's code in a separate script, not from the notebook itself.

## Reproducing the experiments

Run scripts from `part1/`, because they read the CSVs from the current directory. The project environment was Python 3.14 with scikit-learn 1.9.0.

| Script | What it covers |
|---|---|
| `experiment_plan.py`, `experiment_followup.py` | Full comparison of classifiers, labelers, features and ensembles (several minutes each, and the outputs are saved beside them) |
| `gmm_k_sweep.py`, `gmm_k_cv.py` | GMM cluster counts |
| `gmm_seeded.py` | Seeded GMM with `k=5` |
| `knn_eval.py` | kNN as a classifier and as a labeler |
| `kmeans_eval.py`, `dbscan_eval.py` | K-Means and DBSCAN |
| `cluster_features.py` | Clustering with non-embedding fields |
| `pca_second_check.py`, `duplicates_check.py` | Second PCA and duplicate checks |
| `final_comparison_cell.py`, `final_comparison_with_fallback_cell.py` | The compact comparison, without and with the fallback |

The two `final_comparison` files are notebook cell code, not standalone scripts. They need the variables from the earlier `part1.ipynb` cells, through `fd7b3106`.

**Known differences in saved results:**

- Only `final_comparison_with_fallback_cell.py` includes the fallback. Everything else scores without it, so those numbers are lower and not directly comparable.
- `gmm_k_sweep.py`, `gmm_k_cv.py`, `gmm_seeded.py`, `knn_eval.py`, `kmeans_eval.py` and `dbscan_eval.py` use `pca.fit_transform`, which gives different coordinates from `pca.transform` here (by up to 0.24). The notebook now uses `fit` then `transform`, so train, validation and test rows are projected the same way. Results at higher `k` can shift slightly as a result.
