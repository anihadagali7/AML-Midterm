# Ani branch validation comparison

Later results: Michael reported Kaggle **0.87500** for the original 157-label DBSCAN model and **0.86452** for its 235-label refit; Ani remains higher at **0.89423**. The [new paired CV benchmark](../paired_recipe_cv/SUMMARY.md) uses identical 235-label folds and finds Ani 0.914273 vs DBSCAN 0.888316, with the same ranking under both shared feature-access controls. The review below preserves what was established at the earlier validation-only checkpoint.

Reviewed `origin/ani` at `a5487215115e30afb68ab865bc5fd98be6b3e199`: [midterm/part1_best_model.ipynb](https://github.com/anihadagali7/AML-Midterm/blob/a5487215115e30afb68ab865bc5fd98be6b3e199/midterm/part1_best_model.ipynb). This is the latest notebook associated with the commit message reporting Kaggle 0.89423; the older handoff document describes a previous model.

| Model | Validation sample F1 | Validation macro F1 | Kaggle |
|---|---:|---:|---|
| Current DBSCAN + one-vs-rest logistic regression | 0.910256 | 0.811421 | Current candidate not submitted |
| Ani GMM-assisted self-training ensemble | 0.893162 | 0.713700 | 0.89423, user-reported and recorded in Ani's commit message |

Our validation sample F1 is higher by **0.017094**, or about **1.71 F1 points**. The supplied train/validation/test CSVs match byte for byte, and both validation scores use sample-wise F1 on the same 78 trials. The current model's Kaggle score is unknown, so the validation comparison does not establish better Kaggle performance.

Ani's notebook displays rounded validation metrics. Its exact scores were reproduced using its setup, unsupervised preprocessing, model definitions, and final validation call, redirecting only the data path to our unchanged CSVs. No CV, test prediction generation, or final refit was run for this review. `ani_validation_review.json` records exact metrics and source checksums.

Ani uses standardized 150-dimensional embeddings, PCA to 30 dimensions and a 30-component diagonal GMM, balanced multinomial logistic regression blended with cosine kNN, and two self-training rounds. Validation fitting starts with the 157 known training trials and adds 985 pseudo-labeled trials from unlabeled train plus test features. Its decision rule always selects the highest-probability reason and includes other reasons at probability >=0.3. Our classifier is independent one-vs-rest logistic regression with cutoff 0.5 plus fallback and conservative DBSCAN-generated Efficacy labels.

Two protocol differences matter: Ani learns unsupervised preprocessing from all train/validation/test embeddings, while our preprocessing uses training rows only; and Ani refits the submitted model with **235 verified labels**, including the 78 validation labels, whereas our current candidate uses **157 verified training labels**. Validation labels are not used in Ani's supervised fit for the reported held-out validation score. The notebook's displayed repeated-CV sample F1 of about 0.914 is a separate metric, not the held-out validation score.

This review does not change models, submissions, or tags. It reads Ani's branch without merging it into the current experiment branch.
