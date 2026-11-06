# Frozen paired recipe evaluation

Before fitting: eight repeats of five-fold CV over all 235 verified labels, KFold seeds 100–107 and original training-known then validation ordering, matching Ani’s original CV. Each fit sees 188 verified labels and scores the same 47 held-out labels for every variant. No parameter search.

Primary: compare the two original recipes with their native feature-access policies (DBSCAN training-only, Ani transductive). Controls: compare both with training-only features, and both with transductive feature access. These controls modify feature access deliberately; they are not previously submitted models. Preserve each classifier’s own scaling, training rules, and fixed prediction cutoff.

Feature-only transductive preprocessing may see held-out embeddings, never held-out labels. Held-out known rows are excluded from classifier/label-mapping targets and from every pseudo-label training pool in both policies. Test labels are never available. Each round rebuilds label mapping from permitted fitting labels. Train-only preprocessing is refitted separately inside each fold.

Primary metric: pooled trial sample F1 within repeat, then the eight-repeat mean. Macro/per-label scores and paired fold/repeat differences are secondary. Repeats share trials; dispersion is descriptive, not a confidence interval from independent tests. No protocol will be selected based on whether its ranking resembles the already-known Kaggle ranking. No new submission or full-data refit.
