# Paired CV: Ani and DBSCAN recipes

**Current policy for future work:** the user chose **shared_train_only**, with Ani sample F1 **0.908865** and DBSCAN **0.888316**, as the active comparison. All held-out and Kaggle test inputs are excluded from fitting. The original four-variant protocol and results below are preserved as the historical experiment. See BENCHMARK.md and active_benchmark.json for the active policy.

Fixed before fitting: eight repeats of five-fold CV over the same 235 verified labels, 188 fitting labels / 47 held out per model, seeds 100–107 and KFold ordering matching Ani’s original CV. Both recipes use the same folds and primary sample-F1 metric. No parameter tuning or new submission.

## Four variants

| Variant | Mean sample F1 | Repeat SD | Mean macro F1 |
|---|---:|---:|---:|
| dbscan_train_only | 0.888316 | 0.003882 | 0.807876 |
| ani_transductive | 0.914273 | 0.005628 | 0.838763 |
| ani_train_only | 0.908865 | 0.006631 | 0.822043 |
| dbscan_transductive | 0.887606 | 0.003930 | 0.806349 |

## Paired comparisons

| Comparison | DBSCAN F1 | Ani F1 | Ani minus DBSCAN | Ani fold wins / ties / losses | Ani repeat wins / ties / losses |
|---|---:|---:|---:|---|---|
| native_recipes | 0.888316 | 0.914273 | +0.025957 | 35/1/4 | 8/0/0 |
| shared_train_only | 0.888316 | 0.908865 | +0.020550 | 31/2/7 | 8/0/0 |
| shared_transductive | 0.887606 | 0.914273 | +0.026667 | 35/1/4 | 8/0/0 |

Primary **native_recipes** preserves the original feature policies: DBSCAN training-only, Ani transductive. The two shared-policy comparisons deliberately change feature access to separate it from recipe architecture. All policies and comparisons were declared before scores; none was chosen because it resembles Kaggle.

## Label and feature access

Every fold withholds the same 47 known labels from all supervised fitting, cluster-label mapping, and pseudo-label targets. Held-out known rows never enter augmentation, even under transductive preprocessing. Training-only feature fitting uses 188 verified + 1,014 originally unlabeled training trials and excludes held-out/test features entirely. Transductive feature-only scaler/PCA/GMM or DBSCAN use all 1,561 embeddings, never held-out/test labels; both recipes may augment from the same 1,014 unlabeled training + 312 test rows. DBSCAN classifier scaling stays verified-fitting-only, while Ani scaling uses the permitted feature pool, preserving recipe choices.

DBSCAN retains C=1000, LBFGS/tol=1e-4/max_iter=5000, min_samples=5, q25 radius, conservative support/confidence/weight rules, cutoff 0.5 and fallback. Ani retains C=0.3 balanced multinomial LR + cosine kNN10 (70/30), PCA30/GMM30-diagonal, 80/20 teacher/GMM pseudo-label scores, confidence 0.8, two rounds, cutoff 0.3 and top-label inclusion.

## What the scores can establish

Mean sample F1 pools all 235 out-of-fold trial predictions per repeat, then averages eight repeats. Because every fold has 47 trials, it also equals the mean of the 40 fold sample-F1 values. Macro F1 is pooled within repeat before averaging; it may differ from Ani’s original mean fold macro F1. Repeats share trials/training examples: SD and paired win counts are descriptive, not independent-test significance claims.

The old 78-trial validation and previous 157-label three-fold CV are separate settings and should not be compared numerically as if identical to this 235-label five-fold evaluation. Its recipes have already been selected using these data, so this is a consistent exploratory benchmark, not a fresh independent test. Kaggle trials are later than the labeled pool; random folds cannot reproduce that shift. Agreement with the known Kaggle ordering for two recipes is useful evidence, not proof that future rankings will transfer.

Known user-reported Kaggle scores: Ani after 235-label refit **0.89423**; DBSCAN original 157-label fit **0.87500**; DBSCAN after 235-label refit **0.86452**. This CV evaluates fits using 188 verified labels and therefore does not reproduce either complete final submission fit.

## Reproduction and artifacts

Notebook: part1/paired_recipe_cv.ipynb. protocol.json/PROTOCOL.md fix all choices before fitting. splits/ preserves all fold assignments and row roles. datasets/ saves shared raw features/visible labels and every fold’s exact DBSCAN targets/raw/normalized weights or Ani augmented targets/self-training audits. preprocessing/ saves fold-local or feature-only global preprocessing and cluster models. models/ and predictions/ preserve all 160 fitted variants and held-out probabilities/labels. comparison.csv, paired_comparison.csv, repeat_scores.csv, subgroup_scores.csv and fit/role/generation audits support the tables.

Both native validation endpoints reproduce their original saved probabilities and scores. The first native Ani CV fit also matches direct execution of Ani’s original model functions. All 160 model packets reload and independently rescore from persisted predictions; all four first-context variants independently refit from exported targets/weights. Original source models, CSVs, and data keep their hashes. No Kaggle upload, new submission CSV, commit, push, or tag was made.

## Observed results at this checkpoint

Ani’s original recipe averages **0.914273** sample F1, compared with **0.888316** for the original DBSCAN recipe. Ani wins **35/40** folds, ties **1**, and wins the mean score in **all eight repeats**. The direction agrees with the known Kaggle ordering.

With both recipes restricted to training-only feature access, Ani remains ahead: **0.908865** versus **0.888316**. With transductive access available to both, the scores are **0.914273** versus **0.887606**. The advantage is not explained solely by extra feature access in these controls.

Use Ani as the fixed reference for subsequent recipes, preserving these exact folds and the declared feature-access policy. Agreement with two known submission recipes is supporting evidence; it does not guarantee that future CV gains will transfer to later Kaggle trials. See BENCHMARK.md for the comparison contract.
