# Part 1 testing policy

This is the current contract for apples-to-apples Part 1 comparisons. It preserves the saved grouped folds and rules from the [archived policy](research/lda_ensemble/TESTING_POLICY.md), using paths that work in this repository. Archived references to older leaders, worktrees, or submission quotas describe their original checkpoint. [Current model and results](README.md).

## Required checks

| Check | Population and fitting access | Required evaluation |
|---|---|---|
| Train157 CV | 157 original training labels, 150 embedding groups; all 78 validation rows excluded | Eight repeats × five folds: 40 fits |
| Pooled235 CV | 235 training + validation labels, 222 embedding groups; held-out rows excluded within each fold | Eight repeats × five folds: 40 fits |
| Original validation | Fit on 157 gold + all 1,014 originally masked training rows; predict the 78 validation rows | One separate fit |
| Chronology | Fit on earlier eligible gold and masked rows; predict later gold rows | Three fixed windows, 98 held-out predictions total |

A full finalist comparison has **80 CV fits plus four supplementary fits per recipe**. An initial three-repeat screen is 15 fits per pool and is not a full comparison. There is no additional “CV on validation”; Pooled235 includes validation-source rows in the appropriate fold partitions. The final submission refit is separate from these checks.

### Exact saved assignments

Use these files, including their saved IDs and fold assignments. Do not regenerate splits from seeds:

| Pool | Initial three repeats | Additional five repeats |
|---|---|---|
| Train157 | [train3.json](experiments/current_leaders_grouped_v1/snapshots/train3.json) | [train5.json](experiments/current_leaders_grouped_v1/snapshots/train5.json) |
| Pooled235 | [pooled3.json](experiments/current_leaders_grouped_v1/snapshots/pooled3.json) | [pooled5.json](experiments/current_leaders_grouped_v1/snapshots/pooled5.json) |

Chronology uses `chronological_splits` in [chronology.json](experiments/current_leaders_grouped_v1/snapshots/chronology.json). The five files are byte-identical to the copies used by the replay adapter under [research/new_directions/reproduction/protocols](research/new_directions/reproduction/protocols/). Original provenance is retained in the archived policy and each JSON.

Both CV pools use iterative multilabel stratification over embedding groups. Groups hash all 150 reason-embedding coordinates rounded to ten decimals using the frozen helper. Initial seeds are **20261126, 20261127, 20261128**; confirmation seeds are **20261129, 20261130, 20261201, 20261202, 20261203**. Retain every saved repeat, including unfavorable ones.

## Fitting boundaries

1. Before scoring, record a versioned recipe, hypothesis, exact parent/control, features, settings, random seeds, ensemble weights, decoder, and advancement rule. Fix the procedure across folds; refit its learned quantities inside each fold.
2. Fit scaling, vocabulary, imputation, PCA, feature selection, clustering, cluster-to-label mapping, pseudo-labels, classifiers, and calibration only on eligible fitting inputs. Learned thresholds, gates, or hyperparameter selection require inner CV confined to the outer training partition, with the same group boundaries.
3. In grouped CV and chronology, remove originally masked training rows whose embedding group matches a held-out group. Held-out gold rows cannot be relabeled as unknown training rows. Only the fitting gold labels may inform label-dependent operations.
4. Default access is **inductive**: held-out and Kaggle test inputs enter only prediction. Train157 excludes validation-source rows entirely; Pooled235 may fit on validation-source rows assigned to its training partition.
5. Use only supplied Part 1 inputs. Preserve gold labels. No external trial records/labels, deep learning, ID predictors, or test-label reverse engineering. IDs serve alignment and audits. Preserve the assignment's unsupervised missing-label-generation requirement for the course deliverable; see the [assignment PDF](../Midterm_Project.pdf).

The original validation check intentionally retains all original training rows, including any training embeddings matching validation embeddings. It excludes the validation rows themselves from fitting but does **not** apply the grouped CV purge. This preserves the historical check; label any purged-validation diagnostic separately.

Transductive experiments require an explicitly separate access description and results. The previously completed diagnostic does not change the default inductive contract or establish course-staff permission for submission.

## Metrics and comparison

Compute **sample-wise F1 on the complete out-of-fold predictions within each repeat**, then average the repeat scores. Do not average fold F1 values. Use the registered threshold and nonempty-label fallback without selecting them on the outer predictions being evaluated.

For each pool, report all repeat scores, their mean and **population SD (`ddof=0`)**, paired per-repeat differences and wins/ties/losses against the declared parent, macro F1, and each label's precision, recall, F1, and support. Overlapping repeats are descriptive evidence, not independent significance tests. Never rank three-repeat and eight-repeat results together.

For new attempts to improve the submitted model, declare the **anchored 0.89636 recipe** as the parent. Keep current Ani LDA and the LDA ensemble as useful controls, alongside original Ani's inductive reference when relevant. A study of a different parent must identify that parent before scoring. Original Ani's reported .89423 is a different transductive artifact and cannot be attached to a newer source or inductive control.

## Advancement rules

- Screen a small, predeclared set on the first three repeats in **both** pools. Default advancement requires a positive mean paired difference and at least two wins out of three in each pool.
- Confirm shortlisted recipes on the five additional repeats and report all eight. Default confirmation requires positive mean paired differences and at least five wins out of eight in **both** pools. A focused comparison may predeclare all eight from the start.
- Run original validation and chronology after shortlisting. Reuse a prior measurement only when the exact recipe and fitting access match and its source is recorded. Declare diagnostic exceptions before supplementary evaluation.
- Keep supplementary regressions visible. Do not average the four checks into one arbitrary score or promote a tiny CV gain automatically. A higher reported Kaggle score does not imply a win on every local check.

These labels and splits have already influenced development. Reusing them gives consistent comparisons, not a new untouched holdout. Record the full search history and negative results.

## Chronological windows

Use the saved IDs and group purges, not a new search over years:

| Fitting years | Held-out years | Fitting gold | Fitting masked | Held-out gold |
|---|---|---:|---:|---:|
| Through 2004 | 2005–2006 | 136 | 151 | 39 |
| Through 2006 | 2007 | 173 | 458 | 22 |
| Through 2007 | 2008–2010 | 194 | 626 | 37 |

Fit every step on earlier eligible inputs. Report each window and sample-wise F1 pooled over all 98 held-out predictions. Start year is a proxy for distribution shift, not known label-availability time; one window has no positive Funding labels. These are small supplementary checks.

## Evidence and reproducibility

Record source/config/decoder/data hashes, saved protocol hashes, feature access, fitting populations, IDs, fold coverage, and prediction locations. Independently rescore retained predictions and check group separation, input boundaries, and protected hashes. Keep detailed predictions for leaders and useful matched controls. After checking summaries, unsuccessful studies may retain code, fixed settings, failures, and summaries without bulky fold caches; follow the [retention policy](research/lda_ensemble/EVIDENCE.html).

The [Part 1 README](README.md#testing-and-verification) gives working verification and replay commands. Verification rescoring does **not** count as fresh model fitting. A replay of one study does not establish that every competitor was refitted. Version changed recipes or runners and rerun matched controls; do not edit frozen source registries or overwrite saved studies.

## Historical protocols and final refits

| Historical procedure | Saved splits | Difference from current grouped CV |
|---|---|---|
| Train157 classifier CV | [classifier_cross_validation/splits](experiments/classifier_cross_validation/splits/) | Five repeats × three folds, seeds 42–46, no embedding-group purge |
| Pooled235 paired recipe CV | [paired_recipe_cv/splits](experiments/paired_recipe_cv/splits/) | Eight repeats × five ordinary shuffled KFold splits, seeds 100–107, no embedding-group purge |

Their scores retain meaning within those protocols. They cannot be compared directly to the current grouped means. Historical reports also use sample repeat SD in places; current reporting uses population SD. Older chronological cutoffs are separate diagnostics.

A final **235-label refit** uses the selected fixed recipe and relearns transformations, clusters, pseudo-labels, and classifiers on the enlarged fitting pool. The earlier 78-row validation score belongs to the 157-label fit and is not independent validation of this final artifact. Record exact CSV bytes, SHA-256, source/recipe, fitting population/access, and score provenance in the [submission guide](submissions/README.md), following [AGENTS.md](AGENTS.md). Uploads are manual; historical quota statements are not a current submission budget.
