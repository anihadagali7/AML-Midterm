# Central Part 1 testing policy

This is the default for new Part 1 comparisons across branches. It combines the agreed train-only CV, pooled CV, original validation, and chronological stress test. Historical experiment contracts and scores remain frozen. The earlier [235-row KFold benchmark](experiments/paired_recipe_cv/BENCHMARK.md) is a separate historical protocol; its scores must not be mixed with the grouped scores here.

## Fixed evaluation surface

All paths below are relative to [research/independent](research/independent/). Use the saved assignments, not newly generated splits with the same seed.

| Check | Known-label population | Frozen assignments | Role |
|---|---|---|---|
| Train CV | 157 original training labels, 150 embedding groups | `experiments/benchmark_v1/protocol.json` plus `experiments/cross_repo_benchmark_v1/confirmation8/train/protocol.json` | Primary comparison |
| Pooled CV | 235 train + validation labels, 222 embedding groups | `experiments/pooled_benchmark_v1/protocol.json` plus `experiments/cross_repo_benchmark_v1/confirmation8/pooled/protocol.json` | Required complementary comparison for finalists |
| Original validation | Fit on 157 training labels plus 1,014 original masked training rows; predict 78 validation rows | Original provided files | Secondary check after CV shortlisting |
| Chronological | Fit on earlier trials; predict later labeled trials | `experiments/cross_repo_benchmark_v1/protocol.json`, `chronological_splits` | Distribution-shift stress test |

Both CV pools use five folds per repeat, with iterative multilabel stratification over groups. A group hashes all 150 reason-embedding coordinates rounded to ten decimals. Initial seeds are **20261126, 20261127, 20261128**. Confirmation adds **20261129, 20261130, 20261201, 20261202, 20261203**. A finalist has eight repeats / 40 fits per pool. No seed substitution, favorable-fold deletion, or mixing three-repeat and eight-repeat rankings.

There is no separate “CV on validation.” Pooled CV includes validation-source rows; each row's label can train a fold only when that row is in its training partition. This is additional testing of the recipe on reused data, not eight independent datasets or a larger independent test set.

## Fitting boundaries

1. Freeze a named, versioned recipe, source hashes, features, hyperparameters, preprocessing, pseudo-label procedure, random seeds, ensemble weights, and final decision rule before scoring. Learned coefficients, clusters, transforms, and generated labels are refitted in each fold; the recipe stays fixed.
2. Fit every learned step inside the outer training partition: vocabulary, imputation, scaling, PCA, feature selection, clustering, cluster-label mapping, calibration, pseudo-labeling, and classifiers. Any adaptive threshold, feature choice, ensemble gate, or setting selection must use inner CV confined to that partition.
3. In grouped CV and chronology, allow original masked training rows only after removing rows whose embedding group matches the held-out groups. Held-out gold rows cannot become pseudo-labeled training rows. Use only fitting gold labels for all label-dependent steps.
4. The default is **inductive**: held-out and Kaggle test inputs enter only prediction. Pooled folds may use validation-source inputs in their training partition. Standalone validation fits exclude validation inputs entirely.
5. Keep gold labels unchanged. No deep learning, external trial records/labels, ID predictors, or reverse engineering of test labels. Use provided embeddings and metadata. IDs are for alignment, splits, and audit only. Preserve the assignment's unsupervised missing-label-generation requirement when preparing the final course deliverable.

The original standalone validation check preserves the provided train/validation boundary: all original training rows remain eligible, including training rows whose embeddings happen to match validation rows. It does not apply the CV group purge or fit on the validation rows themselves. This keeps prior validation measurements comparable, but makes that check less strict about duplicate patterns than grouped CV. A purged-validation diagnostic would need a separate name and score.

## Metric and reporting

Compute **sample-wise F1** for the complete out-of-fold prediction vector within each repeat, then average the repeat scores. Each trial has equal weight. Use the registered decoder, including its threshold and any nonempty-label fallback; never choose the decoder from the outer predictions being scored.

Report both pools' mean, population SD across repeats, all repeat scores, and paired per-repeat deltas/wins/ties against the same frozen control. Include macro F1 and each label's precision, recall, F1, and support for error analysis. SD and repeat wins are descriptive; overlapping folds do not provide independent significance tests. Do not substitute a mean of fold F1 values for the complete-OOF metric.

Each record must identify its protocol, recipe/config/decoder hashes, code and raw-data hashes, label pool, feature-access regime, exact folds, fitting population, and saved prediction artifact. Save failures and negative results as well as successes. Independently rescore saved predictions and check row IDs, fold coverage, group separation, input boundaries, and unchanged protected files.

## Selection and validation use

- Screen a small, predeclared hypothesis set on the initial three repeats in both pools. Default advancement requires positive mean paired improvement and wins in at least two of three repeats in each pool.
- Confirm shortlisted recipes on the five additional saved repeats; report all eight together. Default confirmation requires positive paired means and at least five of eight wins in **both** pools against the declared parent/control. A focused comparison may predeclare all eight from the start. Record diagnostic exceptions before their supplementary evaluation.
- Prioritize consistent evidence across the two CV pools. Evaluate original validation and chronology after shortlisting, or reuse a prior measurement when the exact recipe and fitting access are unchanged and its source is recorded. Supplementary regressions remain visible; a tiny CV gain does not automatically promote a model.
- Keep Ani's inductive control and the submitted 70/15/15 ensemble as references. For a modification, compare to its exact parent too. Do not choose a different comparator after seeing scores. The [handoff](RESEARCH_HANDOFF.md) identifies the current evidence and score sources.
- Do not combine the four checks into an arbitrary single average. Chronology uses a different population; validation is one repeatedly consulted split. Record tradeoffs explicitly.

All these labels and folds have already influenced development. Freezing a protocol now does not restore an untouched holdout. Repeated CV tuning can also overfit. Neither a benchmark win nor chronological agreement guarantees a Kaggle improvement.

## Chronological stress test

Use the exact saved IDs and embedding-group purges, not a new year search:

| Fitting years | Held-out years | Fitting gold | Fitting masked | Held-out gold |
|---|---|---:|---:|---:|
| Through 2004 | 2005–2006 | 136 | 151 | 39 |
| Through 2006 | 2007 | 173 | 458 | 22 |
| Through 2007 | 2008–2010 | 194 | 626 | 37 |

Only earlier eligible inputs fit the inductive model, including unsupervised steps. Report each window and sample-wise F1 pooled over all 98 held-out predictions. Start year is a proxy for temporal shift, not known label-availability time. One window has no positive Funding labels. These small windows are supplementary and have not reliably predicted all submission rankings.

## Transduction, refitting, and submissions

Transductive experiments are a separate, explicitly labeled track. The user authorized the completed `three_way_transductive_v1` diagnostic. In that study only the current evaluation batch's **inputs** were added with every target label erased; local CV did not fit on actual Kaggle test inputs. Record exactly which components can use query inputs and pseudo-label them. Compare on identical folds, but do not present the score as ordinary inductive generalization. This experiment does not establish course-staff permission for the final submission or change the default policy.

A final 235-label refit is a separate artifact using the frozen recipe. It relearns coefficients, transformations, clusters, and pseudo-labels from the expanded fitting data. Earlier standalone validation F1 belongs to the 157-label fit and cannot be claimed as an independent score of the 235-label model. The two recorded local submissions were explicitly authorized and refitted on 235 gold labels. Further refits/uploads follow the user's requested scope; a CV win alone is not a submission instruction.

Use [submissions/index.csv](submissions/index.csv) for the exact CSV, SHA-256, fitting-label count, reported score, and provenance. Preserve historical files and tags. Uploads are manual; no automatic Kaggle submission. At this integration checkpoint the user reported two remaining submissions. Verify any later budget change before spending one. Kaggle scores are limited development feedback, not a target for blind repeated probing.

## Adding the next experiment

Read [the research README](research/independent/README.md), then create a new versioned study directory and recipe registry. Reuse the four saved protocol files and chronological IDs above. Existing runners expose the folds and context helpers; the completed study runners are examples. **Do not append to the frozen base registries or edit hashed source modules in place**: later study guards depend on them. Add a new adapter/runner, or explicitly version a changed engine and rerun matched controls.

Record the hypothesis and screening rule before fitting, preserve all out-of-fold scores and audit results, and update the central handoff only after the comparison is complete. Historical notebooks with validation-driven searches remain useful context; they do not define the current selection procedure.
