# Anchored model: target-input access

Neither of the two fixed transductive changes met the local advancement criteria against the **anchored 0.89636** parent. The evidence is insufficient to replace the submitted inductive model. Their Kaggle performance remains unknown: this study made no new submission or final test-set fit.

| Fixed recipe | Train157 CV | Pooled235 CV | Validation78 | Chronology98 |
|---|---:|---:|---:|---:|
| Anchored parent, inductive | 0.932059 | 0.921365 | 0.891026 | 0.884354 |
| Target inputs for scaling/PCA/GMM only | 0.933652 | 0.921897 | 0.884615 | 0.875850 |
| Target inputs for geometry and generated labels | 0.931263 | 0.922074 | 0.880342 | 0.879252 |

Both candidates use **transductive access** to the current evaluation batch's embeddings. The parent remains inductive. The rows share the exact saved labels, folds, fitting-row order, group purges and decoder, but represent different input-access regimes. Target truth never enters the model interface. Actual Kaggle test inputs were protected by a hash and were not loaded for fitting, tuning or prediction.

## Hypothesis and controlled change

The [original Ani notebook](../lda_ensemble/research/ani_snapshots/a548721/part1_best_model.ipynb) fits scaling, PCA and GMM on all supplied embeddings, and allows test rows into confidence-filtered self-training. The [older ungrouped benchmark](../../experiments/paired_recipe_cv/BENCHMARK.md) reported an Ani gain of 0.005408 from combined transductive access, on a different recipe and protocol.

This study isolates those two mechanisms on the newer leader. The target batch is appended after the original eligible fitting rows. The geometry-only arm lets it affect StandardScaler, PCA30 and diagonal GMM30, but never adds generated target labels to classifiers or neighbors. The full arm appends targets with all five labels set to -100 and calls the **unchanged** anchored fit method. Only targets passing its existing 0.8 confidence filter receive generated training labels. All features, weights, seeds, C, priors, rounds and 0.3-plus-argmax decoder are unchanged.

The [assignment PDF](../../../Midterm_Project.pdf) describes mostly early visible labels and a later test population, so target-input adaptation was a reasonable hypothesis. It does not guarantee that more generated labels are useful. It also does not explicitly settle transductive submission permission; no upload or claim of course-staff approval is made here.

Ani's historical CV fitted geometry globally and included actual Kaggle embeddings. This experiment instead refits every fold and uses that fold's held-out batch as a proxy for deployment targets. In chronology, later target embeddings are intentionally available to transductive models; those rows are not ordinary future-input-excluded forecasts. Evaluation batches are also smaller than the final 312-row test batch.

## Complete evaluation and decision

The [protocol](protocol.json) and model source were committed as `095288d` before scoring. All three recipes completed **40 Train157 fits, 40 Pooled235 fits, original validation and three chronological windows**: 252 scored fits plus one implementation-equivalence check. All eight saved repeats were predeclared; none were selected or omitted. Supplementary evaluation of both candidates regardless of CV outcome was also declared in advance.

| Candidate | Train157 paired delta; W/T/L | Pooled235 paired delta; W/T/L | Meets both-pool rule? |
|---|---|---|---|
| Geometry only | +0.001592; 4/3/1 | +0.000532; 4/1/3 | No |
| Geometry + target generated labels | -0.000796; 1/5/2 | +0.000709; 4/0/4 | No |

The fixed advancement rule required positive mean differences and at least five wins in **each** eight-repeat pool. Neither qualifies. Both also lose on original validation and pooled chronology. The full arm improves the first and last chronological windows, but regresses strongly in the middle window; every window is retained in the results.

Geometry-only changes just 6 of 1,256 Train157 repeat-level label sets and 19 of 1,880 Pooled235 label sets. Its small CV gains do not survive the supplementary checks. Full target self-training accepts 1,025 of 1,256 Train157 target appearances and 1,521 of 1,880 Pooled235 appearances, but does not consistently improve accuracy. These are repeated appearances, not independent trials or significance tests.

The parent reproduced **all 84 saved prediction sets**, with exact score agreement on all 80 CV fits. The geometry implementation with an empty target batch matches the parent within 2.96e-13. An additional independent TP/FP/FN calculation checked all 1,386 reported metric values before prediction caches were removed. No model settings were changed after observing results. The development folds have been reused before; no untouched holdout or new Kaggle improvement is claimed.

## CV and Kaggle ordering

Better CV has not always translated into better reported Kaggle performance. The [existing matched comparison](../../experiments/current_leaders_grouped_v1/comparison.csv) contains these three inductive recipes, whose submitted fits all used 235 verified labels:

| Recipe and scored checkpoint | Train157 CV | Pooled235 CV | Reported Kaggle F1 |
|---|---:|---:|---:|
| [Earlier 70/15/15 ensemble](../../submissions/results/three-way-trainval235.json) | 0.925690 | 0.916489 | 0.89401 |
| [LDA 70/15/15 ensemble](../../submissions/results/lda-ensemble-trainval235.json) | 0.931794 | 0.921631 | 0.89401 |
| [Anchored](../../submissions/results/anchored-trainval235.json) | 0.932059 | 0.921365 | 0.89636 |

The LDA revision improved Train157 CV by 0.006104 and Pooled235 CV by 0.005142, but its reported Kaggle score tied the earlier ensemble. Anchored was effectively tied with LDA on CV: Train157 +0.000265 and Pooled235 -0.000266. Nevertheless, anchored improved Kaggle by 0.00235. Its original-validation F1 was also lower, 0.891026 versus 0.905983; its chronology F1 was higher, 0.884354 versus 0.877551.

These are concrete exceptions to a universal ordering. This small set of submitted comparisons cannot establish how reliably CV will rank future candidates. The limited labelled pool, repeated development on the same folds, and later-year Kaggle population are plausible reasons for imperfect agreement. Original Ani's reported 0.89423 is excluded from this clean comparison because its submission was transductive while its row in the common local comparison is an inductive adaptation.

For this study, the conclusion is **insufficient local evidence to promote either transductive variant**. We have not shown that either variant would score worse on Kaggle, and a leaderboard improvement remains possible. The small geometry-only CV gains and supplementary regressions inform the decision; they do not measure its unobserved Kaggle result. The predeclared advancement rule, scores and current leader are unchanged. Any later exploratory final fit or submission should be recorded separately with its exact recipe, input access and artifact.

This interpretation reuses the existing comparison and scored checkpoints. No new fits, score records, prediction copies or evidence archives were created for it.

## Records and reproduction

- [All repeat and per-label metrics](results/repeat_scores.csv), [CV means and population SD](results/cv_summary.csv), [paired summaries](results/paired_summary.csv), and [all repeat differences](results/paired_repeat_deltas.csv).
- [Every validation/chronology result and per-label metric](results/supplementary_scores.csv), [prediction-change counts](results/prediction_change_summary.csv), and [fitting-access counts](results/fit_access_summary.json).
- [Full-run checks](results/verification.json), [independent rescoring record](results/independent_rescore.json), and [retention manifest](retention.json).
- [Access ablations](models.py) and [registered runner](run.py). The frozen parent implementation remains [anchored_solution.py](../../anchored_solution.py).

From the repository root, verify retained evidence without fitting:

```sh
.venv/bin/python part1/research/anchored_transductive_v1/verify_retained.py
```

Replay the complete registered comparison into a new disposable directory, using the existing [anchored environment](../../requirements-simple.txt):

```sh
.venv/bin/python part1/research/anchored_transductive_v1/run.py run \
  --output /tmp/anchored-transductive-replay --workers 4
.venv/bin/python part1/research/anchored_transductive_v1/run.py verify \
  --output /tmp/anchored-transductive-replay
```

The run automatically checks frozen parent predictions and rescores complete out-of-fold results. The second command repeats verification without refitting. Registration is already frozen; do not run `register` over it. Use a new study version for any different setting or access policy.

Following [RETENTION.md](../../RETENTION.md), this unsuccessful experiment keeps roughly 37 KB of results plus its recipe and source. Its 84 raw prediction packets and detailed per-fit audits were removed after verification. `verify_retained.py` checks hashes, aggregate consistency and promotion rules; it cannot rescore discarded trial-level predictions. The existing detailed parent/control evidence remains at its original path.
