# Part Two: model choices and tuning history

Our best scored submission is **[file 13](submissions/13_file10_legacy30.csv),
Kaggle log loss 0.58465**, with user-reported first place. It combines the
recency/neighbor extension of Ani's model with our earlier text/tree ensemble.
This guide explains how the experiments led to that recipe. The linked reports
retain the complete comparisons, unsuccessful trials, and reproduction commands.

## The submitted results

Lower log loss is better. These are reported Kaggle results for specific saved
CSVs, not scores for every subsequent refit of the same recipe. Ani's scores
come from the preserved source history; our numbered submission scores were
reported by the user. The [score ledger](submissions/leaderboard_results.json)
records their sources and exact artifact hashes.

| Model or file | Reported Kaggle loss | What we learned |
| --- | ---: | --- |
| Ani's original unweighted stack | 0.58810 | Strong reference for grouped CV and stacking |
| Ani's recency-weighted stack | 0.58688 | Better reported performance on the later test cohort |
| 03: phase-specialized ensemble | 0.59562 | Its local advantage did not transfer to Kaggle |
| 02: text-heavy ensemble | 0.58993 | Reducing tree influence helped, but still trailed Ani |
| 10: recency/neighbor blend | 0.58609 | First submitted improvement over Ani's recency model |
| 11: file 10 + 20% original ensemble | 0.58474 | The earlier ensemble contributed useful complementary predictions |
| **13: file 10 + 30% original ensemble** | **0.58465** | A smaller further gain; current scored winner |
| 15: original ensemble, recent tree, and phase trend added to file 10 | 0.58492 | Better local averages did not transfer on this upload |

File 16 was prepared but **not submitted**. Other files without confirmed scores
remain marked unknown in the detailed records.

## 1. Start with text, clinical context, and classical ensembles

The initial baseline used eligibility-text TF-IDF, phase indicators, and logistic
regression. It provided a simple way to share text signal across phases. We then
tested disease/drug text, clinical metadata, supplied drug annotations, molecular
features, recency weights, phase-specific fits, and tree ensembles. Trees could
capture interactions beyond the linear text model; probability blends let us
test how much tree influence helped.

The initial results also exposed reasons for caution: smaller phase-specific
fits could overfit, molecular fingerprints did not consistently help, and drug
annotation coverage was lower in test than training. Part 1 stopping-reason
analysis highlighted possible weaknesses in the eligibility-only baseline and
motivated testing broader clinical context. Those reason groups were small and
often inferred, so they supplied hypotheses rather than reliable subgroup
optimization targets.

File 03 had the best initial joint local loss but scored 0.59562 on Kaggle.
File 02 reduced tree fractions in Phases II/III and improved to 0.58993. That
supported testing less tree influence, but one aggregate score could not tell
us which phase or mechanism caused the improvement. Once Ani's source became
available, we paused the proposed tree-removal follow-ups and made a direct
comparison with his stronger model.

Evidence: [initial methods and diagnostics](README.md#methods-and-comparisons),
[original ensemble recipes](submissions/manifest.json), and
[submission review with superseding decisions](experiments/model_search/POST_SUBMISSION_REVIEW.md).

## 2. Reproduce both Ani models and establish a common benchmark

Ani's models combine eight base predictions using logistic stacking trained on
trial-grouped out-of-fold predictions. His second model adds exponential recency
weighting. Reproducing both models gave us working controls and revealed another
difference: our initial submissions fitted 8,037 training rows, while Ani's final
models fitted all 9,050 train + validation rows. The original leaderboard gap
therefore could not be attributed solely to model architecture.

We fixed the evaluation trials, grouping, seeds, time boundaries, and metric for
subsequent comparisons. The later all-submission run also refitted the complete
original ensembles, including their trees, on those same splits. File 10 beat
file 02 in all six chronological windows, while file 02 retained better supplied
validation and grouped-CV scores. This showed why a single local ranking was
insufficient for a test set drawn from later years.

Evidence: [benchmark methodology](LOCAL_TESTING.md),
[Ani source provenance](experiments/ani_comparison/provenance.json), and
[all-submission comparison](experiments/all_submissions/RESULTS.md).

## 3. Learn from the first unsuccessful tuning round

The next round evaluated 27 recipes on ten identical splits under two
preprocessing policies. Changes included slower recency, calibration and
regularization, phase interactions, blending the two Ani winners, capped year
effects, joint text features, and supervised word-ratio features.

Some changes improved grouped CV but weakened forward prediction. For example,
the unweighted text/trend variant improved strict grouped-CV loss to 0.577144,
but its chronological mean was 0.542480 versus 0.537446 for Ani's recency model.
No proposal passed that round's declared selection rule, so we retained Ani.
The failed recipes remain part of the record.

Evidence: [hypotheses and original selection rule](IMPROVEMENT_PROTOCOL.md),
[complete findings](experiments/stack_improvements/RESULTS.md), and
[executed notebook](stack_improvements.ipynb).

## 4. Address recency and the neighbor component: file 10

The first round had mostly changed stacking/calibration and tested slower
recency. Inspecting Ani's neighbor component exposed more specific opportunities:
it ignored recency, used linear eligibility-text similarity, and shrank estimates
toward a fixed 50% success prior. We tested stronger recency, sharper similarity,
phase-specific priors, different neighborhood sizes, and clinical-context
similarity.

Before scoring the new batch, we revised the selection rule to prioritize
chronological performance against Ani's recency model, with limits on CV and
worst-window regressions. Requiring near-best random CV against the unweighted
model had been too restrictive for the later-cohort objective. The old rule and
results remain recorded; this was an explicit adaptive change in research direction.

File 09 replaced the original neighbor estimate. **File 10** instead averaged
Ani's recency stack with a faster-recency stack (`tau=2`) receiving eight extra
neighborhood estimates. Its fixed 50/50 blend improved all three forward windows
and all three longer-gap windows against Ani's recency model under both
preprocessing policies. Grouped CV improved slightly against that model;
supplied-validation loss worsened. Kaggle then improved to **0.58609**.

Evidence: [extension hypotheses and revised rule](EXTENSION_PROTOCOL.md),
[mechanisms, comparisons, and trade-offs](experiments/ani_extensions/RESULTS.md),
and [executed notebook](ani_extensions.ipynb).

## 5. Test complementary predictions: files 11 and 13

The common benchmark showed that our earlier ensemble retained strengths in
validation and grouped CV even though it was weaker on later cohorts. This
suggested testing it as a contributor to file 10. The subsequent batch evaluated
32 blend/meta-model proposals on the same ten splits plus three gap checks;
a separate batch tested four recent-tree mixtures.

The 30% original-ensemble blend had stronger local averages, but its worst-gap
and latest-window trade-offs favored trying **20% first**. File 11 used 80%
file 10 + 20% original ensemble and scored **0.58474**. This supported trying the
already-tested 30% weight. File 13 scored **0.58465**, a further 0.00009 gain.
The small increment argued for cautious refinement rather than extrapolating
that more of the original ensemble would keep helping.

Evidence: [32-proposal batch and the 20% selection rationale](experiments/leaderboard_sprint/runs/4dc0f274d45bce9e/RESULTS.md),
[recent-tree comparisons](experiments/leaderboard_sprint/recent_tree/dea847f57c7c07ee/RESULTS.md),
and [how Kaggle feedback changed the decisions](experiments/leaderboard_sprint/RESULTS.md#what-the-feedback-changed).

## 6. Test further refinements and retain file 13

We evaluated 19 further mixtures and nine final-slot alternatives. File 15
combined 60% file 10, 20% original ensemble, 10% recent tree, and 10% Ani phase
trend. It had favorable local chronological averages within the declared
worst-gap limit, but scored **0.58492** on Kaggle, 0.00027 worse than file 13.
That result weakened the case for increasing the tree/trend additions solely
because historical averages favored them.

File 16 retained the successful 30% original-ensemble weight and added a
sharper-neighbor contribution. Its local gains were small, uncertainty intervals
included zero, and some other metrics regressed. With first place reported and
one submission slot left, the final decision was to hold it. File 13 remains
the best scored artifact; no Kaggle improvement is claimed for file 16.

Evidence: [refinements and submission decision](experiments/leaderboard_sprint/refine/2e5c792462013673/RESULTS.md),
[final-slot alternatives](experiments/leaderboard_sprint/finish/b235582538f63f9c/RESULTS.md),
and [overall sprint comparison](experiments/leaderboard_sprint/RESULTS.md).

## The final recipe and data boundaries

File 13's probability blend is:

```text
P13 = 0.70 * P10 + 0.30 * P_original_ensemble
P10 = 0.50 * P_Ani_recency + 0.50 * P_fast_recency_with_neighbors
```

The original ensemble is 50% a three-logistic-model mixture, 25% depth-6
CatBoost, and 25% depth-8 CatBoost. Its contribution was refitted on all 9,050
labeled rows for the final submission; we did not simply reuse the old file 01
predictions fitted on 8,037 rows. Exact weights and artifact provenance are in
[file 13's submission record](submissions/records/13_kaggle_0.58465.json), the
[original recipe manifest](submissions/manifest.json), and the
[export receipt](experiments/leaderboard_sprint/runs/4dc0f274d45bce9e/exports/13_file10_legacy30.json).

| Stage | Data use |
| --- | --- |
| Official validation comparison | Fit 8,037 training rows; score the separate 1,013 validation rows |
| Additional grouped CV | Pool the 9,050 labeled rows, then use two repeats of trial-grouped three-fold CV; each scored trial is excluded from that fold's supervised training |
| Chronological comparisons | Use earlier-year rows from the same pool to predict later-year rows; three original windows plus three longer-gap checks |
| Strict benchmark preprocessing | Fit vocabulary, IDF, and other learned transformations only on each fitting population, including inside stacking folds |
| Published-recipe comparison | Preserve Ani's vocabulary/IDF fitting on supplied train, validation, and unlabeled test inputs; original ensemble preprocessing stays local to its fitting rows |
| Final submission fitting | Refit supervised components on all 9,050 labeled rows after selecting the recipe; use the published component preprocessing policies |

Vocabulary/IDF fitting uses text to determine the feature vocabulary and how
common each word is. In the Ani-derived final components, test text therefore
influences the feature representation. **Test outcome labels are unavailable
and never used.** That preprocessing choice is distinct from fitting the
success/failure classifiers. Trial IDs keep combined-phase rows together and
match outputs; they are not predictive features.

The model families are classical ML. Inputs come from the supplied datasets,
including their drug annotations; no external outcome dataset was retrieved.
Part 1 reason labels and embeddings are excluded from prediction inputs. Reason
tags are joined after outcome prediction for error analysis, preferring known
tags and otherwise using frozen Part 1 predictions. Thus Part 1 can inform
modeling hypotheses without supplying the labels used to score Part 2.

Details: [saved split definitions](experiments/local_benchmark/protocol.json),
[preprocessing policies](ALL_SUBMISSIONS_PROTOCOL.md), and
[stopping-reason analysis](README.md#stopping-reason-error-analysis).

## Interpreting and reproducing the evidence

The primary local metric is mean log loss across the three phases. We considered
chronological averages, individual adverse periods, grouped CV, and both
preprocessing policies together. Per-phase F1, ROC-AUC, PR-AUC, and reason-group
errors remain diagnostic views; no candidate won every metric.

Historical labels were repeatedly consulted during tuning. Paired bootstrap
intervals account for shared trials across phases but do not correct for
adaptive model selection. Gap checks reuse previously consulted labels, and
Kaggle feedback also informed later proposals. These results describe the
evidence used to choose the submission; they are not untouched estimates for
the full research process. Final train + validation refits are kept separate
from the held-out benchmark scores.

For the full record, start with the [benchmark instructions](LOCAL_TESTING.md),
[all-submission notebook](all_submissions.ipynb), and
[sprint reproduction commands](experiments/leaderboard_sprint/RESULTS.md#verification-and-reproduction).
The [cache guide](CACHE_REPRODUCTION.md) explains which fitted artifacts must be
rebuilt in a fresh checkout. The [trial commits and scored tags](submissions/HISTORY.md)
and [main-integration mapping](MAIN_INTEGRATION.md) connect this narrative to
the preserved Git history. Score tags exist only for confirmed submissions;
failed experiments and unsubmitted reserves remain available in their phase records.
