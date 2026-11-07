# Comparing both Ani winners and new Part Two proposals

This experiment extends the existing ten-split protocol without changing its
data, IDs, seeds, time boundaries, or phase-balanced log-loss metric. The frozen
registry in `improvement_models.py` contains two references and eleven proposals
or controls. `improvement_benchmark.py` records the entire registry, source
hashes, package versions and protocol hash before it fits a model.

## References and proposals

The first Ani winner (`628088a`, reported Kaggle 0.58810) is the unweighted
eight-model stack. The second (`1b8bd11`, 0.58688) uses exponential recency
weighting with tau=3 in base learners and stacking. Both use trial-grouped
five-fold out-of-fold predictions. Kaggle results come from their Git commit
messages and saved submission files; they are not newly queried leaderboard
results. The existing Phase III trend alternative is not the first winner.

Proposals test moderate recency (tau=6), slower recency in the meta-model only,
equal total phase weight in the meta-model, regularized phase interactions,
removing the overconfident naive Bayes input, learning a stack from both recency
scales, and a fixed equal blend of the winners. Ani's Phase III year-trend
alternative remains a control. No test labels or Part One inputs are used.

## Comparability

The strict track learns every vocabulary and IDF exclusively from the outer
training population, and refits them inside all stacking folds. Each held-out
trial is absent from its corresponding fitting population. The published track
separately reproduces Ani's feature-only transduction: all supplied covariates
shape the vocabulary/IDF, including test text, while evaluation outcomes remain
excluded. All proposals in that track receive the same preprocessing policy.
Results from these two tracks must not be mixed in one ranking.

All thirteen recipes are scored on all ten saved splits. There is no pruning
based on supplied validation, favorable folds, or per-phase subgroups. Every
prediction is saved, allowing identical trial-level paired comparisons with
both Ani references. Chronological means weight each of the three windows and
each phase equally; repeated CV pools all out-of-fold rows within each repeat.

## Selection rule fixed before first scores

An eligible proposal must improve chronological mean versus both references,
remain within 0.001 log loss of the better reference in grouped CV, and have
worst chronological loss no more than 0.003 above the better reference's worst
loss. Rank eligible proposals by chronological mean; carry at most two forward.
These tolerances are practical screening rules, not statistical significance.
If no proposal qualifies, retain Ani and report the mixed or negative result.
Supplied validation is a diagnostic and cannot override this rule.

Before final submission-file selection, check the frozen shortlist on additional
forward tests: fit through 2005/evaluate 2008–2009, fit through 2007/evaluate
2010–2011, and fit through 2009/evaluate 2012–2013. These intentionally leave a
gap to probe transport beyond adjacent years. Reuse the original grouped CV as
corroboration. Additional time windows reuse previously consulted labels, so
they are robustness checks, not an untouched external test. Do not use their
results to launch another parameter sweep in this experiment.

Pair bootstrap differences by trial, sharing resampling weights across phases
and repeats. Report intervals versus both references and include zero when it
falls inside them. Intervals are conditional on fitted models and exploratory
selection; they do not correct for the search or predict performance in
2014–2021.

## Final fit and reproduction

Freeze selection before fitting all 9,050 labeled rows. Final models use the
published preprocessing policy consistently. Reproduce *both* original Kaggle
files and report maximum probability differences with package versions. Keep
source files intact; rebuilt files do not inherit their scores. Export new
files in sample-submission order with exactly 1,500 unique IDs, finite
probabilities, and `ID,label` columns. Kaggle scores for new files remain
unknown until actually submitted and reported.

Commands from the repository root:

```sh
.venv/bin/python part2/improvement_benchmark.py --track strict
.venv/bin/python part2/improvement_benchmark.py --track published
.venv/bin/python part2/improvement_benchmark.py --final
```

`experiments/stack_improvements/current_run.json` identifies the source-keyed
run. Original benchmarks and submission files are preserved. The local virtual
environment and regenerable native caches are ignored by Git; saved base
prediction arrays, fitting IDs, and scored predictions are retained.

## Declared adaptive follow-up

After the first batch's chronological results and early grouped folds showed
opposing recency/random-CV rankings, a separate eight-recipe batch was frozen
in `improvement_trends.py` and `trend_followup/manifest.json`. It tests explicit
capped Phase III or all-phase year effects on the unweighted, tau=6 and tau=3
bases, with fixed full or quarter-strength year features. The year cap is the
same training-maximum-plus-two rule used by Ani. The first batch is unchanged.

This is adaptive exploration on previously consulted data. It does not receive
an untouched-test interpretation. All 21 recipes use the same ten splits, and
the selection tolerances above remain unchanged. Freeze at most two proposals
from the combined tables before the gapped-window checks and final fitting:

```sh
.venv/bin/python part2/improvement_trends.py declare
.venv/bin/python part2/improvement_trends.py run
.venv/bin/python part2/improvement_trends.py select
.venv/bin/python part2/improvement_report.py compare
.venv/bin/python part2/improvement_robustness.py --track strict
.venv/bin/python part2/improvement_robustness.py --track published
.venv/bin/python part2/improvement_benchmark.py --final
.venv/bin/python part2/improvement_trends.py final
.venv/bin/python part2/improvement_report.py export
```

## Final adaptive text batch

Neither of the first two batches passed the fixed rule. The final batch in
`improvement_text.py` adds a joint logistic model over all clinical/text blocks
and a binary word-count logistic model using supervised log-count ratios.
The ratios are re-learned inside every inner fold, with no held-out outcomes.
Six fixed stacks add these predictions to the unweighted or recency-weighted
Ani bases, optionally retaining his existing Phase III trend. There are now
27 recipes in total: two references, the trend control, and 24 other variants.
This batch is explicitly adaptive; the same selection rule remains in force,
and no more search is planned after the robustness checks.

Run both text tracks after the preceding batches, then freeze the combined
selection with `improvement_text.py select` (instead of the earlier selection
commands). The gapped tests run the final shortlist and both references only.
If text candidates are selected, run `improvement_text.py final` after the
base final fit and before export. If nothing qualifies, produce no new
submission and report the failed proposals with their complete scores.

```sh
.venv/bin/python part2/improvement_text.py run --track strict
.venv/bin/python part2/improvement_text.py run --track published
.venv/bin/python part2/improvement_text.py select
```
