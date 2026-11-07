# Repeatable Part Two local benchmark

The [all-submission comparison](experiments/all_submissions/RESULTS.md) evaluates
the exact recipes for files 01–10, both Ani winners, and the trend control on all
ten splits below plus the three saved gap checks. It replaces linear-component
proxies with the full original ensembles. Run `make -C part2 compare-all`, then
`compare-all-check` and `compare-all-report`, after installing
`requirements-all-submissions.txt`. [Policy](ALL_SUBMISSIONS_PROTOCOL.md).

The [latest neighbor extensions](experiments/ani_extensions/RESULTS.md) reuse all
splits below and compare both Ani winners. Their separate, prospectively declared
decision rule is in [EXTENSION_PROTOCOL.md](EXTENSION_PROTOCOL.md); it prioritizes
later-cohort prediction and retains all CV and supplied-validation trade-offs.

The [previous comparison](experiments/stack_improvements/RESULTS.md) adds Ani's earlier
unweighted 0.58810 winner as a second reference and evaluates 27 recipes on these
unchanged splits. Follow [IMPROVEMENT_PROTOCOL.md](IMPROVEMENT_PROTOCOL.md) for the
previous two-reference selection rule and adaptive-batch history. This document
continues to describe the original saved protocol and historical six-model suite.

The Git checkout includes all saved results but omits regenerable fitted-model caches. See [CACHE_REPRODUCTION.md](CACHE_REPRODUCTION.md) before running native-model verification or inference on a fresh checkout. Use `local_benchmark.py --suite full --force` to rebuild missing benchmark models; reading the executed notebook does not require them.

Hold submissions 05 and 06 while reviewing Ani's model and the benchmark results. They remain diagnostic candidates; lower tree weight improved one leaderboard comparison but does not establish that removing trees will beat 0.58688.

## Fixed protocol

`experiments/local_benchmark/protocol.json` freezes the exact fitting/evaluation IDs, trial grouping, random seeds, year boundaries, sample counts and dataset hashes. Reuse it for every candidate. If data or boundaries change, create a new protocol version rather than editing the existing split definitions.

The primary metric is **equal mean of Phase I, II and III log losses**, matching the test set's phase proportions. Lower is better. F1 at 0.5, ROC-AUC, average precision, Brier score and predicted/observed success rates are secondary diagnostics. A higher validation F1 alone is not a submission criterion.

| Evaluation | Fitting rows | Evaluation rows | Purpose |
| --- | ---: | ---: | --- |
| Supplied validation | 8,037 | 1,013 | Reproduce familiar comparisons |
| Train through 2007, evaluate 2008–2009 | 5,319 | 1,441 | Early historical transport |
| Train through 2009, evaluate 2010–2011 | 6,760 | 1,805 | Later historical transport |
| Train through 2011, evaluate 2012–2013 | 8,565 | 485 | Most recent labeled cohort |
| Trial-grouped 3-fold CV, seeds 42 and 43 | About 6,033 per fold | About 3,017 per fold | Six reproducible tests of within-period generalization |

The evaluation universe is the 9,050 rows in outcome_train plus outcome_val. Each fold excludes its own held-out trials completely. Combined-phase rows always stay together. Kaggle-test outcomes are absent and its rows never enter these splits. Temporal evaluation windows do not overlap; all fitting years precede their evaluation years.

All vocabularies, IDF values and other learned preprocessing are fitted on the fold's fitting rows only. Stacking also refits preprocessing inside each of its five trial-grouped inner folds. Outcome identifiers and Part One information are never predictive inputs.

## Ani reference and source audit

Ani branch commit: `1b8bd11c47428fb5453e53cdc275542b92a3b347`, whose message reports 0.58688. Preserved notebook and CSV snapshots are in `experiments/ani_comparison/source`; hashes and dataset equality checks are in `provenance.json`.

His final submission uses a grouped-OOF logistic stack of eight field-specific models, exponential recency weights with tau=3, text vocabularies/IDF learned from all available train/validation/test covariates, and final supervised refitting on train plus validation. It excludes stopping-reason inputs. Recency weights affect logistic/NB models and the meta-model; his kNN component uses similarity weights and ignores the recency argument.

`reproduce_ani.py` preserves that transductive recipe separately. Its reproduced final probabilities match his published submission within about 3.6e-9. His training-only validation phase-mean loss reproduces as 0.556108.

The main benchmark's `ani_stack` is an explicit **training-only preprocessing adaptation**, including inner-fold refitting. This prevents held-out text from shaping features in a prospective local test. It is not silently claimed to be identical to the transductive final submission. New models and the baseline use the same data boundaries and preprocessing rules. The exact reproduction and the strict comparison answer different questions.

`benchmark_ani_recipe.py` also runs the faithful published recipe on all ten saved splits in a separate `published_recipe_track`. This permits Ani's feature-only transduction while excluding evaluation outcome labels. Inspect it alongside the strict track, keeping their preprocessing policies explicit.

## Compared recipes

Six fixed candidates avoid another large hyperparameter search: Ani's strict stack, stronger meta regularization C=0.3, his existing capped Phase III year-trend alternative, our original three-model linear component, and fixed 75/25 or 50/50 Ani/linear probability blends. Blends use component predictions from the same held-out trials; no validation labels fit their weights.

The runner stores models, exact held-out predictions, fold audits, phase-level scores and a summary under a source-code and package-version fingerprint. Changing code creates a separate run directory. A failed or incomplete run cannot receive a complete-comparison status. Repeated-CV results pool all held-out trials within each repeat before calculating equal-phase loss, rather than averaging differently sized folds.

Conservative follow-ups reuse saved predictions on every split: 75% Ani plus 25% of his Phase III trend alternative, and the already-tested 75/25 Ani/linear blend. Comparisons are in `derived_candidates` under each track. Optional final CSVs 07/08 preserve Ani's published winning probabilities as their anchor. Our final linear component is refitted on the same 9,050 train+validation rows as Ani; those final-fit predictions are never called validation performance.

## Run and extend

From the repository root, using the existing virtual environment:

```sh
.venv/bin/python part2/benchmark_protocol.py
.venv/bin/python part2/local_benchmark.py --suite full
.venv/bin/python part2/local_benchmark.py --summarize
.venv/bin/python -m unittest discover -s part2 -p 'test_local_benchmark.py'
```

For the official and chronological comparisons first, use `--suite chronological`. The full command resumes cached completed folds. `experiments/local_benchmark/current_run.json` points to the latest results. Add a candidate adapter alongside `benchmark_models.py` and register its recipe in the runner; run it on the same saved splits, not new favorable boundaries.

Convenience commands from the repository root:

```sh
make -C part2 benchmark-test
make -C part2 benchmark
make -C part2 benchmark-report
make -C part2 benchmark-verify
```

`benchmark-report` reproduces the archived Ani recipe, scores its separate sensitivity track, generates the optional candidates and builds `local_benchmark.ipynb`. `benchmark-verify` checks native model replay and confirms a cached full rerun changes no fitted models and reproduces metrics/intervals. Results are in `experiments/local_benchmark/verification.json`.

## Reading results and choosing a submission

Start with the three chronological losses: inspect their mean, the worst period, and per-phase changes against Ani. Check whether grouped-CV performance corroborates the direction. Read the paired trial-bootstrap interval for the **difference** from Ani. Negative is an improvement. The temporal bootstrap averages the three periods equally; CV averages repeated losses per trial before resampling, so the same trial is not treated as independent several times. Combined-phase rows share resampling weights.

Prefer a candidate with consistent, useful chronological improvement and no substantial regression elsewhere. An interval spanning zero means the local evidence is inconclusive. Keep the official validation score as a diagnostic rather than letting it overrule the time-based checks. Do not select on a subset of completed folds or retune against one small phase subgroup.

These intervals are descriptive, conditional on fitted models and the consulted data. They do not account fully for model selection or quantify transport to 2014–2021. The latest chronological check has only 94 Phase I and 168 Phase III rows. Label noise, cohort shifts and public-leaderboard sampling can still reverse rankings. This environment reduces dependence on one validation score; it cannot promise Kaggle performance.

Freeze a recipe and benchmark results before producing its final CSV. If final fitting uses train plus validation, do that consistently for every final component and keep its held-out benchmark metrics separate. Do not call a refitted model's training predictions validation performance.
