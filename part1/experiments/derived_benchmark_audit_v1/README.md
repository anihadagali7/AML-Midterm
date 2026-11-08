# Derived benchmark audit v1

Retrospective audit of Dylan v2's fixed `J = (A + B)/2` against six previously
scored Part 1 recipes. Parent: anchored, reported Kaggle **0.89636**.

Read [report.html](report.html) for findings, caveats and the recommendation.
The default testing policy is unchanged. No new submission was generated.

## Result

- J correctly orders 10/14 unequal reported Kaggle pairs; current grouped
  Train157 orders 12/14. Excluding original Ani's transductive artifact mismatch,
  those counts are 8/9 and 9/9.
- J prefers LDA ensemble to anchored, reversing their recorded Kaggle scores.
- A stricter chronological boundary reverses that local result. This supports
  retaining chronology as a diagnostic, not promoting using J alone.
- The close leaders are not clearly separated by paired conditional intervals.
  Six selected recipes and reused labels cannot validate a future selection rule.

## Reproduce from repository root

Use Python 3.12.13 with the exact package versions in
[requirements-lock.txt](requirements-lock.txt). Core versions are also recorded
in [protocol.json](protocol.json). `iterative-stratification` is needed by the
archived benchmark adapter.

```sh
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r part1/experiments/derived_benchmark_audit_v1/requirements-lock.txt
.venv/bin/python part1/experiments/derived_benchmark_audit_v1/study.py --workers 4
.venv/bin/python part1/experiments/derived_benchmark_audit_v1/analyze.py
.venv/bin/python part1/experiments/derived_benchmark_audit_v1/verify.py
.venv/bin/python part1/experiments/derived_benchmark_audit_v1/build_report.py
```

`study.py` uses hash-verified saved predictions when present. That is rescoring,
not fresh fitting. For a fresh replay, copy this committed repository to a
disposable checkout and remove only this study's `results/` directory there
before running. Do not delete retained matched controls from the research branch.
The initial run freshly evaluated all six recipes on all 140 contexts.
Ensemble components share fits within a context.

The existing main-branch model sources, scored CSVs, datasets and exact canonical
splits are inputs. No frozen source is rewritten. Small unchanged source snapshots
make Dylan's supplied split/score definitions available even if the external folder
is relocated. A verifier-only correction is documented in
[failure_notes.json](failure_notes.json); recipes and results were unchanged.

## Evidence

- `protocol.json`: predeclared recipes, splits, metric formulas, access exceptions,
  versions and 167 protected source/data/artifact hashes; documented verifier fix.
- `splits.json`: all 140 exact fitting and query ID lists, including access audits.
- `results/`: compact score/prediction arrays and metric/hash sidecars for useful
  matched controls. No fitted estimators or duplicate datasets.
- `comparison.csv`, `rank_agreement.csv`, `pairwise_orderings.csv`: scores and
  retrospective rankings, including exclusion and leave-one-model-out panels.
- `aggregate_label_metrics.csv`, `case_scores.csv`, `repeat_scores.csv`,
  `paired_repeat_summary.csv`: precision/recall/F1/support, repeat SDs and paired
  wins/ties/losses. Per-trial F1 values are retained in a compressed CSV.
- `paired_differences.csv`, `conditional_intervals.csv`, `bootstrap_protocol.json`:
  shared-group bootstrap, conditional on the already fitted predictions.
- `verification.json`, `control_replay_checks.json`: independent rescoring,
  boundaries, source integrity and historical-control agreement.

All Kaggle scores are user-reported historical records. Original Ani's scored
CSV is transductive; its local adapter here is inductive. The cleaner five-recipe
sensitivity excludes it. The same DBSCAN recipe's 157-label and 235-label refits
cannot be differentiated by fixed-recipe CV; only the 235-label artifact enters
the primary panel. Dylan cand1/cand5 have no recorded Kaggle score and are shown
only as supplied context, not newly measured competitors or correlation points.
