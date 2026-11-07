# Dylan v2 candidates on both benchmarks

Read [report.html](report.html). Parent: anchored, reported Kaggle **0.89636**.
Four fixed candidates (cand1, cand2, cand4, cand5) and three supplied equal-weight
blends were freshly compared with anchored. No candidate was tuned or submitted.

The subsequent [error-complementarity review](diversity_report.html) identifies
Cand1/Cand2's graph mechanism as a focused research follow-up despite the weaker
whole-model scores. Reproduce with `python
part1/experiments/dylan_v2_candidates_v1/diversity_review.py` from the repository
root. This is retrospective analysis of saved predictions, with no new model fits,
gates or hyperparameter selection; the proposed follow-ups have not been run.

## Outcome

All seven alternatives score below anchored on each of our four checks. No recipe
passes the paired advancement rule. All four standalone candidates lose every
Pooled235 repeat. Several score higher on Dylan's J; cand2 is best at .898785 versus
anchored's .885733. All supplied native candidate/blend A/B/C/J values reproduce.

The native runs preserve Dylan's transductive feature access and source calibration.
The current-policy runs use [strict_adapter.py](strict_adapter.py) to enforce
inductive fitting, grouped inner calibration, complete inner refitting, group-safe
synthetic features and training-pool SLD. These adaptations are explicitly versioned;
their scores are not attached to the original five-seed candidate CSV artifacts.

The primary evaluation uses inner seed0, matching Dylan's published benchmark.
The supplementary `five_seed/` study evaluates the final builder's five-inner-seed
average on every context. This check was declared after seeing the primary results;
all candidates were repeated without selection or tuning. There are no known
candidate Kaggle outcomes in the repository ledger.

## Replay from repository root

Python 3.12.13; use [requirements-lock.txt](requirements-lock.txt). The repository
environment was retained for matching controls. Its small version differences from
Dylan's supplied requirements did not affect any reported native score.

```sh
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r part1/experiments/dylan_v2_candidates_v1/requirements-lock.txt
.venv/bin/python part1/experiments/dylan_v2_candidates_v1/run.py --workers 4
.venv/bin/python part1/experiments/dylan_v2_candidates_v1/analyze.py
.venv/bin/python part1/experiments/dylan_v2_candidates_v1/check_boundaries.py
.venv/bin/python part1/experiments/dylan_v2_candidates_v1/seed_ensemble.py
.venv/bin/python part1/experiments/dylan_v2_candidates_v1/analyze.py --five-seed
.venv/bin/python part1/experiments/dylan_v2_candidates_v1/check_seed_average.py
.venv/bin/python part1/experiments/dylan_v2_candidates_v1/build_report.py
```

Saved hash-verified `results/` are reused by the runner; reuse is not fresh fitting.
To repeat all fits, use a disposable checkout and remove only this study's `results/`
directory there before running; remove `five_seed/results/` there for the seed-average
replay as well. The source snapshot and manifests are sufficient;
the external Dylan directory is not required for replay.

The exact plans reference the preceding `derived_benchmark_audit_v1` study. All 80
saved grouped folds, original validation and three chronology windows are retained.
Dylan's native benchmark has 50 A folds, three B cuts and one C fit. Fresh controls
are matched to the previous study's predictions, not just its rounded means.

## Evidence and retention

- `protocol.json`: predeclared models, full settings, fitting boundaries, seed
  convention, advancement rule and protected source/data hashes.
- `case_plan.json`: exact saved fitting/query populations for 138 contexts.
- `source/p1lib/`, `manifests/`: unchanged original implementation and candidate
  definitions; strict changes are confined to the separate adapter.
- `strict_inductive_comparison.csv`, `native_dylan_comparison.csv`: main results.
- `case_metrics.csv`, `block_metrics.csv`, `aggregate_metrics.csv`: sample and macro
  F1, each label's precision/recall/F1/support, repeat SD and all chronology windows.
- `paired_repeat_summary.csv`, `paired_intervals.csv`: paired differences and
  uncertainty, conditional on fitted predictions.
- `native_reproduction.csv`, `verification.json`, `boundary_checks.json`: native
  replay, all anchored controls, independent rescoring and leakage probes.
- `failure_notes.json`: negative findings and seed/artifact limitations.
- `five_seed/`: separately frozen seed-average protocol, equivalent detailed
  results and independent source-builder verification. Anchored arrays are reused
  from the primary study because its recipe is deterministic across these seeds.

Retain compact predictions and scores because these eight fixed recipes across two
access protocols are useful matched controls for interpreting the external proposal.
They are not a large hyperparameter search. No fitted estimators, graph caches,
transformed feature matrices or duplicate datasets are retained. Temporary logs
remain outside the repository. Source evidence documents are unchanged supplied
claims, not new testing policy.
