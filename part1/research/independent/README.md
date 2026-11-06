# Independent Part 1 research, integrated snapshot

Start with the central [handoff](../../RESEARCH_HANDOFF.md) and [testing policy](../../TESTING_POLICY.md). This directory preserves the independent workspace's scripts, trials, saved predictions, fold assignments, reports, and submission evidence without rewriting their frozen code.

`IMPORT_MANIFEST.json` maps 3,607 imported files to their original paths and SHA-256 hashes. The five raw CSVs are byte-identical to the repository's `part1/data`; `data` is a relative symlink to that directory. The assignment PDF also links to the shared repository copy. Git must preserve symlinks; on systems without symlink support, replace these links with copies of those same shared files before running code.

The machine-specific virtual environment, bytecode, old ZIP, and duplicated portable replay trees were excluded. Their original workspace remains intact. Top-level numbered notebooks and HTML reports were copied from its `midterm` package into this directory so their relative data/experiment paths resolve. They are historical executed deliverables; later study reports below contain the latest results. A final course submission notebook/report still needs to incorporate the chosen final model and later experiments.

## Run from this directory

Use the recorded Python/package environment in `requirements-lock.txt`. In a new environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python verify_integration.py
```

The integration was checked using the existing research environment; a clean environment install on a second machine has not been tested. Some historical optional model families require their native packages. `verify_integration.py` checks imported hashes, protected contracts, saved folds, OOF metrics, and submission artifacts without fitting a model or rewriting frozen results. It uses subprocesses for study guards because some historical runners configure shared module globals at import time.

For new work, use a new versioned study and reuse saved protocols as described in the central policy. Do not run old `prepare`, registry-registration, exporter, or full-search commands merely to inspect results: several intentionally write historical artifact paths. Exporting a scored submission again in place could replace its provenance manifest and break later guards.

## Current evidence map

| Directory under `experiments/` | Contents |
|---|---|
| `benchmark_v1`, `pooled_benchmark_v1` | Common grouped three-repeat benchmark and historical candidates |
| `cross_repo_benchmark_v1` | Six cross-branch recipes, original validation, chronology, and eight-repeat confirmation |
| `ani_improvement_v1` | Blends, pseudo-label trust/agreement, clustering seeds, Other specialist |
| `ani_combination_v1` | Matched follow-ups on both ensemble parents, including failed confirmation |
| `ani_bagging_v1` | Gold-subset averaging and its CV/validation tradeoff |
| `continued_search_v1` | Consolidated pre-transduction comparison/report |
| `refit_diagnostic_v1` | Paired reduced/expanded-label refit comparison |
| `three_way_transductive_v1` | Two explicitly authorized transductive arms; neither qualified for export |
| `architecture_review_v1` | Feature/model coverage, persistent errors, and proposed next hypotheses |

All earlier phases, logs, negative trials, and cached predictions remain under `experiments`. The central handoff adds the other branch's parsed-ICD experiments, which were outside the scope of the independent architecture inventory.

`READ_ME_FIRST.txt`, `experiments/current-best.json`, and `predictions/current_best` are preserved **historical** checkpoints. They are not the latest Kaggle winner or the selected three-way submission. Likewise, frozen pre-submission protocols correctly record their state at creation; subsequent `kaggle_result.json` files and the central submission ledger record later uploads.
