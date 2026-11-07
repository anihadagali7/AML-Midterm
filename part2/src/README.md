# Source-code guide

Edit these files here; execute them with `python part2/run.py SCRIPT.py` from
the repository root. Their original contents are preserved so saved experiment
hashes and notebook-embedded source continue to match.

| Purpose | Main files |
| --- | --- |
| Shared supplied-data loading, features, and metrics | [experiments.py](experiments.py) |
| Ani's base models and original linear ensemble | [ani_models.py](ani_models.py), [benchmark_models.py](benchmark_models.py) |
| Fixed split definitions and benchmark runner | [benchmark_protocol.py](benchmark_protocol.py), [local_benchmark.py](local_benchmark.py) |
| Initial searches and original submissions | [boosted.py](boosted.py), [refinements.py](refinements.py), [ensemble_search.py](ensemble_search.py), [submission_models.py](submission_models.py) |
| First stack-improvement round | [improvement_models.py](improvement_models.py), [improvement_benchmark.py](improvement_benchmark.py), [improvement_text.py](improvement_text.py) |
| Recency and neighbor extensions (files 09/10) | [extension_neighbors.py](extension_neighbors.py), [extension_followthrough.py](extension_followthrough.py), [extension_predict.py](extension_predict.py) |
| Complete comparison of original recipes | [all_submission_models.py](all_submission_models.py), [all_submissions_benchmark.py](all_submissions_benchmark.py) |
| Final blend recipes and export (including file 13) | [sprint_models.py](sprint_models.py), [sprint_benchmark.py](sprint_benchmark.py), [sprint_legacy_fit.py](sprint_legacy_fit.py), [sprint_export.py](sprint_export.py) |
| Later tree/refinement/final-slot trials | [sprint_tree.py](sprint_tree.py), [sprint_refine.py](sprint_refine.py), [sprint_finish.py](sprint_finish.py) |
| Part 1 reason-error analysis | [error_analysis.py](error_analysis.py) |
| Checks and reports | `verify_*.py`, `*_verify.py`, `*_report.py`, and `build_*.py`; see [Make targets](../Makefile) |

The [development narrative](../MODEL_DEVELOPMENT.md) explains the decisions
behind these families. The [experiment index](../experiments/README.md) points
to their saved evidence. Models and preprocessing are unchanged by the folder
reorganization; [execution-layout details](../docs/ORGANIZATION.md#why-the-launcher-exists)
explain how their recorded paths continue to work.
