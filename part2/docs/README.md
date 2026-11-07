# Documentation guide

Start with the [model-development narrative](../MODEL_DEVELOPMENT.md) for the
reasoning and chronology, or the [Part Two overview](../README.md) for navigation.

| Document | Purpose |
| --- | --- |
| [ORGANIZATION.md](ORGANIZATION.md) | Current layout, launcher, and workflow commands |
| [LOCAL_TESTING.md](LOCAL_TESTING.md) | Fixed CV/time splits, metrics, and preprocessing policies |
| [CACHE_REPRODUCTION.md](CACHE_REPRODUCTION.md) | Rebuilding ignored native models and reproducing predictions |
| [EXPERIMENT_OVERVIEW.md](EXPERIMENT_OVERVIEW.md) | Previous detailed README, including initial-model and reason-error tables |
| [MAIN_INTEGRATION.md](MAIN_INTEGRATION.md) | How the Part Two work was merged while preserving score tags |
| [layout_verification.json](layout_verification.json) | Checks that the reorganization preserved data, source, predictions, and working analyses |

The following protocols are frozen experiment records. Their exact bytes are
checked against saved manifests, so they retain their original paths and command
examples. In current commands, replace `python part2/SCRIPT.py` with
`python part2/run.py SCRIPT.py`; dependencies now live in `requirements/`.
Use the linked findings to distinguish an earlier recommendation from the
final decision.

| Research phase | Frozen protocol | Findings |
| --- | --- | --- |
| First attempts to improve Ani | [IMPROVEMENT_PROTOCOL.md](IMPROVEMENT_PROTOCOL.md) | [Results](../experiments/stack_improvements/RESULTS.md) |
| Recency and neighbor extensions | [EXTENSION_PROTOCOL.md](EXTENSION_PROTOCOL.md) | [Results](../experiments/ani_extensions/RESULTS.md) |
| All original recipes on identical splits | [ALL_SUBMISSIONS_PROTOCOL.md](ALL_SUBMISSIONS_PROTOCOL.md) | [Results](../experiments/all_submissions/RESULTS.md) |
| Blending after file 10 | [SPRINT_PROTOCOL.md](SPRINT_PROTOCOL.md) | [Results](../experiments/leaderboard_sprint/RESULTS.md) |
| Recent-tree additions | [SPRINT_TREE_PROTOCOL.md](SPRINT_TREE_PROTOCOL.md) | [Results](../experiments/leaderboard_sprint/recent_tree/dea847f57c7c07ee/RESULTS.md) |
| Refinements around files 11/13 | [REFINE_PROTOCOL.md](REFINE_PROTOCOL.md) | [Results](../experiments/leaderboard_sprint/refine/2e5c792462013673/RESULTS.md) |
| Final-slot alternatives | [FINISH_PROTOCOL.md](FINISH_PROTOCOL.md) | [Results](../experiments/leaderboard_sprint/finish/b235582538f63f9c/RESULTS.md) |
