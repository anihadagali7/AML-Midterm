# Saved experiments

For the current decision, start with the [submission sprint](leaderboard_sprint/RESULTS.md).
For the reasoning across all phases, read [model development](../MODEL_DEVELOPMENT.md).

| Folder | Research phase and entry point |
| --- | --- |
| [model_search/](model_search) | Initial searches and reason analysis; [submission review](model_search/POST_SUBMISSION_REVIEW.md) |
| [ani_comparison/](ani_comparison) | Archived Ani source, original submissions, and reproduction provenance |
| [local_benchmark/](local_benchmark/RESULTS.md) | Original fixed comparison protocol and results |
| [stack_improvements/](stack_improvements/RESULTS.md) | First tuning round against both Ani winners |
| [ani_extensions/](ani_extensions/RESULTS.md) | Recency/neighbor experiments leading to file 10 |
| [all_submissions/](all_submissions/RESULTS.md) | Every original recipe on identical fitting/evaluation trials |
| [leaderboard_sprint/](leaderboard_sprint/RESULTS.md) | Files 11–16, final winner, unsuccessful refinement, and hold decision |
| [main_integration/](main_integration/verification.json) | Checks performed when merging the Part Two work |

Start with each `RESULTS.md`. Run-ID directories contain exact saved predictions,
per-phase scores, figures, manifests, and verification records. These IDs
preserve the experiment history; they are not separate recommended models.
Large fitted-model caches are ignored by Git and rebuilt using the
[cache instructions](../docs/CACHE_REPRODUCTION.md).
