# Ani source snapshots

These are exact files read from Git commits, with SHA-256 and source paths in `provenance.json`.

- `a548721`: original Part 1 notebook, generated training labels, and CSV associated with the reported Kaggle .89423 result. This is the source generation underlying the earlier cross-repo comparison.
- `8a8c3fc`: later Part 1 revision discovered when fetching during integration. Its `part1_first_submission.csv` exactly matches the original `a548721` CSV. Its changed `part1_best_submission.csv` has **no established Kaggle score in this handoff** and has not been evaluated under the new central protocol.

Notebooks are retained as source evidence. Their original relative data paths refer to Ani's `midterm` layout; they are not relocated standalone runners. Use the frozen adapters in `../independent` for the completed common-protocol comparisons. Review and version a new adapter before evaluating the later revision.
