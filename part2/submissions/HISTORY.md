# Part Two trial phases and scored submissions

The work is split across trial-phase commits. Each of our six confirmed submitted
CSVs now also has a dedicated submission-record commit and an annotated Git tag
containing its reported Kaggle score. Lower log loss is better.

## Original trial history

| Commit | Phase | Evidence |
| --- | --- | --- |
| `c12df86` | Initial model search, original ensembles, Ani reproduction and initial fixed benchmark | [Initial experiments](../experiments/model_search), [benchmark](../experiments/local_benchmark/RESULTS.md) |
| `6f23dd5` | Compare both Ani winners and improvement variants | [Stack improvement results](../experiments/stack_improvements/RESULTS.md) |
| `135765c` | Recent and sharper neighbor extensions; prepare files 09/10 | [Neighbor extension results](../experiments/ani_extensions/RESULTS.md) |
| `3a880e9` | Refit and compare all original numbered recipes on identical splits | [Common benchmark](../experiments/all_submissions/RESULTS.md) |
| `269630e` | Sprint: 32 blend/meta proposals and four recent-tree mixtures; prepare files 11–14 | [Initial sprint](../experiments/leaderboard_sprint/runs/4dc0f274d45bce9e/RESULTS.md), [recent-tree batch](../experiments/leaderboard_sprint/recent_tree/dea847f57c7c07ee/RESULTS.md) |
| `a250cb9` | Post-first-place tuning: 19 refinements, nine final-slot checks, files 15/16 and the hold decision | [Refinements](../experiments/leaderboard_sprint/refine/2e5c792462013673/RESULTS.md), [final-slot checks](../experiments/leaderboard_sprint/finish/b235582538f63f9c/RESULTS.md) |

These are broad phases. The two sprint batches share `269630e`; the refinement
and final-slot batches share `a250cb9`. They were not originally committed one
recipe or one upload at a time. Files 13 and 15 also first had their scores
recorded together in `a250cb9`. Separate score updates for file 10 and file 11
already existed as `8d63e10` and `b6f29a2`.

## Dedicated submission records and score tags

| File | Reported Kaggle score | Submission-record commit | Annotated tag |
| --- | ---: | --- | --- |
| [03: phase-specialized](records/03_kaggle_0.59562.json) | 0.59562 | `0346bbe` | `part2-submission-03-kaggle-0.59562` |
| [02: text-heavy](records/02_kaggle_0.58993.json) | 0.58993 | `a5d3004` | `part2-submission-02-kaggle-0.58993` |
| [10: recent-neighbor blend](records/10_kaggle_0.58609.json) | 0.58609 | `6fa16c3` | `part2-submission-10-kaggle-0.58609` |
| [11: 20% original ensemble](records/11_kaggle_0.58474.json) | 0.58474 | `341fd10` | `part2-submission-11-kaggle-0.58474` |
| [13: 30% original ensemble](records/13_kaggle_0.58465.json) | **0.58465** | `9c009aa` | `part2-submission-13-kaggle-0.58465` |
| [15: recent tree and phase trend](records/15_kaggle_0.58492.json) | 0.58492 | `22e03c4` | `part2-submission-15-kaggle-0.58492` |

Each commit in this table adds only its corresponding submission record. The
record identifies the exact CSV and SHA-256, reported score, training population,
recipe, evidence manifest, original export commit, trial-phase commit and original
score-recording commit. The annotated tag repeats the artifact/hash and provenance.

These records were added retrospectively. Existing commits were not rewritten,
and no CSV was changed. Tags point to the new record commits, which contain the
current repository history; they do **not** reconstruct the whole repository as
it stood on the historical upload date. Use the record's `trial_phase_commit` and
`artifact_first_commit` to locate those original states. Tag creation dates are
not submission dates. Scores remain user-reported; no new Kaggle upload or API
query was made during this organization.

Files without a confirmed score—including held file 16—have no score tag.
Ani's original 0.58810 and 0.58688 reference submissions remain attributed to
their original commits in [the ledger](leaderboard_results.json); they are not
relabeled as our numbered submissions. Part One's existing tags are untouched.

**Current winner:** file 13 at **0.58465**, user-reported first place. File 15's
0.58492 regression remains tagged and documented. File 16 stays unsubmitted,
with the final slot deliberately held.

## Inspect or recover a submission

From the repository root:

```sh
git tag --list 'part2-submission-*'
git show --stat part2-submission-13-kaggle-0.58465
git show part2-submission-13-kaggle-0.58465:part2/submissions/13_file10_legacy30.csv > /tmp/part2_winner.csv
.venv/bin/python part2/verify_submission_tags.py
```

The verifier checks one distinct commit per record, annotated tags, the score
ledger, original score commits, manifest hashes, and identical CSV bytes in the
tagged, original-export and original-phase commits. It also checks 1,500 unique
IDs and valid probabilities. This organization creates local commits and tags;
it does not publish them to a remote.

## Convention for subsequent work

1. Commit each trial phase with its protocol, implementation, full comparison
   results and selection rationale before starting another materially different phase.
2. Preserve the exact submitted CSV and recipe in a dedicated submission record,
   referencing the phase and export commits. Use a new filename for changed predictions.
3. When the score arrives, commit that submission's result separately and create
   an annotated `part2-submission-NN-kaggle-SCORE` tag. Preserve unsuccessful results too.
4. Treat score tags as immutable. Keep unknown scores and unsubmitted candidates
   untagged, and use the live ledger to identify the current best.
