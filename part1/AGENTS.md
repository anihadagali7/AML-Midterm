# Part 1 experiment and submission workflow

Start new research on a task branch based on current `main`. The consolidated
Part 1 model, testing procedure and historical records live in this project.
Do not merge branches unless Michael asks. Keep unrelated work out of commits.
The workspace GitHub account and commit identity rules still apply.

Read `README.md` and `TESTING_POLICY.md` before new model work. Use
`RESEARCH_HANDOFF.md` for the short handoff and `RESEARCH_INDEX.html` to find
relevant historical experiments. Model search is closed for the current report. If new model work is requested,
declare the selected anchored80/Cand2_20 0.89935 recipe as the parent before
scoring, and retain anchored 0.89636 as the historical benchmark control.
Preserve the frozen testing policy and original study advancement decisions. Use only supplied
Part 1 inputs; keep held-out labels and test inputs outside fitting under the
default inductive protocol.

Commit each completed experiment or coherent batch of ideas with its code,
fixed settings, evaluation protocol, summary results, and failure notes.
Record negative findings too. Keep detailed predictions for leaders and useful
matched comparisons; discard unsuccessful fold caches, logs and duplicate data
once summaries are checked. Follow `RETENTION.md`; do not create duplicate
backup archives of discarded unsuccessful outputs. Keep exact scored artifacts.

Each manually submitted result gets a separate commit with the exact CSV (or
an existing tracked copy), SHA-256, frozen recipe/source reference, fitting
population/access, reported score and score provenance. Create an annotated,
immutable tag `part1-<recipe>-kaggle-<score to five decimals>` pointing to that
result commit. Distinguish models that tie by recipe name. Never put a CV score
in a Kaggle tag or move an existing scored tag. Backfilled result records must
say they document an earlier submission, not a new upload.

Use ordinary descriptive commits for unsubmitted experiments. Commit locally
as part of completed work; pushing, uploading to Kaggle and merging are separate
actions. Preserve other agents' work and record its provenance when included.
