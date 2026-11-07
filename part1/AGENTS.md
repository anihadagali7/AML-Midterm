# Part 1 experiment and submission workflow

Use the existing `codex/part1-research-integration` branch for this research.
Do not merge branches unless Michael asks. Keep unrelated work out of commits.
The workspace GitHub account and commit identity rules still apply.

Read `README.md` and `TESTING_POLICY.md` before new model work. Use
`RESEARCH_HANDOFF.md` for the short handoff and `RESEARCH_INDEX.html` to find
relevant historical experiments. When improving the submitted model, declare
the anchored 0.89636 recipe as the parent before scoring. Use only supplied
Part 1 inputs; keep held-out labels and test inputs outside fitting under the
default inductive protocol.

Commit each completed experiment or coherent batch of ideas with its code,
fixed settings, evaluation protocol, summary results, and failure notes.
Record negative findings too. Keep detailed predictions for leaders and useful
matched comparisons; discard unsuccessful fold caches, logs and duplicate data
once summaries are checked. Follow `research/lda_ensemble/EVIDENCE.html`.

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
