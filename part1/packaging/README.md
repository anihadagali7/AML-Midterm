# Part 1 notebook packaging

This builds the report-focused hand-in at `submission/midterm/part1/` without
changing any frozen experiment, selected model, score record, or annotated tag.
It started from main `b83f8bd2`, after the new submission layout was pulled.

- `notebooks.py` authors the five notebooks, their narrative, data analysis and figure code.
- `handin.py` supplies portable model fitting and retained-prediction rescoring. The original anchored/Cand2 implementations are copied unchanged and remain visible in notebook 05.
- `build_handin.py` selects local source/data/evidence, records exact hashes, generates and executes notebooks, and creates `submission/dist/part1-midterm-code.zip`.
- `verify_handin.py` extracts that ZIP away from the repository and runs every finalized notebook, including fresh validation and final winning-model fits.

See the [hand-in README](../../submission/midterm/part1/README.md) for the contents,
dependencies and commands. The notebook package covers representative report
methods and negative directions, rather than inventing a notebook for every
historical parameter setting. Fitting boundaries and retained versus freshly
computed evidence are explicit in each notebook. The original feature model
used transductive preprocessing; this is preserved and labeled, not silently
substituted for the current winner.

The notebooks and compact outputs are committed. Materialized copies of existing
source/data and the ZIP are ignored and rebuilt. Existing research worktrees,
Part 2 hand-in and report are preserved. This branch is prepared for later
integration; packaging itself does not merge, push or upload anything.
