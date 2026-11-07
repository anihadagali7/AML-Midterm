# Running the organized Part Two project

The [folder map](../README.md#folder-map) separates material by purpose. Research
results remain under `experiments/`, and scored CSVs/records remain under
`submissions/` at their original paths. All original Python source, supplied
data, pinned requirements, and frozen protocol documents retain their bytes.
The [path mapping](../layout.json) records every moved top-level file and the
Git commit immediately before the reorganization.

## Commands

Run these from the repository root after installing the appropriate
[dependencies](../requirements/README.md):

```sh
# Existing model/benchmark tests plus layout tests, and scored-tag verification.
make -C part2 check

# List scripts or execute a particular preserved script.
.venv/bin/python part2/run.py list
.venv/bin/python part2/run.py verify_submission_tags.py
.venv/bin/python part2/run.py all_submissions_verify.py

# Existing workflows retain their Make target names.
make -C part2 compare-all
make -C part2 compare-all-report
```

Other existing workflows include `benchmark`, `improve`, `extensions`, `sprint`,
and `sprint-refine`; their report/check/final targets are listed in the
[Makefile](../Makefile). Training targets can be expensive and may require
rebuilding ignored caches; see [CACHE_REPRODUCTION.md](CACHE_REPRODUCTION.md).
Reading saved reports or opening notebooks does not require training.

For a Python environment elsewhere, pass `PYTHON=/absolute/path/to/python` to
Make. Script input/output arguments stay relative to the directory from which
you launch the command. For example:

```sh
.venv/bin/python part2/run.py extension_predict.py \
  --input part2/data/outcome_test.csv --output /tmp/replayed.csv
```

That example replays the earlier extension candidate; the final winning blend
and its export commands are documented in the [sprint report](../experiments/leaderboard_sprint/RESULTS.md).

## Why the launcher exists

Historical scripts compute their data paths from their own location, and saved
manifests check source hashes. Editing all those scripts merely to move files
would change the provenance of the recorded experiments. The standard-library
[launcher](../run.py) instead builds an ignored `.runtime/part2/` directory with
byte-identical Python copies and links to canonical data, notebooks, protocols,
results, and submissions. It also preserves the sibling Part 1 path used only
by reason-error analysis. Data and fitted models are not copied.

Execute scripts through `run.py` or Make, and edit the canonical files in
`src/` or `tests/`. Changes to those files are copied on the next invocation.
Never edit `.runtime/`; it is disposable and regenerated as needed. Historical
notebook cells that write Python files now write only these execution copies.
Each notebook has a small initial setup cell so running it from `notebooks/`
finds the correct working directory. The original cells and saved outputs are
preserved. Report generation reinstalls that setup cell if it regenerates a
notebook and updates generated Markdown links for the organized paths.

Source edits that change a model still change its hash normally; the launcher
does not bypass provenance assertions or assign existing scores to new recipes.
Frozen protocol documents retain their historical command spellings; translate
them using the launcher convention above. Git tags still identify the exact
original submission records and artifacts.

## Reorganization verification

The [verification record](layout_verification.json) records the checks against
the pre-move commit: preserved source/data/protocol bytes, unchanged saved
artifacts, notebook execution, benchmark rescoring, tagged submissions, and
native replay of the winning model. The three layout tests cover isolation of
notebook-written source copies, notebook startup, and migrated report links.
