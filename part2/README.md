# Part Two: start here

**Best scored submission: [file 13](submissions/13_file10_legacy30.csv),
Kaggle log loss 0.58465 (user-reported first place).** Lower is better.
File 15 scored 0.58492; file 16 remains unsubmitted.

This folder is the **working research history**. The finalized notebooks and
hand-in entry point are in [submission/midterm/part2/](../submission/midterm/part2/README.md).
Build their portable code/data ZIP with `python part2/packaging/build_handin.py`.

## What do you want to do?

| Task | Start here |
| --- | --- |
| Read or build the finalized hand-in | [Final Part 2 notebooks](../submission/midterm/part2/README.md) and [package builder](packaging/README.md) |
| Understand why we chose the final model | [Model development and tuning history](MODEL_DEVELOPMENT.md) |
| Find the exact winning CSV | [13_file10_legacy30.csv](submissions/13_file10_legacy30.csv) |
| Compare models and understand the latest decision | [Submission sprint results](experiments/leaderboard_sprint/RESULTS.md) |
| Understand CV, time splits, and preprocessing | [Testing methodology](docs/LOCAL_TESTING.md) |
| Read executed analyses | [Notebook guide](notebooks/README.md) |
| Find or edit the implementation | [Source-code guide](src/README.md) |
| Reproduce results or rebuild model caches | [Running the organized project](docs/ORGANIZATION.md), then [cache reproduction](docs/CACHE_REPRODUCTION.md) |
| Look up a score, commit, or tag | [Score ledger](submissions/leaderboard_results.json) and [submission history](submissions/HISTORY.md) |

## Folder map

| Folder | Contents |
| --- | --- |
| [data/](data/README.md) | Supplied train, validation, test, sample submission, and pathway CSVs |
| [src/](src/README.md) | Preserved model, benchmark, export, and reporting scripts |
| [notebooks/](notebooks/README.md) | Executed notebooks with saved outputs and implementation source |
| [docs/](docs/README.md) | Methodology, frozen experiment protocols, reproduction instructions, and historical overview |
| [experiments/](experiments/README.md) | Saved comparisons, predictions, figures, manifests, and verification records, grouped by research phase |
| [submissions/](submissions/) | Numbered prediction CSVs, exact-score records, and the live score ledger |
| [tests/](tests/) | Benchmark/model checks and folder-layout checks |
| [requirements/](requirements/README.md) | Pinned dependencies for the different workflows |
| [logs/](logs/README.md) | Historical console output; use experiment reports for conclusions |

The final recipe is **70% file 10 + 30% our original text/tree ensemble**,
with supervised components refitted on all 9,050 labeled rows. The
[development guide](MODEL_DEVELOPMENT.md#the-final-recipe-and-data-boundaries)
explains the components, pooled CV, and use of unlabeled test text for
vocabulary/IDF fitting in the Ani-derived components.

## Run checks or an experiment

From the repository root, using the project's Python environment:

```sh
make -C part2 check
.venv/bin/python part2/run.py list
.venv/bin/python part2/run.py all_submissions_verify.py
```

Existing named Make targets, such as `compare-all`, `extensions`, and `sprint`,
remain available. Supply `PYTHON=/path/to/python` to Make when using a different
environment. See the [workflow commands](docs/ORGANIZATION.md#commands).

Use `run.py` to execute scripts and edit their originals in `src/`. The launcher
preserves the execution paths expected by the recorded experiment sources.
The generated `.runtime/` directory is ignored by Git; no data or model copies
need to be committed there.

For the earlier initial-model tables and Part 1 reason-error analysis, see the
[historical experiment overview](docs/EXPERIMENT_OVERVIEW.md). The detailed phase
reports retain unsuccessful trials and superseded recommendations with context.
