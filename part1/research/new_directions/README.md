# Part 1 new directions: compact experiment record

**No model from these experiments was promoted or submitted.** The 136 settings across 14 sequential studies did not establish a reliable improvement over their relevant full-model comparators. These include controls and related variants, not 136 independent architectures. The newer anchored model was not evaluated by this research track.

Start with the [searchable summary](index.html), [complete attempt table](attempts.csv), or [study overview](studies.csv). The attempt table records hypotheses, exact settings, comparator names, selection rules, per-repeat scores and seeds, diagnostic exceptions, and reasons for rejection. [Supplementary results](supplementary.csv) cover validation and chronological windows. [Research notes](RESEARCH.md) retain the reasoning, literature links and limitations; [failures](failures.json) record the initial numerical failure and a concurrent documentation change.

## What was retained

The original 154 MiB workspace is represented by roughly 1.6 MiB of summaries and reproduction inputs. Frozen model and runner sources, recipes, exact fold assignments, chronological splits, diagnostic plans, small fixed comparator arrays, and recorded package versions are under `reproduction/`. The supplied data remain in the main project's `part1/data/`; they are not duplicated here. The original sibling folder is not required.

The bulky per-sample fitting audits, experiment prediction arrays, caches, logs, downloaded papers, duplicate reports/notebooks and obsolete failed source snapshots are intentionally omitted. The small reference prediction arrays are inputs to the original comparison runner and total less than 0.4 MiB. Paper links and checksums remain; downloaded papers are not executable dependencies.

Before removal, all 926 saved out-of-fold prediction sets (4,630 outer folds) and 40 supplementary fits were rescored and matched to the compact tables. [Migration check](migration_check.json) records that one-time verification. [Replay check](replay_check.json) records fresh reruns of kernel, clustering-average and pairwise models: all 15 replayed folds and four supplementary fits reproduced the original probabilities and predictions exactly. [Manifest](manifest.json) identifies retained files and the document-only omissions from old source hash lists. This preserves results and reproduction inputs, not the historical per-sample audit trail. Fresh reruns can regenerate numerical outputs and new fitting logs; original timings and historical logs cannot be recreated exactly.

## Testing and interpretation

The [copied testing policy](reproduction/references/TESTING_POLICY.md) and [frozen protocols](reproduction/protocols/train3.json) specify grouped five-fold evaluation, training-only preprocessing, and removal of masked rows sharing held-out embedding groups. Both train157 and pooled235 were screened on three saved repeats, with positive mean changes and at least two paired wins required in both pools. Eight-repeat confirmation required positive means and at least five wins in both. Declared diagnostic extensions after failed gates remain marked as diagnostics.

Versions 1–4 compared primarily with inductive Ani; versions 5–10 used the earlier submitted 70/15/15 ensemble; versions 11–14 used the stronger C.1/LDA ensemble. Early improvements over weaker parents do not establish improvements over the later ensemble. Repeatedly consulted validation and chronology checks are development diagnostics. Negative findings apply to the tested configurations and comparators, not all future uses of a method. No external trial data or deep learning was used.

## Check or reproduce

Check the compact record from the repository root with standard Python; this does not fit models:

```sh
python3 part1/research/new_directions/verify.py
```

To refit, use an environment matching [requirements](reproduction/requirements.txt) and [recorded versions](reproduction/environment.json), then run from this directory:

```sh
# Check inputs and imports only.
python reproduce.py v6 --recipe teacher_mean3 --check

# Refit one saved repeat (five folds); defaults to the training pool.
python reproduce.py v6 --recipe teacher_mean3 --repeats 1

# Refit the pooled comparison. Use a new output directory for each run.
python reproduce.py v6 --recipe teacher_mean3 --pool pooled --repeats 8

# Recreate this recipe's recorded validation and chronological diagnostics.
python reproduce.py v6 --recipe teacher_mean3 --supplementary --output .runs/cluster-diagnostics
```

The wrapper copies the frozen inputs to a new, disposable `.runs/` directory, connects the main project's supplied data, and calls the unchanged model and evaluation code. This directory is ignored by Git. It verifies hashes before fitting and handles explicitly omitted paper files without editing frozen sources or registries. Use `--data` for a relocated copy of the same supplied CSVs and `--output` for another new output directory. Do not run the old `freeze` commands or original archival verification scripts: the compact layout is intentionally different.

Exact numerical replay depends on matching software and numerical libraries. Seeds, data hashes, recipes, and saved splits make the experiments reproducible in principle; changing dependencies can change results. A rerun of an old experiment is not additional statistical confirmation and does not select or export a submission.
