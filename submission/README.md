# Submission area

This folder separates the hand-in materials from the working projects. `part2/`
at the repository root remains the full Part 2 research history.

| Material | Location |
| --- | --- |
| Current combined report PDF | [report.pdf](report.pdf) |
| Combined report source and supporting files | [report/](report/README.md) |
| Final Part 1 notebooks and exact winning CSV | [midterm/part1/](midterm/part1/README.md) |
| Part 1 code/data ZIP | `dist/part1-midterm-code.zip` (generated locally) |
| Final Part 2 notebooks, winning CSV, and generated figures | [midterm/part2/](midterm/part2/README.md) |
| Part 2 code/data ZIP | `dist/part2-midterm-code.zip` (generated locally) |
| Reproduction and portability checks | [Verification](VERIFICATION.md) |
| Current notebook review against the assignment | [Notebook readiness](NOTEBOOK_READINESS.md) |

The assignment calls for a folder named **`midterm`** for the code/data hand-in,
and a separate report PDF upload. The Part 2 ZIP therefore contains `midterm/part2/`.
It is the Part 2 portion of the hand-in; it does not package or certify Part 1.

The combined report includes the updated Part 1 **0.89935 anchored/Cand2 winner**
and its evidence from `ee750f6f`. Its PDF is preserved byte-for-byte; source edits
only adapt TeX paths to this layout. See the [reconciliation record](RECONCILIATION.md).

The current Part 1 hand-in is under [midterm/part1/](midterm/part1/README.md): five notebooks covering the complete .89935 winner, representative baseline/feature experiments, benchmarks/blends, unsuccessful directions, and source code. Its builder creates a separate portable ZIP. The older Part 1 files directly under `midterm/` remain historical artifacts; use the new numbered notebooks for the selected winner.

The old root `midterm/` is a compatibility symlink to this folder's `midterm/`, so
historical paths continue to resolve. `midterm/report` similarly points to the
preserved earlier report source. These aliases are not included in the Part 2 ZIP.

## Build the Part 2 hand-in

From the repository root, with Python 3.12 and the pinned research environment:

```sh
python -m pip install -r part2/requirements/requirements.txt
python part2/packaging/build_handin.py
```

This materializes the supplied data, all trial settings and summary results,
and the row predictions needed by the final comparisons. It embeds all Python
implementations in a source notebook, executes the three finalized notebooks,
and creates a compact ZIP. Intermediate arrays, repeated per-candidate predictions
and audits stay in the working project. Generated copies are not duplicated in Git.
No data is downloaded by the builder and no submission is uploaded.

The finalized notebooks default to rescore/replay mode. Fresh winning-model
training is an explicit cell in notebook 01; its independent preparation-time
verification is included with the evidence.

## Build the Part 1 hand-in

```sh
python -m pip install -r part1/reporting/requirements.txt rdkit==2026.9.1
python part1/packaging/build_handin.py
python part1/packaging/verify_handin.py
```

The resulting ZIP contains `midterm/part1/`. Part 1 and Part 2 packages can be extracted into the same `midterm/` folder. The Part 1 notebook package freshly fits the exact winner by default and explicitly distinguishes current inductive measurements from historical transductive feature comparisons. See its README for scope and runtime modes.
