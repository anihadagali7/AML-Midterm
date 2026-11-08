# Submission area

This folder separates the hand-in materials from the working projects. `part2/`
at the repository root remains the full Part 2 research history.

| Material | Location |
| --- | --- |
| Current combined report PDF | [report.pdf](report.pdf) |
| Combined report source and supporting files | [report/](report/README.md) |
| Final Part 2 notebooks, winning CSV, and generated figures | [midterm/part2/](midterm/part2/README.md) |
| Part 2 code/data ZIP | `dist/part2-midterm-code.zip` (generated locally) |
| Reproduction and portability checks | [Verification](VERIFICATION.md) |

The assignment calls for a folder named **`midterm`** for the code/data hand-in,
and a separate report PDF upload. The Part 2 ZIP therefore contains `midterm/part2/`.
It is the Part 2 portion of the hand-in; it does not package or certify Part 1.

The combined report includes the updated Part 1 **0.89935 anchored/Cand2 winner**
and its evidence from `ee750f6f`. Its PDF is preserved byte-for-byte; source edits
only adapt TeX paths to this layout. See the [reconciliation record](RECONCILIATION.md).

The older Part 1 hand-in files in `midterm/` remain unchanged historical files;
they do not implement the latest winner. The current Part 1 implementation,
exact scored artifact and reproduction instructions are linked from
[part1/MODEL.md](../part1/MODEL.md) and [the report handoff](../part1/REPORT_HANDOFF.md).
This task reconciles that existing work; it does not build a new Part 1 notebook package.

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
