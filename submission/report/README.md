# Combined report

`main.tex` is the combined report from the Part 1 integration at `ee750f6f`,
previously named `report.tex` at the repository root. It describes the selected
Part 1 anchored/Cand2 blend at 0.89935 and the Part 2 file-13 model at 0.58465.
`../report.pdf` is the compiled report. The prose describes the work in the
team's voice, with explicit Methods, Results, and Discussion sections and
Part 1/Part 2 subsections. For Part 1, it explains the progression from validation-based selection to
repeated grouped cross-validation. Part 2 describes its grouped and
chronological evaluation separately.
The Part 2 tables compare the baseline and final model on the same held-out
validation trials and distinguish observed from predicted stopping reasons.
These tables use the saved validation evidence; no models were retrained.
The assignment allows five pages including figures, with references excluded.
The current bibliography includes the incoming Part 1 additions. The
[Part 1 report verifier](../../part1/reporting/README.md) checks the relocated
source, PDF and bibliography against that upstream checkpoint. Its exact-source
and PDF snapshot checks predate these editorial revisions.

Compile from this directory with a LaTeX installation:

```sh
latexmk -pdf main.tex
cp main.pdf ../report.pdf
```

`previous/` preserves the earlier multi-file report project that lived in
`midterm/report/`, including its figures. The active combined source uses the
support files directly in this directory. The Part 2 working report draft remains
in `part2/report/` at the repository root.
