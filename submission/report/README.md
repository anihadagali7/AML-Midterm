# Combined report

`main.tex` is the combined report from the Part 1 integration at `ee750f6f`,
previously named `report.tex` at the repository root. It describes the selected
Part 1 anchored/Cand2 blend at 0.89935 and the Part 2 file-13 model at 0.58465.
`../report.pdf` is its verified four-page PDF, preserved byte-for-byte.
Only TeX resource paths and compilation instructions changed during relocation.
The current bibliography includes the incoming Part 1 additions. The
[Part 1 report verifier](../../part1/reporting/README.md) checks the relocated
source, PDF and bibliography against that upstream checkpoint.

Compile from this directory with a LaTeX installation:

```sh
latexmk -pdf main.tex
cp main.pdf ../report.pdf
```

`previous/` preserves the earlier multi-file report project that lived in
`midterm/report/`, including its figures. The active combined source uses the
support files directly in this directory. The Part 2 working report draft remains
in `part2/report/` at the repository root.
