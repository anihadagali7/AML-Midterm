# Combined report

`main.tex` is the combined report previously named `report.tex` at the repository
root. `../report.pdf` is the corresponding existing PDF, preserved byte-for-byte.
Only TeX resource paths and compilation instructions changed during relocation.
This move does not revise the scientific content or validate the Part 1 section.

Compile from this directory with a LaTeX installation:

```sh
latexmk -pdf main.tex
cp main.pdf ../report.pdf
```

`previous/` preserves the earlier multi-file report project that lived in
`midterm/report/`, including its figures. The active combined source uses the
support files directly in this directory. The Part 2 working report draft remains
in `part2/report/` at the repository root.
