# Part 2 hand-in verification

- A fresh fit from the supplied CSVs used 9,050 labeled rows and read no existing
  fitted-model caches. Its CSV is **byte-identical** to the scored file 13:
  `be29fc849f63ed4c23b31e67c97aaa97c6940f569b0af0920cf9799b428f7636`.
  Fitting took approximately six minutes in the pinned Python 3.12 environment.
- Native inference remained identical after replacing IDs and adding dummy outcome
  labels. The numerical comparison against the parsed scored CSV differed by at
  most `1.11e-16`; serialization produced the exact original CSV bytes.
- All 19 existing Part 2 tests passed. The preserved model sources, original data,
  and scored submission files were not edited.
- All three finalized notebooks executed successfully, including 100 row-level
  recipe/split/track metric checks and byte verification of 68 embedded Python files.
- The compact ZIP was extracted to a temporary location outside the
  repository. All three notebooks ran there without Git, Part 1, or old model
  caches. All packaged resource files retained their original hashes after
  execution. The exact archive size, file count, hash and detailed receipt are in
  [part2_package_verification.json](part2_package_verification.json).

The default notebook checks rescore saved validation predictions and replay
recorded native components. They are distinct from the separate fresh fit above.
The complete historical search was not retrained for packaging. Every experiment's
settings, summary results, figures, implementations and original notebook records
are included. Intermediate arrays and repeated row predictions/audits are retained
in the working project, with only those used by the final comparisons in the ZIP.

The original Part 1 hand-in files were preserved byte-for-byte during relocation.
The report was subsequently reconciled with the newer Part 1 checkpoint at
`ee750f6f`: the incoming four-page PDF is byte-identical, its complete authored
TeX content is preserved with resource paths adjusted, and its bibliography is
unchanged. See [reconciliation checks](RECONCILIATION.md). No report recompilation
or editorial revision was needed for this folder reconciliation.

## Part 1 notebook package

The report-focused Part 1 package is under `midterm/part1/`. Five notebooks were
executed in the working tree and again from the exported ZIP in a temporary
folder with no Git repository or research checkout. The default final-model
notebook freshly fits anchored and all five Cand2 seeds for original validation
and final235. Both reproduce their frozen predictions; final235 reproduces the
exact `.89935` CSV SHA-256
`c20aa937fa8b1e5aea55bcd6564c6acd952f0d74030cc7c1a437eac54a1ee257`.

The baseline/feature notebook freshly reproduces the report's historical
validation feature comparison and explicitly labels its transductive boundary.
Other notebooks rescore 84 winner/control cases, paired-repeat and J rankings,
63 retained Other-combiner rows, 63 partial-label rows, 189 complementary-head
rows, and 96 grouped PySR repeat summaries from sufficient statistics. These
are retained-evidence checks, not fresh fits of every historical experiment.

[Part 1 verification receipt](part1_package_verification.json) records the ZIP
hash/size, notebook execution, fitting populations, and fresh-fit errors. The
ZIP includes supplied inputs and selected evidence as real files, without
fitted-model dumps. Its source notebook exposes the packaged implementations.
Historical runners remain source records and can require omitted intermediate
regeneration; portability is asserted for the five finalized notebooks.
