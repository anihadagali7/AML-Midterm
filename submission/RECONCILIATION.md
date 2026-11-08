# Reconcile the updated Part 1 work with the submission layout

The Part 1/report integration at `ee750f6fbbc9d3e15c729db7364043d9669393ea`
was merged with Part 2 packaging checkpoint
`efe9a8f23a0085c4010d6f9f0964cc3523d93213`. The original research commits and
immutable score tags are preserved. No model search, fitting, or Kaggle upload
was performed during this reconciliation.

## Final locations

- [Combined PDF](report.pdf): the incoming four-page report, byte-for-byte identical.
- [Combined TeX source](report/main.tex): all incoming authored content retained;
  only resource paths and compilation instructions differ.
- [Bibliography](report/main.bib): all incoming Part 1 additions preserved exactly.
- [Part 1 model and artifact](../part1/MODEL.md): selected anchored80/Cand2_20
  winner, reported Kaggle sample-wise F1 **0.89935**.
- [Final Part 2 notebooks](midterm/part2/README.md): file 13, reported Kaggle
  log loss **0.58465**, with the same verified **22.9 MB** compact archive.

The Part 1 working models, data, protocols, result tables and scored files match
the incoming checkpoint. Changes within `part1/` are confined to report links,
verification instructions, and adapting the report verifier to the new paths.
The verifier continues to check the original report audit at its historical
checkpoint, checks exact current report content against that checkpoint, and
separately protects the subsequent Part 2 packaging work. Historical integration
receipts and score records were not rewritten.

## Checks completed

- Part 1 report tables regenerate unchanged; all 12 canonical scored artifacts,
  recipe hashes and annotated tags pass, with nine model-comparison rows.
- The Part 1 report verifier passes, including 241 documentation links and exact
  preservation of report text, PDF, bibliography and frozen research files.
- The winning Part 1 retained-evidence verifier passes independent rescoring,
  repeated aggregation and exact-recipe validation checks.
- All 19 Part 2 tests pass. All three finalized notebooks execute from the extracted
  compact ZIP without Git, Part 1 or existing fitted-model caches; 825 packaged
  resources retain their hashes. The archive itself is unchanged from packaging.
- PDF extraction confirms four pages and both current reported scores.

See [reconciliation_checks.json](reconciliation_checks.json) for exact file
hashes and machine-readable verification summaries. The earlier
[fresh Part 2 training verification](VERIFICATION.md) remains valid; this
integration did not change either winning model or the Part 2 package.

The older Part 1 notebook/CSV files beneath `midterm/` remain historical hand-in
files, not a newly assembled notebook for the latest Part 1 winner. The current
model and reproduction entry point are in the linked Part 1 guide.
