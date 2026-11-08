# Report verification

[Overview](../README.md) · [Complete model](../MODEL.md) · [Report guide](../REPORT_GUIDE.md) · [Integration audit](INTEGRATION.md).

`benchmark_comparison.csv` maps nine report recipes to exact local measurement tables and scored records. `scored_submissions.csv` lists twelve canonical scored artifacts, their hashes, result commits, and annotated tags. Earlier logistic/clustering artifacts remain in the [historical submission ledger](../submissions/README.md).

From the repository root, Python 3.12 or later:

```sh
python3 part1/reporting/build.py
python3 part1/reporting/verify.py
```

The first command regenerates the guide and tables from retained evidence, checks CSV and frozen-recipe hashes, and compares all twelve result records and exact CSVs to their annotated tags. The second checks generated tables, report links, preserved model/data/protocol files, and unchanged incoming Part 2 files and report text. Neither fits a model. A full Git checkout with the original scored tags is required; fetch the tags when verifying a new clone.

## Scientific evidence

Activate an environment with [the reporting dependencies](requirements.txt), which add `iterative-stratification` to the project's scientific dependencies, then run the relevant retained evidence check. For a fresh environment: `python3 -m venv /tmp/part1-report-env`, activate it, and run `python -m pip install -r part1/reporting/requirements.txt`. These commands were run successfully in the report integration checkout; they rescore retained outputs or sufficient statistics rather than train models.

```sh
python part1/experiments/cand2_followup_v1/verify_retained.py
python part1/current_leaders_local.py verify
python part1/reporting/verify_derived.py
python part1/experiments/blend_refinement_v2/verify.py
python part1/research/other_combiner_v2/verify_retained.py
python part1/research/partial_secondary_v1/evidence_partial.py verify
python part1/research/complementary_heads_v2/evidence.py verify
python part1/research/pysr_v1/retained.py verify
python part1/research/pysr_blend_submission_v1/verify_retained.py
python part1/research/blend_followup_v1/verify.py
```

The derived/J study froze hashes of the then-current retention document and submission ledger. Those two living records changed as research continued. `verify_derived.py` checks their historical bytes at the protocol's base commit, and all **165 other protected files from the live checkout**. It runs the original verifier unchanged in a disposable symlink tree, rescoring 840 model/case predictions. It does not rewrite the frozen protocol or saved verification. This exception covers documentation and the historical ledger only.

Evidence checks are not a fresh training replay. Use each study's recorded dependencies and replay instructions in a disposable output directory for that purpose. In particular, the [Cand2 lock](../experiments/cand2_followup_v1/requirements-lock.txt), [LDA environment](../research/lda_ensemble/research/independent/requirements-lock.txt), and [New Directions environment](../research/new_directions/reproduction/requirements.txt) describe distinct historical environments. Passing the commands above does not establish that every archived training dependency is installed. PySR retained verification does not require reinstalling its removed Julia training environment.

## Report build

From the repository root, with a LaTeX toolchain:

```sh
latexmk -pdf report.tex
```

The committed PDF was also built with Tectonic 0.17.0 using `tectonic --keep-logs --outdir /tmp/aml-part1-report-build report.tex`, then copied into the root after checking its pages and text. It is four pages including references, within the five-page course limit. T1 font encoding keeps the supplied CVPR template's Times fonts under either engine. Two existing BibTeX warnings concern undated RDKit/ClinicalTrials.gov references, which retain access dates; there are no unresolved citations or overfull boxes. Build intermediates stay outside the repository.
