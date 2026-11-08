# Part 1: stopping reasons

**Selected report winner: 80% anchored / 20% strict-inductive five-seed Cand2, reported Kaggle sample-wise F1 0.89935.** Model exploration is closed for this report phase. The complete winning source, exact scored CSV, benchmark evidence, and subsequent negative experiments are available in this checkout.

## Read this first

1. [Winning model](MODEL.md): inputs, both components, fitting boundaries, and the final prediction rule.
2. [Report guide](REPORT_GUIDE.md): comparable measurements, per-label results, negative findings, and limitations. Its tables are generated from retained evidence.
3. [Exact checkpoint and report handoff](REPORT_HANDOFF.md): immutable score tag, CSV hash, source references, and report files.
4. [Research index](RESEARCH_INDEX.html): every research family, including the later PySR and complementary-head trials.

The [report source](../submission/report/main.tex) and [compiled PDF](../submission/report.pdf) describe the selected blend. The [executed notebook](part1.ipynb) describes and refits its anchored component. Use the [winning exporter](experiments/cand2_followup_v1/export_blend.py) for the complete blend; the notebook alone is not the final model.

## Testing and verification

[TESTING_POLICY.md](TESTING_POLICY.md) defines the saved grouped folds, access boundaries, metric, and advancement rule. Train157 and Pooled235 each use eight repeats of five folds. Original validation uses a separate 157-label fit; chronology combines three earlier-to-later windows. The winner's blend comparisons reuse verified held-out component scores.

The winner improved both grouped means versus anchored, but its 5/1/2 Train157 and 4/1/3 Pooled235 wins/ties/losses did not satisfy the original five-wins-in-both-pools rule. The subsequent authorized exploratory submission supplied the highest reported Kaggle score. The previous 75/25 anchored/LDA blend still has a higher Pooled235 mean. These are reused development comparisons, not an untouched holdout or a significance test.

From the repository root:

```sh
python part1/reporting/build.py
python part1/reporting/verify.py
```

These verify retained artifacts and report organization without fitting models. The [verification guide](reporting/README.md) lists the scientific evidence checks and recorded environments. For retraining, follow each study's frozen recipe in a disposable checkout; numerical environments differ between studies. Historical absolute paths are provenance, not current run commands.

## Folder guide

| Location | Purpose |
|---|---|
| [MODEL.md](MODEL.md), [REPORT_GUIDE.md](REPORT_GUIDE.md), [REPORT_HANDOFF.md](REPORT_HANDOFF.md) | Current report entry points |
| [reporting/](reporting/) | Generated comparison and submission tables, verification, integration record |
| [submissions/](submissions/README.md) | Exact scored artifacts, immutable result records, historical unsubmitted files |
| [experiments/](experiments/) and [research/](RESEARCH_INDEX.html) | Frozen studies at their original paths; use the index to navigate both |
| [data/](data/) | Supplied Part 1 inputs |
| [part1.ipynb](part1.ipynb), [anchored_solution.py](anchored_solution.py) | Anchored component, preserved for exact reproduction |
| [RESEARCH_HANDOFF.md](RESEARCH_HANDOFF.md) | Closed research phase and outcome summary |
| [TESTING_POLICY.md](TESTING_POLICY.md), [RETENTION.md](RETENTION.md), [AGENTS.md](AGENTS.md) | Evaluation, storage, and workflow rules |
| [Historical notebooks](research/historical_notebooks/README.md) | Earlier entry notebooks and exports |

Study directories retain their original names because frozen protocols and hashes depend on them. Preparation-time manifests and archived READMEs describe their original checkpoints; later [scored records](submissions/README.md) determine submitted status. The report winner is explicitly identified above, regardless of historical filenames containing “best” or “current.”
