# Part 1 report integration

Incoming main was `c2fb3fc18bb00a588c08f594977a1cf304950583`, including the completed PySR study and retention cleanup. The user confirmed that PySR was the other branch to reconcile. Integration started in a separate worktree on `codex/part1-report-integration`.

## Included research

Original commits and annotated score tags are preserved. The selected .89935 result remains on its own `cde3d500` result commit. No tag was moved, and no new score or submission was created.

| Branch | Included tip | Integration merge |
|---|---|---|
| Incoming PySR/main | `c2fb3fc1` | Integration base |
| `codex/part1-complementary-heads` | `5a019cfe` | `9cbfba97` |
| `codex/part1-cand2-followup` | `cde3d500` | `73ec270d` |
| `codex/part1-cand2-submission` | `d43b5c0c` | `939dbb14` |
| `codex/part1-derived-benchmark-audit` | `2bf7e4f9` | `94889f07` |
| `codex/part1-blend-refinement` | `da68252a` | `8e4271e2` |
| `codex/part1-anchored-graph` | `7a0649d7` | `ef09ab29` |
| `codex/part1-other-combiner` | `0243e9ac` | `b0cdf6d2` |
| `codex/part1-partial-secondary` | `0394e373` | `6070e2c4` |
| `codex/part1-blend-followup` | `2b4ebf5f` | `fd0b8528` |

Conflicts were confined to documentation and submission-index rows. Resolution retained each study's evidence and made the report winner explicit. The separate fresh-start line and other Part 2 branches were outside this integration. Existing research worktrees remain available.

## Documentation and report

The overview now leads to one complete model description, generated report tables, an exact-checkpoint handoff, a research index, and verification instructions. Stale references to separate worktrees and older “current” winners were removed from these entry points. Historical study files retain their original conclusions and paths.

The report's Part 1 method, abstract, comparison table and interpretation now describe the selected anchored/Cand2 blend, its exact fitting boundaries, negative experiments, and selection limitations. The anchored notebook remains a component notebook. Part 2 data/code, root README content, and report prose/tables are unchanged from incoming main. The PDF was rebuilt in the supplied template; font encoding is shared formatting, not a Part 2 model change.

## Verification and storage audit

The [verification guide](README.md) lists the commands actually run; [machine-readable checks](integration_checks.json) record the environment, successful exits, and report hashes. They cover winner scores and source dependencies, all twelve canonical scored CSVs and annotated tags, the grouped leader comparison, 840 derived-benchmark model/case rescores, complete later experiment summaries, PySR retained statistics and exact export, and 8,316 metric values from the closed blend follow-ups. No model was trained and no Kaggle upload occurred during integration.

The tracked Part 1 tree at the evidence-merge checkpoint was **119.01 MiB**. The merged research added **40.91 MiB** relative to incoming main. Those additions contain **no serialized fitted estimators and no `fold_cache/` or `cache/` directories**. Large additions are frozen split/fit-ID plans, useful matched predictions, and source/provenance; these support reproducible comparisons and verification. Plans remain at their hashed paths. Older selected checkpoint datasets and models already on main remain under their existing retention records. This audit does not claim the entire historical tree contains no models or generated data.

Unsuccessful recent fitted state and redundant fold caches had already been pruned in their experiment commits. Compact negative results, failures, exact scored CSVs, and the two explicitly unsubmitted follow-up exports are retained. No duplicate backup archive or new fitted-model output was added for the report. The final documentation commit changes no frozen model recipe, data, protocol, or scored CSV.
