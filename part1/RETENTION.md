# Part 1 retention policy

Keep **complete results for active leaders and useful matched controls**. For unsuccessful experiments, keep the recipe and results well documented, then remove the bulky generated evidence. This extends the [existing LDA retention policy](research/lda_ensemble/EVIDENCE.html) across Part 1.

## Keep for every experiment

- Source code or notebook, exact settings, data/feature choices, seeds, saved protocol or split reference, and decision rule.
- Complete compact score tables, the declared parent/control, selection criteria, and a short conclusion. Include negative results, failures, and limitations.
- Enough dependency and environment information to regenerate the experiment when it becomes useful again.

## Keep detailed evidence selectively

- The active 75/25 anchored/LDA blend: frozen recipe, exact scored CSV, full component/blend predictions, and matched evaluation evidence. Preserve the existing anchored and LDA component artifacts and fitting/generated-label records without duplicating them.
- The common grouped comparison and competitive LDA/Ani references: predictions, scores, truth, IDs, and audits needed to rescore and make paired comparisons.
- Exact scored submission CSVs, their hashes, source/recipe, fitting population/access, score provenance, and immutable tags. Preserve selected historical checkpoint models when their result records identify them.
- Original supplied inputs and saved split assignments. Keep useful benchmark-control predictions when existing comparisons or replay helpers depend on them.

A former within-study winner does not automatically earn permanent full storage. Preserve its conclusions and any submitted artifact; retain detailed outputs only when it remains a useful reference or dependency.

## Remove after checking summaries

Remove unsuccessful fitted models, repeated generated datasets, transformed feature matrices, per-trial label audits, fold caches, prediction dumps, and logs. Check their aggregate scores against the retained summaries before removal. Do not replace them with a duplicate compressed archive or an exhaustive discarded-file inventory.

Retain compact per-study counts and verification results. Historical manifests may still name discarded outputs: those hashes describe the original run, not the current retained files. Current retention manifests define what can still be verified.

Run historical replays in disposable output directories. Extract useful summaries and remove regenerated bulk afterward. Regenerate upstream model/data dependencies when an old notebook expects a discarded artifact; keep its original recipe unchanged.

## Current records

| Research family | Retention and verification |
|---|---|
| Older main Part 1 experiments | [31-study catalog and cleanup verification](research/legacy_retention_v1/README.md) |
| Earlier simple/trusted iterations | [Compact records and comparison correction](research/early_iterations/README.md) |
| Anchored transductive ablations | [Fixed recipes, complete compact results and interpretation](research/anchored_transductive_v1/README.md); unsuccessful prediction caches removed after rescoring |
| Anchored neighbor/blend ablations | [Three fixed ideas and complete results](research/anchored_neighbor_blend_v1/README.md); full evidence for the submitted 0.89850 blend and matched controls, compact results for unsuccessful neighbor changes |
| Closed blend follow-ups | [Ten fixed recipes across three batches](research/blend_followup_v1/README.md); both exports explicitly unsubmitted, exact small CSVs archived, four duplicate candidate arrays removed after rescoring; retained components reproduce all results |
| Submitted 75/25 blend | [Frozen final-235 preparation and score record](research/anchored_lda_blend_submission_v1/README.md); exact scored CSV and complete scores retained, existing component fits referenced |
| LDA and feature research | [Existing retention and replay guide](research/lda_ensemble/EVIDENCE.html) |
| New Directions | [Existing compact archive](research/new_directions/index.html); unchanged by this cleanup |
| Current leaders | [Comparison](experiments/current_leaders_grouped_v1/comparison.csv) and [verification commands](README.md#testing-and-verification) |

This is storage cleanup. It does not change scores, fitting boundaries, model selection, or submission status. Older Git commits and scored tags remain intact; removing files from the current checkout does not rewrite repository history.
