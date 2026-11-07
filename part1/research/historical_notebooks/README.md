# Historical Part 1 entry notebooks

The main Part 1 notebook now presents the anchored classifier associated with the reported **0.89636** submission. These two earlier entry notebooks are preserved byte for byte, with their original cells, outputs and metadata.

| Snapshot | Origin | Contents |
|---|---|---|
| [main_gmm_logistic_b1cfb38.ipynb](main_gmm_logistic_b1cfb38.ipynb) | `main` at `b1cfb38`, originally `part1/part1.ipynb` | Earlier GMM/PCA label generation, logistic classifiers, submission and reason-label exports |
| [research_baselines_c0d443b.ipynb](research_baselines_c0d443b.ipynb) | Research branch at `c0d443b`, originally `part1/part1.ipynb` | Supervised baseline, K-means augmentation, year features and Safety adjustments |

The [manifest](manifest.json) records full source commits and SHA-256 hashes. The older main notebook's [submission CSV](main_gmm_logistic_b1cfb38_outputs/part1_submission.csv) and [reason-label export](main_gmm_logistic_b1cfb38_outputs/part1_reason_labels.csv) are preserved separately from the current scored artifacts. They are historical outputs, not the anchored submission or Ani's original .89423 artifact.

The former long [project overview](research_overview_c0d443b.md) is also archived; only its relative links were rebased. Its model recommendations describe that historical checkpoint.

The [pre-reconciliation working overview](working_overview_before_main.md) preserves the fuller uncommitted root README, including the anchored development narrative. It is an exact copy: its links still assume the original repository-root location, and its older recommendations remain historical. The [reconciliation record](workspace_reconciliation.json) explains how the original checkout was accounted for.

## Reading and replaying history

Use these notebooks to understand the original experiments. Their relative paths, environments, recommendations and output-writing cells are unchanged. Replay them only in a disposable reconstruction of their original project layout, regenerating any discarded intermediate outputs. Running them in the current project can overwrite historical outputs or recreate discarded caches.

The [current notebook](../../part1.ipynb), [testing policy](../../TESTING_POLICY.md), and [research index](../../RESEARCH_INDEX.html) describe the active workflow. Ani's [original notebook](../lda_ensemble/research/ani_snapshots/a548721/part1_best_model.ipynb) and [later LDA revision](../lda_ensemble/research/ani_snapshots/8a8c3fc/part1_best_model.ipynb) remain in their existing frozen archive.
