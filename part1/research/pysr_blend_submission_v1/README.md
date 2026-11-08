# User-requested PySR blend trial

Prepared the exact fixed `current80_pysr20` recipe after the user requested a manual Kaggle trial despite its weaker local results. The user reported **Kaggle 0.89423**, a decrease of **0.00427** from our 0.89850 parent blend. The [exact-file result](../../submissions/results/current80-pysr20-trainval235.json) supersedes the preparation-time unknown-score status. The candidate is not promoted; the local and Kaggle changes agree in direction for this trial. No upload was performed by the agent. The [earlier study](../pysr_v1/README.md) and its outcome remain frozen historical records from before this request; this preparation supersedes their no-final-export status, without changing any evaluation result or promotion rule.

The CSV is [part1_current80_pysr20_trainval235.csv](../../submissions/ready/part1_current80_pysr20_trainval235.csv). It contains 312 rows and changes 9 labelsets from our submitted 0.89850 blend.

SHA-256: `742dc787fd33b24ab626dcf2ee29810cc3481f1d707de8df0678dec437127d66`.

## Exact fit and blend

- Fresh PySR fit on all 235 observed labels and 1,014 originally masked training rows, ordered as original train then validation. All five equations were freshly learned with the unchanged registered settings and seeds. The original gold labels were preserved.
- The existing anchored75/LDA25 component reuses its verified final235 fits and exact score packet. Its scored CSV hash and labels were checked before blending.
- Combine 80% of that component with 20% of PySR, after centering PySR's native .3 cutoff to .5. Decode at .5 with an argmax fallback. Effective component weights are 60% anchored, 20% LDA ensemble, 20% PySR.
- Actual test inputs enter only after fitting. No changes to thresholds, search budget, formula selection, features, pseudo-label weights or ensemble weights.

Preparation was frozen in commit `99bf06e8` before fitting or opening final test inputs. This is a user-authorized exploratory submission, not a locally confirmed upgrade. All four local headline checks regressed; the full 84-case comparison is retained in the prior study. The previous 78-row validation score remains an assessment of its separate 157-label fit, not independent validation of this all 235 artifact.

## Retention and verification

After the reported regression, the retention audit removed the final fitted state, augmented-label copy and score dump (366,692 bytes). The exact scored CSV, formulas, recipe, hashes, score record, fit summary and compact evaluation results remain. The separate PySR study's detailed arrays contain only matched leader/control predictions; its failed candidate CV caches were already removed.

The task-specific Python/Julia environment was also removed after confirming that retained verification runs in the existing project Python environment. Pinned Python requirements and Julia project/manifest files remain for a future refit. Other environments and shared package caches were not changed.

From the repository root, with an existing Python environment containing NumPy, pandas and scikit-learn:

```sh
python part1/research/pysr_blend_submission_v1/verify_retained.py
python part1/research/pysr_v1/retained.py verify
```

The compact export check verifies source and retention hashes, the exact 312-row CSV, ID/label format, all nine changed labelsets, reported score and immutable tag. The complete equation/model reconstruction passed before removal (maximum score error about 5.55e-11); it is not repeated by the compact check. The second command reconstructs all 2,772 CV/supplementary metrics from retained sufficient statistics and rechecks full leader/control predictions.

`manifest.json`, `verification.json`, `protocol.json` and the scored result are historical records and may name removed artifacts. `retention.json` defines the current files; `retention_audit.json` records the verified cleanup. The immutable scored tag `part1-current80-pysr20-trainval235-kaggle-0.89423` still preserves its original checkpoint. No tag was moved and Git history was not rewritten.

For full prediction reconstruction, use that scored checkpoint in a disposable checkout, or rebuild the environment from `../pysr_v1/requirements-lock.txt` and the recorded Julia versions, then rerun the unchanged exporter in a disposable checkout with generated outputs absent. The historical `verify.py` expects the removed final state and score packet. It remains frozen so the tagged preparation stays reproducible.
