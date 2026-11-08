# User-requested PySR blend trial

Prepared the exact fixed `current80_pysr20` recipe after the user requested a manual Kaggle trial despite its weaker local results. No Kaggle score has been reported and no upload was performed by the agent. The [earlier study](../pysr_v1/README.md) and its outcome remain frozen historical records from before this request; this preparation supersedes their no-final-export status, without changing any evaluation result or promotion rule.

The CSV is [part1_current80_pysr20_trainval235.csv](../../submissions/ready/part1_current80_pysr20_trainval235.csv). It contains 312 rows and changes 9 labelsets from our submitted 0.89850 blend.

SHA-256: `742dc787fd33b24ab626dcf2ee29810cc3481f1d707de8df0678dec437127d66`.

## Exact fit and blend

- Fresh PySR fit on all 235 observed labels and 1,014 originally masked training rows, ordered as original train then validation. All five equations were freshly learned with the unchanged registered settings and seeds. The original gold labels were preserved.
- The existing anchored75/LDA25 component reuses its verified final235 fits and exact score packet. Its scored CSV hash and labels were checked before blending.
- Combine 80% of that component with 20% of PySR, after centering PySR's native .3 cutoff to .5. Decode at .5 with an argmax fallback. Effective component weights are 60% anchored, 20% LDA ensemble, 20% PySR.
- Actual test inputs enter only after fitting. No changes to thresholds, search budget, formula selection, features, pseudo-label weights or ensemble weights.

Preparation was frozen in commit `99bf06e8` before fitting or opening final test inputs. This is a user-authorized exploratory submission, not a locally confirmed upgrade. All four local headline checks regressed; the full 84-case comparison is retained in the prior study. The previous 78-row validation score remains an assessment of its separate 157-label fit, not independent validation of this all 235 artifact.

## Retention and verification

`model_state.npz` preserves portable preprocessing, cluster and gold-bank state, feature scalings, all fitting IDs, augmented labels and weights. `equations.json` keeps the five formulas. `prediction_scores.npz` keeps both components and final predictions. The exact augmented-label table, changed-labelset list, manifests and verification are retained. No Julia search cache or pickled search model is needed to reproduce predictions.

```sh
.venv/bin/python part1/research/pysr_blend_submission_v1/verify.py
.venv/bin/python part1/research/pysr_v1/retained.py verify
```

The export verifier reconstructs normalized inputs, cluster assignment, similarities and equations independently from portable state, reproduces the blended scores and exact CSV labels, checks every ID and all artifact/source hashes, and verifies gold preservation. Maximum reconstructed PySR-score error was about 5.55e-11; all labels match exactly. The smallest final distance to the .5 cutoff is .00072957. Single-row versus batch predictions also matched. The full final evidence is retained while this trial is active; earlier failed CV caches remain compact under the retention policy.

`export.py prepare` can regenerate the fixed fit only in a disposable checkout where this version's generated outputs are absent; it refuses to overwrite the existing candidate. Python/Julia dependencies and underlying model settings are frozen in the prior study. Reported Kaggle results should get a separate exact-file result commit and immutable score tag.
