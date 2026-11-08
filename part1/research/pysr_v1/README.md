# PySR reason classifiers

This study tests one compact symbolic-regression pipeline and one fixed 80/20 addition to our submitted 0.89850 blend. A logistic head on exactly the same features is a diagnostic control. The current 0.89935 anchored/Cand2 blend and the original anchored model are also matched controls. All 84 cases are complete. **This PySR pipeline and its fixed blend both underperform our submitted model. No candidate was promoted, fitted on all 235 labels for export, or submitted.** The complete experiment was preregistered in commit `033180cb`.

## Results and decision

| Recipe | Train157 CV | Pooled235 CV | Validation78 | Chronology98 |
|---|---:|---:|---:|---:|
| anchored | 0.932059 | 0.921365 | 0.891026 | 0.884354 |
| current75_25 | 0.932059 | 0.923404 | 0.891026 | 0.884354 |
| anchored80_cand2_20 | 0.935244 | 0.921631 | 0.891026 | 0.887755 |
| pysr | 0.869559 | 0.884184 | 0.844017 | 0.750000 |
| linear_geometry | 0.884820 | 0.883564 | 0.854701 | 0.797279 |
| current80_pysr20 | 0.927548 | 0.919770 | 0.884615 | 0.877551 |

Neither PySR candidate improves any of the four headline checks over our submitted 0.89850 recipe. The symbolic classifier falls behind the logistic diagnostic on Train157, validation and chronology; it is only .000621 ahead on Pooled235. That small difference does not rescue the compact-feature approach. These results support retaining the existing model, not spending a Kaggle submission on this trial.

Paired comparisons against our submitted 75/25 blend:

| Candidate | Train157 delta / wins-ties-losses | Pooled235 delta / wins-ties-losses |
|---|---|---|
| pysr | -0.062500 / 0-0-8 | -0.039220 / 0-0-8 |
| linear_geometry | -0.047240 / 0-0-8 | -0.039840 / 0-0-8 |
| current80_pysr20 | -0.004512 / 1-1-6 | -0.003635 / 1-0-7 |

None of the three evaluated heads/blends passes the strict both-pool rule against any of the three controls. Full per-repeat, per-label, population-SD, window, paired-delta and fit-access records are in `results/`. The newer cross-agent 0.89935 control is included from its frozen result commit; this study does not modify its source branch.

All 420 symbolic searches completed with no constant-target fallbacks or model failures. Selected equations average 4.91 nodes (maximum 15); `results/equations.csv` keeps every selected expression. The compact search and similarity features were fixed before scoring. This result does not establish that every possible PySR feature construction or search budget would fail.

## Example learned equations

These are the original-validation fit's five equations. Predicted score is `sigmoid(expression)`. Variables refer to the five derived, training-standardized similarity signals described below, so coefficients are specific to this fold's transformations and are not direct clinical effects.

| Reason | Selected expression |
|---|---|
| Safety | `centroid_margin*neighbor_logodds*(centroid_margin + cluster_logodds) - 5.449652517110098` |
| Funding | `(neighbor_logodds - 3.465965110568405)*3.9208404060432014` |
| Efficacy | `rival_neighbor_logodds*(-6.598356156306799) - 8.78078714512483` |
| Recruitment | `neighbor_logodds*3.0796055145512113 + 1.2890948116355418` |
| Other | `-rival_neighbor_logodds - 1.784033395546473` |

The assignment describes randomly rotated reason embeddings. Instead of searching arithmetic over arbitrary coordinates, PySR receives five signals per reason: nearest-neighbor label support, positive-versus-negative centroid similarity, cluster label support, and the strongest competing reason's neighbor and centroid support. All learned geometry and label-derived signals are restricted to each case's permitted fitting rows. Training features exclude the row's entire embedding group from the gold bank. KMeans generates the originally missing labels; gold labels remain unchanged, and generated rows receive only 10% of the total gold weight.

Five independent equations predict logits using weighted logistic loss. Search uses only addition, subtraction and multiplication, at most 15 nodes and depth five. Each outer fold gets a fresh deterministic search with 80 iterations, eight populations, fixed per-label seeds, and no held-out model selection. The formula-selection rule uses training loss and complexity. Sigmoid scores use the fixed .3 cutoff plus nonempty argmax. The blend centers that cutoff before adding 20% PySR to the existing blend. No thresholds, blend weights or search settings were tuned after scoring.

All 84 saved cases were completed: both eight-repeat grouped CV pools, original validation, and all three chronological windows. Report all complete-OOF repeat metrics, population SD, paired differences and per-label scores. Every supplementary case is evaluated regardless of the CV result, declared in advance. `protocol.json` freezes settings, source/data/control hashes and the advancement rule.

The compact `controls.npz` is self-contained. Its 0.89935 component comes from commit `cde3d500`, with exact case keys, fitting ID sets, query IDs and truth verified against our unchanged split generator; `build_controls.py` records the provenance. Reusing those predictions does not count as retraining the controls. Actual Kaggle test rows are not used to fit this study.

Setup: Python 3.12.13; install `requirements-lock.txt` in an isolated environment. Julia 1.13.3 and SymbolicRegression 2.7.2 are recorded in the Julia project and manifest snapshots. Before any real-label evaluation, synthetic checks exercised the complete pipeline, weighted custom loss, batch invariance and a full-budget timing run (32.45 seconds on 1,000 artificial rows including first-use compilation). `check_geometry.py` tests whole-group exclusion separately. The two-worker execution limit avoids excessive CPU/memory use on this machine.

```sh
.venv/bin/python part1/research/pysr_v1/check_geometry.py
.venv/bin/python part1/research/pysr_v1/study.py run --workers 2
.venv/bin/python part1/research/pysr_v1/retained.py verify
```

The runner resumes only complete, hash-matched cases and refuses partial case files. Search models and logs are temporary. For a fresh 84-case replay, pass `--output /tmp/pysr-replay-NEW`; then use `study.py verify --output /tmp/pysr-replay-NEW` while that replay's prediction cache exists. The normal retained verifier above checks compact evidence without retraining. Before cleanup, raw predictions were independently rescored and all 2,772 metric values matched exactly. Sufficient statistics reproduce those metrics, all paired counts and CV averages; detailed matched controls are also independently rechecked. Under `part1/RETENTION.md`, nonwinning candidate prediction caches and logs were removed. Source, settings, dependency snapshots, all scores, 420 equations and 84 fit audits remain; no backup archive was created.

PySR is a bounded exploratory test, not a comprehensive verdict on symbolic regression. These similarity features are learned transformations, so the equations are not formulas in physical variables. The development labels have already influenced previous work.

References: [PySR API](https://pysr.ai/api), [PySR source](https://github.com/astroautomata/PySR), and [Cranmer's PySR paper](https://arxiv.org/abs/2305.01582). The assignment PDF permits library implementations and requires citation of public code.
