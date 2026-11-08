# PySR reason classifiers

This study tests one compact symbolic-regression pipeline and one fixed 80/20 addition to our submitted 0.89850 blend. A logistic head on exactly the same features is a diagnostic control. The current 0.89935 anchored/Cand2 blend and the original anchored model are also matched controls. Results are pending; no candidate has been promoted or submitted.

The assignment describes randomly rotated reason embeddings. Instead of searching arithmetic over arbitrary coordinates, PySR receives five signals per reason: nearest-neighbor label support, positive-versus-negative centroid similarity, cluster label support, and the strongest competing reason's neighbor and centroid support. All learned geometry and label-derived signals are restricted to each case's permitted fitting rows. Training features exclude the row's entire embedding group from the gold bank. KMeans generates the originally missing labels; gold labels remain unchanged, and generated rows receive only 10% of the total gold weight.

Five independent equations predict logits using weighted logistic loss. Search uses only addition, subtraction and multiplication, at most 15 nodes and depth five. Each outer fold gets a fresh deterministic search with 80 iterations, eight populations, fixed per-label seeds, and no held-out model selection. The formula-selection rule uses training loss and complexity. Sigmoid scores use the fixed .3 cutoff plus nonempty argmax. The blend centers that cutoff before adding 20% PySR to the existing blend. No thresholds, blend weights or search settings will be tuned after scoring.

All 84 saved cases are scheduled: both eight-repeat grouped CV pools, original validation, and all three chronological windows. Report all complete-OOF repeat metrics, population SD, paired differences and per-label scores. Every supplementary case is evaluated regardless of the CV result, declared in advance. `protocol.json` freezes settings, source/data/control hashes and the advancement rule.

The compact `controls.npz` is self-contained. Its 0.89935 component comes from commit `cde3d500`, with exact case keys, fitting ID sets, query IDs and truth verified against our unchanged split generator; `build_controls.py` records the provenance. Reusing those predictions does not count as retraining the controls. Actual Kaggle test rows are not used to fit this study.

Setup: Python 3.12.13; install `requirements-lock.txt` in an isolated environment. Julia 1.13.3 and SymbolicRegression 2.7.2 are recorded in the Julia project and manifest snapshots. Before any real-label evaluation, synthetic checks exercised the complete pipeline, weighted custom loss, batch invariance and a full-budget timing run (32.45 seconds on 1,000 artificial rows including first-use compilation). `check_geometry.py` tests whole-group exclusion separately. The two-worker execution limit avoids excessive CPU/memory use on this machine.

```sh
.venv/bin/python part1/research/pysr_v1/check_geometry.py
.venv/bin/python part1/research/pysr_v1/study.py run --workers 2
.venv/bin/python part1/research/pysr_v1/study.py verify
```

The runner resumes only complete, hash-matched cases and refuses partial case files. Search models and logs are temporary. After complete independent rescoring, unsuccessful predictions will be removed under `part1/RETENTION.md`; source, settings, compact scores, selected equations and fit-access audits remain.

PySR is a bounded exploratory test, not a comprehensive verdict on symbolic regression. These similarity features are learned transformations, so the equations are not formulas in physical variables. The development labels have already influenced previous work.

References: [PySR API](https://pysr.ai/api), [PySR source](https://github.com/astroautomata/PySR), and [Cranmer's PySR paper](https://arxiv.org/abs/2305.01582). The assignment PDF permits library implementations and requires citation of public code.
