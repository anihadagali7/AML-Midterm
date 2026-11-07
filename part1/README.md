# Part 1

Start here for the current model and testing procedure. The **anchored reason classifier** has the highest recorded, user-reported Kaggle score: **0.89636**. Its submitted fit used all **235 verified labels** and the **1,014 originally masked training rows**; Kaggle test inputs were used only for prediction.

The model blends logistic regression, cosine nearest neighbors, and shrinkage LDA over the supplied reason embeddings. Clustering supports missing-label generation. See the [implementation](anchored_solution.py), [frozen recipe](experiments/paired_improvement/candidate/manifest.json), and [scored result with exact CSV hash](submissions/results/anchored-trainval235.json). The preparation manifest predates submission; the result record carries the later score.

## Comparable results

These results use the same saved **grouped, eight-repeat, five-fold** plans in each pool. Validation describes a separate fit on 157 labels; chronology pools the three fixed windows. [Complete comparison and paired results](experiments/current_leaders_grouped_v1/comparison.csv).

| Model | Train157 CV | Pooled235 CV | Validation78 | Chronology98 | Reported Kaggle |
|---|---:|---:|---:|---:|---:|
| Anchored | .932059 | .921365 | .891026 | .884354 | **.89636** |
| LDA 70/15/15 ensemble | .931794 | .921631 | .905983 | .877551 | .89401 |
| Current Ani LDA source | .928079 | .915426 | .871795 | .874150 | Unknown |

Anchored is effectively tied with the LDA ensemble on CV: slightly higher on Train157 and slightly lower on Pooled235. Its validation score is lower and its pooled chronology score is higher. It does **not** beat LDA on every check. Original Ani's .89423 belongs to a different, transductive submitted artifact; it is not the score of the current Ani LDA source.

## Testing and verification

Use [TESTING_POLICY.md](TESTING_POLICY.md) for the exact folds, fitting boundaries, metric, and advancement rules. A full comparison requires **40 fits on Train157, 40 on Pooled235, the original validation fit, and three chronological fits per recipe**. Older ungrouped CV remains a separate historical benchmark.

Run these commands from the repository root with the project's Python environment active ([dependencies](requirements-simple.txt)). They check existing evidence without retraining:

```sh
python part1/current_leaders_local.py verify
python part1/research/consolidation_v1/verify.py
python part1/research/consolidation_v1/verify_leaders.py
python part1/research/new_directions/verify.py
```

To explicitly replay the frozen comparison, choose an output directory that does not exist:

```sh
python part1/current_leaders_local.py replay --output /tmp/part1-leaders-replay
```

The replay performs 80 anchored CV fits, 80 current Ani control CV fits, and four additional anchored validation/chronology fits. Other comparison entries use retained predictions from the identical plans. It does not refit every archived competitor. Use the [research index](RESEARCH_INDEX.html) for individual study replay instructions. Historical scripts can contain paths to removed worktrees; use these relocation wrappers instead of running old drivers or submission exporters directly.

### Research environments

The research families recorded different package versions. Keep their environments separate when reproducing numerical results:

| Work | Recorded dependencies |
|---|---|
| Anchored model and common comparison | [requirements-simple.txt](requirements-simple.txt) |
| LDA research | [requirements-lock.txt](research/lda_ensemble/research/independent/requirements-lock.txt) |
| New Directions | [requirements.txt](research/new_directions/reproduction/requirements.txt) |

The documentation audit passed all four evidence checks above in the existing environment. The LDA replay import check also passed. The New Directions replay import check stopped because that environment lacks `iterative-stratification`; its recorded NumPy, pandas, and scikit-learn versions also differ. Install that archive's requirements in a separate environment before replaying it, then check imports without fitting:

```sh
python part1/research/new_directions/reproduce.py v6 --recipe teacher_mean3 --check
```

Passing evidence verification checks retained files and scores; it does not establish that every historical training environment is installed. The [documentation audit](research/documentation_audit_v1.json) records the checks and this remaining setup requirement.

## Where things belong

| Location | Purpose |
|---|---|
| [Testing policy](TESTING_POLICY.md) | Current evaluation contract and saved protocol paths |
| [Research handoff](RESEARCH_HANDOFF.md) | Short current status and guidance for the next experiment |
| [Research index](RESEARCH_INDEX.html) | Experiment families, negative findings, and replay guides |
| [Submission guide](submissions/README.md) | Scored CSVs, immutable tags, and score provenance |
| [Workflow](AGENTS.md) | Branch, commit, and retention conventions |

Keep current model artifacts at their existing paths: frozen recipes and provenance records refer to them. New work goes in a named, versioned study directory. Load the relevant research history when needed; the archived handoffs do not redefine the current benchmark or model status.
