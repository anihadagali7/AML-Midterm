# Part 1

Start here for the current model and testing procedure. The **75% anchored / 25% LDA ensemble blend** has the highest recorded, user-reported Kaggle score: **0.89850**, up **0.00214** from anchored's 0.89636. Both components use their existing final fits on **235 verified labels** and the **1,014 originally masked training rows**; Kaggle test inputs were used only for prediction. See the [winning blend recipe and verification](research/anchored_lda_blend_submission_v1/README.md) and [exact-file score record](submissions/results/anchored75-lda25-trainval235.json).

The blend aligns component decision cutoffs before combining scores. Its anchored component combines logistic regression, cosine nearest neighbors, and shrinkage LDA over the supplied reason embeddings, with clustering supporting missing-label generation. The [executed Part 1 notebook](part1.ipynb) still explains and refits that anchored component and reproduces its 0.89636 CSV; the winning blend's separate export adds the existing LDA ensemble. The notebook also verifies saved comparisons; fresh 80-fold CV is a separate replay. Frozen preparation manifests retain their earlier status; later score records supersede it.

## Comparable results

These results use the same saved **grouped, eight-repeat, five-fold** plans in each pool. Validation describes a separate fit on 157 labels; chronology pools the three fixed windows. [Original component comparison](experiments/current_leaders_grouped_v1/comparison.csv) and [complete blend study with paired results](research/anchored_neighbor_blend_v1/README.md).

| Model | Train157 CV | Pooled235 CV | Validation78 | Chronology98 | Reported Kaggle |
|---|---:|---:|---:|---:|---:|
| **Anchored75/LDA25 blend** | .932059 | **.923404** | .891026 | .884354 | **.89850** |
| Anchored | .932059 | .921365 | .891026 | .884354 | .89636 |
| LDA 70/15/15 ensemble | .931794 | .921631 | .905983 | .877551 | .89401 |
| Current Ani LDA source | .928079 | .915426 | .871795 | .874150 | Unknown |

The winning blend improved Pooled235 in all eight repeats and tied anchored's other three sample-F1 headlines. It did not meet the strict both-pool CV advancement rule because Train157's average tied (3 wins / 3 ties / 2 losses). The user authorized a manual trial after reviewing that distinction; its subsequently reported Kaggle score improved. Preserve that outcome without retroactively changing the evaluation rule. Original Ani's .89423 belongs to a different, transductive submitted artifact; it is not the score of the current Ani LDA source.

Ani's [original notebook](research/lda_ensemble/research/ani_snapshots/a548721/part1_best_model.ipynb), [original scored CSV](research/lda_ensemble/research/ani_snapshots/a548721/part1_best_submission.csv), and [result checkpoint](submissions/results/ani-original.json) are preserved on this branch. The notebook matches source commit `a548721` byte for byte; the score tag is `part1-ani-original-kaggle-0.89423`. The [later LDA notebook](research/lda_ensemble/research/ani_snapshots/8a8c3fc/part1_best_model.ipynb) is preserved separately. Archived handoff statements about its evaluation status describe their original checkpoint; the common comparison above includes the later source.

## Testing and verification

Use [TESTING_POLICY.md](TESTING_POLICY.md) for the exact folds, fitting boundaries, metric, and advancement rules. A full comparison requires **40 fits on Train157, 40 on Pooled235, the original validation fit, and three chronological fits per recipe**. Older ungrouped CV remains a separate historical benchmark.

Run these commands from the repository root with the project's Python environment active ([dependencies](requirements-simple.txt)). They check existing evidence without retraining:

```sh
python part1/current_leaders_local.py verify
python part1/research/anchored_neighbor_blend_v1/evidence.py verify
python part1/research/anchored_lda_blend_submission_v1/verify.py
python part1/research/legacy_retention_v1/verify.py
python part1/research/early_iterations/verify.py
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

The earlier documentation audit passed the four original evidence checks in the existing environment. The later retention cleanup added its own retained-file check. The LDA replay import check also passed. The New Directions replay import check stopped because that environment lacks `iterative-stratification`; its recorded NumPy, pandas, and scikit-learn versions also differ. Install that archive's requirements in a separate environment before replaying it, then check imports without fitting:

```sh
python part1/research/new_directions/reproduce.py v6 --recipe teacher_mean3 --check
```

Passing evidence verification checks retained files and scores; it does not establish that every historical training environment is installed. The [documentation audit](research/documentation_audit_v1.json) records the checks and this remaining setup requirement.

## Evidence storage

Follow [RETENTION.md](RETENTION.md): keep full leader/control results and compact recipes plus results for unsuccessful experiments. The [older-study cleanup](research/legacy_retention_v1/README.md) reduced tracked experiment outputs from 1,474 MiB to 38 MiB after checking the saved scores.

The [earlier-iteration archive](research/early_iterations/README.md) also preserves the simple/trusted prototypes and corrected Ani comparison that previously existed only in the working folder. Their recipes and summaries are committed; nonwinning generated caches are excluded. The selected anchored ablation and useful Ani control retain detailed evidence at their original paths.

## Where things belong

| Location | Purpose |
|---|---|
| [Testing policy](TESTING_POLICY.md) | Current evaluation contract and saved protocol paths |
| [Research handoff](RESEARCH_HANDOFF.md) | Short current status and guidance for the next experiment |
| [Research index](RESEARCH_INDEX.html) | Experiment families, negative findings, and replay guides |
| [Historical entry notebooks](research/historical_notebooks/README.md) | Both former `part1.ipynb` versions and older main exports |
| [Submission guide](submissions/README.md) | Scored CSVs, immutable tags, and score provenance |
| [Workflow](AGENTS.md) | Branch, commit, and retention conventions |

Keep current model artifacts at their existing paths: frozen recipes and provenance records refer to them. New work goes in a named, versioned study directory. Load the relevant research history when needed; the archived handoffs do not redefine the current benchmark or model status.
