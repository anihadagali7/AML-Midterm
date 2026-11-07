# Anchored + Dylan graph components, v1

**Neither mechanism produced a qualifying improvement.** C&S pseudo-labels hurt both CV pools. Graph correction was much closer and slightly helped Pooled235, but both scales hurt Train157. No variant advanced beyond the registered three-repeat screen; there is no new Kaggle submission or score.

This study started from `origin/main` at `256fa8d` in the separate `AML-Midterm-graph-components` worktree, branch `codex/part1-anchored-graph`. The parent is the submitted **anchored 0.89636** recipe. The [protocol](protocol.json) froze the source hashes, settings, hypotheses and advancement rule before any model fit. [Results](results/summary.csv), [every repeat and paired delta](results/paired_repeat_scores.csv), [per-label results](results/per_label_summary.csv), and [verification](results/verification.json) are retained.

## Matched screen results

Sample-wise F1, mean of the **same first three saved grouped repeats** in each pool. Each repeat contains all held-out predictions from five folds. Delta is against freshly fitted anchored; wins are paired repeat wins out of three.

| Recipe | Train157 | Delta | Wins | Pooled235 | Delta | Wins |
|---|---:|---:|---:|---:|---:|---:|
| Anchored control | .929229 | — | — | .915130 | — | — |
| Replace pseudo-label teacher with C&S | .922859 | −.006369 | 1/3 | .906856 | −.008274 | 0/3 |
| Add raw graph residual correction | .927813 | −.001415 | 1/3 | .916076 | +.000946 | 1/3 |
| Add automatically scaled graph correction | .927813 | −.001415 | 0/3 | .915603 | +.000473 | 2/3 |

The rule requires positive mean improvement and at least two wins in **both** pools. None passed. Raw correction tied one pooled repeat; automatically scaled correction tied one training repeat. Consequently the extra five repeats, original validation and chronology were not run. These are screen results, not a full eight-repeat comparison or Dylan's A/B/C/J benchmark. The parent values differ from the main README's eight-repeat means because this table uses only the first three repeats; all 30 fresh parent fits exactly reproduced their saved scores and binary predictions.

## What changed

1. **C&S pseudo-label teacher:** reuse Dylan cand1's graph configuration and Correct & Smooth algorithm, with grouped inner OOF powerset logistic predictions and fit-only preprocessing. Turn its combination probabilities into five label scores. Use these to fill eligible originally masked training rows, keeping anchored's confidence .8, label cutoff .3 plus argmax fallback, generated-row weight .25, three classifier passes, class weights and final .5 logistic/.3 neighbors/.2 LDA blend. The teacher is learned once from original fitting gold and reused on both refreshes. This intentionally replaces iterative self-generated labels with a fixed graph teacher. Cand1's lower confidence, different pseudo weight, classifier, temperature and expected-F1 decoder are not imported.
2. **Graph residual correction:** fit the unchanged parent. Within each outer training partition, refit anchored on five grouped inner folds, including scaling, PCA, GMM and pseudo-labeling, to obtain errors on gold rows the inner fit did not see. Diffuse the five independent label errors on a fitting-only graph. Extend the field inductively to query rows and add it to the parent's scores. Clip each label to [0,1], keeping the parent's decoder. Test raw magnitude and Dylan's automatic L1 scaling as the two fixed controls. This tests the correction component only; there is no final smoothing stage or cross-label normalization.

All graphs use standardization, PCA64, L2 normalization, ten neighbors, RBF scale 1, union symmetrization and correction alpha .8. The pseudo teacher also uses smoothing alpha .6382007398005838. Model and inner-fold seeds are 0. See [models.py](models.py) and the minimal teacher hook in [pseudo_anchored.py](pseudo_anchored.py). Passing no teacher to the latter reproduces the parent exactly.

## What the diagnostics suggest

Automatically scaled correction improved macro F1 in both pools: **.879262 → .885741** on Train157 and **.837236 → .842586** on Pooled235. It improved Funding and Other label F1 in both pools; Other improved by .027976 and .009916 respectively. Recruitment declined in both, and Efficacy declined on Train157. This is a category tradeoff, not a gain on the sample-wise objective used to evaluate these models. It is a useful clue for future work, not evidence that a label-specific correction chosen after seeing these results would generalize.

The graph field moves scores, but does not uncover a large supply of reliable fixes. On an average Train157 repeat, automatic scaling made 1.67 previously wrong label sets completely correct while breaking 5 previously correct sets; on Pooled235 it fixed 4 and broke 5. Partial-credit changes leave its pooled sample F1 slightly positive. Raw correction changed only 2.33 and 3 rows per repeat, respectively. [Complete correction counts](results/complementarity.csv) include improvements and regressions in row F1 rather than relying only on exact matches.

The C&S teacher admitted fewer pseudo-labeled rows: means of 659 versus anchored's 757 on Train157, and 635 versus 719 on Pooled235. Funding F1 fell by .096620 and .063848. It improved some other label metrics, but did not create a useful overall replacement under the fixed confidence rule. The observed decline cannot isolate label quality from the changed quantity of selected pseudo-labels; testing equal coverage would be a separately registered experiment.

These fixed tests do **not** rule out every graph method. They do show that neither a direct teacher swap nor these two residual scales justify replacing anchored or spending a submission on them. No threshold search, graph parameter search, label-specific gate or outer-label tuning followed the negative result.

## Boundaries and verification

- Same saved grouped Train157 and Pooled235 folds and rounded-embedding groups as the current testing policy; masked duplicates of held-out groups are purged in both outer and inner fits. Train157 never includes validation rows. No test CSV is loaded by the runner.
- Only eligible fitting rows are passed to the model components. Query labels are used for scoring and reference checks. Gold labels remain unchanged. All preprocessing and pseudo-label generation are relearned within the appropriate fit.
- Thirty outer cases, 150 independently checked nested group partitions and 4,704 recipe-row predictions. Teacher and residual stages each use those five nested partitions. Independent arithmetic rescoring agreed with the sklearn sample and per-label metrics; IDs, truth, coverage, decoder and protected hashes passed.
- Every fresh parent score matched the saved reference exactly. First-fold checks also passed for the no-teacher clone, zero-residual identity and query-batch independence. The fitting source and protected input hashes remained unchanged after freezing.
- One setup failure occurred before freezing or fitting: the initial package copy omitted a module imported by `p1lib.__init__`. The unchanged original modules were restored. There were no model-fit failures or post-score recipe changes.

These labels and folds have already influenced earlier development. Repeat variation is descriptive, not independent statistical evidence. The graph settings originated in Dylan's selected candidate, which is also a source of selection history.

## Reproduce and retain

Python 3.12.13 and [locked dependencies](requirements-lock.txt). The numerical package versions match anchored's environment. Vendor files are unchanged copies from the previous audit branch at `2bf7e4f`; [source provenance](source_provenance.json) records their hashes and original paths. The copied package contains unused native evaluators and exporters; this experiment invokes only its graph, feature, classifier and strict-adapter components. It does not run the vendor submission code.

From the repository root, with the study environment active:

```sh
python part1/experiments/anchored_graph_components_v1/verify.py

# Fresh fits in a new disposable directory:
python part1/experiments/anchored_graph_components_v1/run.py preflight --output /tmp/graph-replay
python part1/experiments/anchored_graph_components_v1/run.py screen --output /tmp/graph-replay --workers 4
python part1/experiments/anchored_graph_components_v1/verify.py --output /tmp/graph-replay --predictions
```

Do not call `freeze` again or edit frozen fitting sources. The existing protocol is checked before replay. If future variants are proposed, start a new version and predeclare them. The runner's confirmation and supplementary commands refuse to run without a screen passer.

Under the retention policy, temporary unsuccessful per-row predictions and fold files were removed after independent rescoring. Keep the source, protocol, full compact score tables, counts, fitting audits and verification receipt; canonical parent predictions already remain in the common leader study. The default verification command checks source/protected hashes and retained summary arithmetic, while the recorded full rescoring result refers to the checked predictions before disposal.
