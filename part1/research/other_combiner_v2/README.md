# Other-label specialists, pseudo-label reliability and learned combination

**Outcome: none of the seven variants passes the registered improvement rule. Keep the user-reported .89935 arithmetic blend. No new submission was exported.** All 84 outer cases and 252 nested folds finished; independent checks reproduced the parent controls exactly and verified every reported metric and fitting boundary.

| Recipe | Train157 | Pooled235 | Validation78 | Chronology98 | Pooled Other F1 |
|---|---:|---:|---:|---:|---:|
| Current .89935 parent | .935244 | .921631 | .891026 | .887755 | .621284 |
| Other RBF specialist | .934183 | .921631 | .891026 | .891156 | .621331 |
| Conditional Other RBF | .934713 | .921099 | .891026 | .891156 | .617835 |
| LDA pseudo weight .25 | .934713 | .921897 | .886752 | .874150 | .630235 |
| LDA confidence weights | .934713 | .921897 | .886752 | .874150 | .630235 |
| Per-label learned combiner | .934979 | .922252 | .891026 | .884354 | .619903 |
| Other-only learned combiner | .935244 | .921631 | .891026 | .891156 | .621045 |
| Parent-only calibration control | .934979 | .921720 | .891026 | .884354 | .621484 |

The full combiner's pooled gain is only .000621, with 4 wins /3 ties /1 loss; Train157 has 2/4/2 and a negative mean. The Other-only combiner ties both CV averages, with 1/6/1 on Train157 and 2/4/2 pooled. Its chronology gain is .003401, but this does not establish a generally better model. The specialized RBF heads do not provide a reliable improvement in Other classification. No thresholds, penalties or classifiers were changed after these outcomes.

Quarter-weighting pseudo labels increases pooled Other F1 by .008951 and recall from .546429 to .560714. Recruitment+Other detection increases from .25 to .875 of the nine positive cases per repeat on average, but it still misses most of them. Its pooled sample-F1 gain is only .000266 with **3 wins /0 ties /5 losses**. Train157, validation and chronology all regress; chronology drops .013605. The confidence-weighted version has different scores but identical hard predictions across the completed benchmark.

The most useful additional finding is that the unchanged anchored final235 fit generates **757 pseudo-labeled examples, of which 756 have one label and only one has two**. This explains why discounting uncertain secondary pseudo-labels has limited reach. The pattern is consistent with self-training underrepresenting combinations such as Recruitment+Other; it does not prove which unlabeled examples are wrong. Preserving uncertainty about unassigned secondary labels is a distinct future hypothesis, not a tested improvement here.

Complete repeat scores, population SDs, paired differences, per-label results and every chronological window are in [results/summary](results/summary/). The [independent verification receipt](results/summary/verification.json) records the checks before retention cleanup. [Retention](retention.json) explains which useful comparisons remain individually rescorable; unsuccessful inner fitting caches and detailed prediction dumps are removed after validation.

This study tests seven fixed variants in the three directions requested by Michael. The primary reference is the arithmetic 80% anchored /20% strict-inductive five-seed Cand2 blend, with user-reported Kaggle **0.89935**. Original anchored **0.89636** remains a secondary control. The branch starts at current main `766d16d3`; the newer parent is supplied by result commit `cde3d500` and retained benchmark inputs from `da68252a`, with provenance recorded here. Main's older .89850 leader documentation is not the parent of this study.

The initial [v1 implementation](../other_combiner_v1/FAILURE.md) stopped on pandas read-only array conversion before producing any candidate results. V2 requests a writable label-array copy. No model settings or statistical choices changed. Both registered source versions are retained.

## Hypotheses and fixed recipes

The parent misses most supplied Recruitment+Other examples: across its eight pooled OOF repeats, recall is 2.78% on nine such examples, versus 72.60% on 26 Other-only examples. These are descriptive reused development results, not an independent holdout. A subsequent supplied-label audit found no conflicting label sets within identical embedding groups. This does not establish that nearby embeddings are separable.

| Variant | Fixed change |
|---|---|
| `other_rbf` | Gold-only binary RBF SVC specialist for Other, used through an inner-OOF learned correction to the parent's Other score. |
| `other_conditional_rbf` | Separate Other specialists fitted to gold examples with and without Recruitment; their corrections are gated by the parent's predicted Recruitment score. |
| `pseudo_lda_quarter` | Within anchored's self-training, LDA discounts pseudo-labeled positive rows to quarter weight. Gold has full weight; original LR and neighbor treatment remains unchanged. |
| `pseudo_lda_confidence` | Quarter-weight LDA additionally reduces each pseudo-positive label's influence according to its confidence relative to the strongest label on that example. |
| `combiner_per_label` | Learns small, separate corrections from anchored and Cand2 scores for each reason. |
| `combiner_other_only` | Applies only the learned Other correction, preserving the other four score columns. |
| `calibration_control` | Learns corrections from the parent score alone, to distinguish recalibration from added information in separate component scores. |

SVC settings are C1, RBF gamma1/150, balanced classes and standardized supplied embeddings. Every learned correction uses offset logistic regression, with summed binary log loss and ridge penalty20 including its intercept. Zero correction exactly preserves the parent's decisions. All fixed scaling, clipping and missing-class fallback rules are in [study_models.py](study_models.py) and the [protocol](protocol.json). The weighted LDA implementation reproduces ordinary uniform-prior shrinkage LDA at unit weights in an independent numerical check.

This is not a new search over blend weights, thresholds or J weights. Related Other corrections were tried on older parents; these fixed variants test nonlinear specialists and the actual .89935 parent/Cand2 pair. Earlier blanket discounts on pseudo-labeled neighbors were negative and are not repeated here.

## Evaluation and fitting boundaries

Every variant receives all 40 Train157 and 40 Pooled235 saved grouped folds, original validation78, and the three fixed chronological windows. This is 84 outer cases with component fits shared across variants, not 84 independent full refits of each arithmetic formula. The original validation check retains its historical fitting inputs; grouped CV and chronology purge held-out embedding groups from both gold and masked fitting rows.

Each outer case builds three additional grouped inner folds (seed270201). Anchored, the exact five-seed strict Cand2 recipe and all specialists are fitted again within those inner folds, including the models' own pseudo-label generation and calibration. The learned heads see only these inner-held-out predictions and outer-training gold labels. Reusing ordinary outer OOF predictions to train these heads would leak outer-held labels indirectly and is deliberately avoided.

The source Cand2 manifest describes its original transductive recipe. The unchanged strict adapter overrides that access: all preprocessing and pseudo-labeling use only eligible fitting inputs. Query labels are removed before constructing model contexts. Its frozen eight-combination alphabet is retained; no held-out class frequencies are used. Actual Kaggle test inputs are not loaded during benchmark evaluation.

Fresh anchored scores must reproduce the retained control on every case. A fresh five-seed Cand2 fit on the first case must reproduce the retained component. The pseudo-label subclass must reproduce anchored when its changes are disabled. Saved IDs, source hashes, grouped boundaries, complete OOF coverage and metric arithmetic are checked independently.

Advancement requires positive mean sample-F1 change and at least five wins in eight repeats in **both pools against the .89935 blend**. Supplementary regressions remain visible. Macro F1, each label's precision/recall/F1/support, and Other-only versus Recruitment+Other diagnostics are reported separately. Raising Other F1 alone is not sufficient if overall sample F1 regresses. All labels and splits have influenced prior development; repeat results are descriptive and correlated.

## Reproduction

Use Python3.12.13 and the recorded package versions in `protocol.json` (Part1 requirements plus iterative-stratification0.1.9). From the repository root:

```sh
python part1/research/other_combiner_v2/run.py run --workers 4 --output /tmp/other-combiner-replay
python part1/research/other_combiner_v2/verify.py /tmp/other-combiner-replay
```

The runner checks the registered source/input hashes before and after fitting. It resumes complete case packets by their case IDs. Use a new directory for reproduction. No upload or model promotion is performed by the benchmark runner.

For retained evidence without refitting, run `python part1/research/other_combiner_v2/verify_retained.py`. The full `verify.py` command above applies to a fresh replay with its complete case packets. An interim `collect --partial` invocation before either pool had a complete repeat raised a reporting-only KeyError; it did not alter fitted models or registered settings, and final collection and verification completed successfully. Partial collection requires at least one complete repeat in both pools.
