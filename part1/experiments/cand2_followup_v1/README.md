# Cand2 result: matched J audit and a fixed blend

**Update: the user reported Kaggle 0.89935 for the exact 80/20 blend, improving the prior best 0.89636 by 0.00299.** [Exact result](../../submissions/results/anchored80-cand2-inductive20-trainval235.json). Preparation-time records below predate that manual submission.

**The new result weakens the case for J as the primary ranking rule. A small Cand2 blend is a more promising next experiment than replacing anchored.** Cand2's user-reported Kaggle score is .89102 versus anchored's .89636. We freshly evaluated the exact submitted strict-inductive five-seed recipe on J's original assignments, resolving the mismatch with Dylan's earlier transductive J score.

The user asked what the new outcome implies for J and future directions. This study was frozen after that Kaggle outcome was known. It did not change J's weights, assignments or model settings. Work starts from current main `31f3fa9` in a separate worktree and branch `codex/part1-cand2-followup`. The new main changes were reports; the canonical model, policy and data were unchanged.

## J gets this ordering wrong

| Measurement | Anchored | Exact submitted Cand2 recipe | Preferred |
|---|---:|---:|---|
| J's A: ten-repeat ordinary CV | .924610 | .909787 | Anchored |
| J's B: mean of three original chronological cuts | .846855 | .882555 | Cand2 |
| **J = (A+B)/2** | **.885733** | **.896171** | **Cand2** |
| Current grouped Train157 CV | .932059 | .923301 | Anchored |
| Current grouped Pooled235 CV | .921365 | .903599 | Anchored |
| **User-reported Kaggle** | **.89636** | **.89102** | **Anchored** |

The matched J score .896171 belongs to the submitted inductive five-seed recipe. Dylan's original single-seed .898785 and our earlier five-seed native .897862 described the transductive version; neither is attached to this submitted artifact. Current-policy scores are reused from the verified prior five-seed audit, not newly refitted here.

J's chronology term rewards Cand2 enough to outweigh its A regression. Our stricter chronology check favors anchored instead (.884354 versus .863946). J's B uses all 1,014 originally masked training rows, including later years; current chronology uses earlier eligible rows and duplicate-group purges. J's A is not group-separated. These differences make the two chronological scores different stress tests. The disagreement does not identify one boundary choice as the sole cause, but it argues against treating their mean with CV as a reliable universal ranker.

The expanded historical panel has seven reported recipes. Excluding original Ani, whose historical submitted artifact differs from its local adaptation, leaves six recipes and 14 pairs with unequal reported Kaggle scores:

| Metric | Correct pair orderings | Wrong |
|---|---:|---:|
| J | 10/14 | 4 |
| Grouped Train157 | 14/14 | 0 |
| Grouped Pooled235 | 13/14 | 1 |

Including original Ani gives J 12/20 and each grouped CV measure 17/20. The earlier six-model audit had J 10/14 and Train157 12/14; those were a different model panel, not directly interchangeable denominators. [Expanded panel](results/expanded_panel.csv), [every pair](results/pairwise_orderings.csv), [agreement counts](results/rank_agreement.csv).

These counts are descriptive. We added **one new Kaggle outcome**, not five or six independent experiments; pairs share models. The small, selected set, reused development labels, model search history and user-reported leaderboard scores prevent a general proof. CV got this direction right but overstated the observed score gap. Keep grouped CV as the primary screen, retain chronological checks separately, and log future candidate predictions before submission. Do not tune a replacement J formula to fit this panel.

## One fixed next direction: anchored80 / Cand2 20

We registered **one** blend: .8 times anchored's native selection scores plus .2 times Cand2's five label marginals, followed by anchored's .3 cutoff and nonempty argmax. No blend-weight, label-specific weight, threshold or confidence-gate search. These are selection scores, not established calibrated probabilities. Cand2's standalone decoder remains expected F1; the blend uses the registered shared threshold.

We reused all saved held-out component predictions for both grouped eight-repeat pools, original validation and the three chronological windows. This full diagnostic, including supplementary evaluation regardless of CV outcome, was declared before computing the blend.

| Current-policy check | Anchored | 80/20 blend | Change |
|---|---:|---:|---:|
| Train157 CV, eight repeats | .932059 | **.935244** | +.003185 |
| Pooled235 CV, eight repeats | .921365 | **.921631** | +.000266 |
| Original validation78 | .891026 | .891026 | .000000 |
| Chronology, pooled98 | .884354 | **.887755** | +.003401 |

The blend wins/ties/loses **5/1/2** Train157 repeats and **4/1/3** Pooled235 repeats. It therefore **does not pass** the registered requirement of at least five wins in each pool. Pooled improvement is tiny. It is a plausible exploratory candidate, not a confirmed promotion.

Chronological window scores remain visible: through2004→2005–2006 improves .863248→.871795; through2006→2007 regresses .939394→.909091; through2007→2008–2010 improves .873874→.891892. Validation labels are completely unchanged. On an average pooled repeat, 4 rows improve in sample F1 and 3.75 worsen; on Train157, 3.125 improve and 2.375 worsen. [Every repeat/window](results/repeat_metrics.csv), [per-label metrics](results/label_metrics.csv), [change counts](results/complementarity.csv).

This is evidence of some useful complementarity on known labels; the 49 standalone test differences or close Kaggle score alone did not establish it. Prior 50/50 blends with Cand1/Cand5 were different tested recipes and generally lost locally. No test labels or inferred correctness gates are used here.

## Prepared exploratory CSV

[anchored80_cand2_inductive20.csv](../../submissions/candidates/anchored80_cand2_inductive20.csv) changes **14/312** test label sets from anchored. The user subsequently reported **0.89935** for this exact file. The final fit uses 235 verified labels and 1,014 masked training rows; actual test inputs are prediction-only.

SHA-256: `c20aa937fa8b1e5aea55bcd6564c6acd952f0d74030cc7c1a437eac54a1ee257`.

The export freshly refits anchored and exactly reproduces its scored CSV labels. It reuses the hash-verified Cand2 final probability packet; reconstructing Cand2's CSV from that packet reproduces the exact hash associated with .89102. The blend settings are unchanged from the local diagnostic. [Export protocol](blend_export_protocol.json), [artifact record](blend_prepared_submission.json), [exporter](export_blend.py).

## Verification and reproduction

- 54 fresh anchored controls and 270 Cand2 seed fits: 50 A folds, three B cuts and original validation. All parent label predictions match their prior references. Maximum raw parent score difference is approximately 2e-11, below the frozen 1e-10 tolerance. The model and input hashes remained fixed.
- Model fitting excludes actual query/test rows. To evaluate the existing J definition, its original outer duplicate overlaps and all-year masked pools remain explicit. Cand2's nested calibration and feature generation use the same group-safe adapter as the submitted model; inner overlap counts are zero.
- Independent sample-F1 and TP/FP/FN arithmetic checks the sklearn output, full repeat coverage and per-label precision/recall/F1/support. Current-policy control metrics reproduce the prior audit. The newly fitted original-validation Cand2 predictions and scores match the previous strict five-seed case.
- Full compact predictions are retained because these are useful matched benchmark controls and a concrete exploratory candidate. No fitted estimators or transformed matrices are retained. [Execution notes](execution_notes.json), [verification](results/verification.json), [retention hashes](retention.json).

Use Python 3.12.13 and [requirements-lock.txt](requirements-lock.txt). From the repository root:

```sh
# Retained evidence only; no model fitting:
python part1/experiments/cand2_followup_v1/verify_retained.py

# Fresh J diagnostic in a disposable directory:
python part1/experiments/cand2_followup_v1/run.py --output /tmp/cand2-j-replay --workers 4
python part1/experiments/cand2_followup_v1/analyze.py --output /tmp/cand2-j-replay
```

The canonical blend evaluation reuses retained component fits; it is score recombination rather than fresh training. The runner's console prints means of individual window records; use the retained summary for the pooled98 chronology metric. A future export must preserve this exact candidate file and use a new version rather than overwrite it. No uploads, pushes or merges were performed.
