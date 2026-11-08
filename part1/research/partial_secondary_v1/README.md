# Keeping unselected Other pseudo-labels unknown

**Outcome: the change recovers more Other labels, but none of the five fixed variants improves both grouped benchmarks. Keep the user-reported .89935 blend. No new Kaggle candidate was exported or uploaded.** All 84 outer cases finished, and independent verification passed.

| Recipe | Train157 | Pooled235 | Validation78 | Chronology98 | Pooled Other F1 |
|---|---:|---:|---:|---:|---:|
| Current .89935 blend | .935244 | .921631 | .891026 | .887755 | .621284 |
| Unknown Other in neighbor votes | .932856 | .919149 | .891026 | .885714 | .611043 |
| Binary Other, complete-label control | .931794 | .919681 | .882479 | .885714 | .611093 |
| Binary Other, unknown secondary labels | .920382 | .914982 | .891026 | .901020 | .588698 |
| Complete-label control, half strength | .933652 | .917553 | .882479 | .885714 | .597025 |
| Unknown-label version, half strength | .928875 | .917376 | .878205 | .892517 | .595786 |

The main unknown-label model loses to the scored parent in **all eight Train157 repeats and all eight Pooled235 repeats**. Its sample-F1 deltas are -.014862 and -.006649, respectively. Against the matched complete-label architecture, its deltas are -.011412 and -.004699, with seven losses and one win in each pool. Therefore the grouped regression is not explained solely by changing classifier architecture.

The effect is a recall/precision tradeoff. On pooled CV, mean Other recall rises **.546429 -> .610714**, but precision falls **.723841 -> .568720**. Mean true Other detections rise from19.125 to21.375 of35 per repeat, while false positives rise **7.5 ->16.25**. Mean Recruitment+Other detections rise from **.25 ->1.125 of9** per repeat; most of these rare combinations remain missed. The binary unknown model generates an average16.775 multi-label pseudo rows per pooled fold versus8.25 for its matched complete-label architecture. These generated labels have no supplied truth, so increased cardinality is not itself evidence of correctness.

The temporal result is better: chronology sample F1 rises **.887755 -> .901020**, and Other F1 rises **.700000 -> .755556** (17/20 true Other detections instead of14/20, with8 false positives instead of6). Validation sample F1 ties; its Other F1 rises .470588 -> .560000. These checks show the method changes the intended behavior and may help some later-year cases, but do not override the consistent grouped-CV regression. All three chronological windows remain in the saved tables.

## Fixed experiment

This branch starts from main `766d16d3`. The comparison parent is the arithmetic anchored80/Cand2_inductive5_20 submission, user-reported Kaggle .89935, recorded at `cde3d500`. The exact benchmark input arrays are copied with provenance from `0243e9ac`. Main's .89850 documentation refers to an older merged leader. Original anchored .89636 remains a secondary control.

The prior study found that anchored's final235 fit generated757 pseudo rows,756 with one label and only1 with two. Michael authorized testing whether an unselected secondary reason should remain unknown. This study focuses on **Other**, retaining complete treatment for the other four pseudo labels and preserving every supplied label.

The neighbor variant excludes pseudo Other=0 from Other neighbor retrieval, while retaining gold negatives and all observed positives. It retrieves the ten closest *observed* Other labels. Its original LR, LDA and cluster teacher are otherwise unchanged. This can broaden the Other neighborhood; it is not simply quarter-weighting all pseudo neighbors, which was previously tested unsuccessfully.

The binary variants replace only the Other outputs of LR and LDA with independent binary classifiers, allowing individual missing labels to be omitted from their losses. The complete-label control treats every pseudo Other0/1 as observed. The unknown-label variant omits pseudo Other0 from those classifiers, Other neighbor retrieval and the Other cluster-rate estimate. Original geometry and the other four class outputs remain; they can change indirectly when pseudo labels are regenerated.

Binary LR uses C.1 and raw gold/pseudo weights1/.25. Within the observed rows, weight masses are normalized to fixed .8 negative/.2 positive priors and total gold-row count. This prevents removing unknown negatives from automatically changing the assumed class prior or effective regularization mass. Binary LDA uses weighted shrinkage covariance and fixed .8/.2 priors. These settings resemble the original equal-five-class Other prior in intent, but the models are not mathematically equivalent; the matched architecture control is essential.

All models keep original scaling, PCA30, GMM30, seeds, two pseudo refreshes, row confidence>=.8, positive-label selection .3+argmax, and final .3+argmax decisions. Full variants blend altered anchored80/unchanged Cand2_20. The two fixed half-strength arms blend original anchored40/altered anchored40/Cand2_20. No thresholds, weights or model settings were adjusted after outcomes.

## Evaluation and verification

All five variants received the exact saved eight-repeat five-fold grouped plans in both pools, original validation78 and three chronological windows. Scaling, clusters and pseudo labels were refitted on eligible training inputs in every case. Held-out gold and matching masked embedding groups were excluded under the saved policy; original validation preserves its historical no-outer-purge boundary. Prediction functions received query embeddings only. Actual Kaggle test inputs were never loaded during this work.

There were84 fresh anchored fits,252 altered-model fits and one no-change subclass check. Half-strength blends reuse scores. The84 exact strict five-seed Cand2 component evaluations were reused with frozen provenance, IDs, truth and plan alignment; they were not refitted. Fresh anchored scores match the retained controls exactly on all84 cases.

The predeclared advancement rule was positive mean sample-F1 change and at least5/8 wins in **both pools against the .89935 parent**. All variants failed. Supplementary checks for every variant were explicitly registered as a diagnostic exception. These are heavily reused development labels, not a fresh holdout.

[Independent verification](results/summary/verification.json) rescored147 complete-repeat/window records and every per-label metric, checked504 partial-label rounds, preserved gold-negative observations and fixed class-weight masses, verified IDs/group boundaries and half-blend formulas, and reproduced all four parent headlines. [Math checks](math_check.json) establish that unknown neighbors do not vote zero and weighted binary LDA matches sklearn under unit weights.

All compact scores, paired differences, per-label results, subgroup counts and fit audits are under [results/summary](results/summary/). [Retention](retention.json) keeps the useful parent/complete-control/unknown-model comparison and removes unsuccessful per-case caches after verification.

## Reproduction

Use Python3.12.13 and the Part1 numerical requirements recorded in [protocol.json](protocol.json). From the repository root:

```sh
python part1/research/partial_secondary_v1/study_run.py run --workers 4 --output /tmp/partial-secondary-replay
python part1/research/partial_secondary_v1/verify_partial.py /tmp/partial-secondary-replay
python part1/research/partial_secondary_v1/evidence_partial.py verify
```

The replay requires a fresh output directory and preserves frozen settings. The last command checks retained evidence without retraining. The standalone `check_math.py` writes its receipt; rerun it in a disposable checkout to preserve the original recorded receipt. No model was promoted, no scored artifact was overwritten, and no upload, push or merge was performed.
