# Anchored75/LDA25 submission preparation

The user authorized a manual Kaggle trial of the fixed blend after reviewing
its local results. The prepared file is
[part1_anchored75_lda25_trainval235.csv](../../submissions/ready/part1_anchored75_lda25_trainval235.csv).
It has **312 rows** and changes **five label sets** from the submitted anchored
0.89636 model. It is ready for upload; no upload or Kaggle score is recorded.

SHA-256: `6331d245a73148ab8d155623059f60371e28a4b25312bd12c8e23fb926c20c8b`.

## Exact recipe and fitting population

This is the same fixed recipe evaluated in the
[complete neighbor/blend study](../anchored_neighbor_blend_v1/README.md):

1. Use the existing anchored model fitted on all **235 verified labels** and
   the 1,014 original masked training rows. Reload that fitted model and
   recompute test scores, checking them against its frozen score CSV and exact
   submitted predictions.
2. Use the frozen full-precision scores from the existing **LDA 70/15/15
   ensemble**, also fitted using all 235 verified labels and the 1,014 masked
   rows. Verify its historical artifact hashes, fitting IDs, scores and scored
   submission. This is the complete LDA ensemble, not a single LDA classifier.
3. Center anchored's native 0.3 cutoff to 0.5 using the preregistered logit
   transform, then mix **75% anchored + 25% LDA ensemble**. Decode at 0.5 with
   nonempty argmax fallback.

Both component fits are inductive; test inputs were excluded from fitting.
This preparation performs **zero new model fits** and reuses the already
verified final-235 components. There is no additional learned blend layer,
weight search or threshold adjustment. The original fitting audits and exact
component sources are referenced in [frozen_recipe.json](frozen_recipe.json).

## Local evidence and submission status

The blend improves Pooled235 CV from .921365 to .923404, winning all eight
repeats, and ties the parent's mean Train157 sample F1 (.932059), validation
sample F1 (.891026), and chronology (.884354). Train157's paired result is
3 wins / 3 ties / 2 losses, so it does not pass the study's strict both-pool
advancement rule. Validation macro F1 also decreases. The full history and
all label metrics remain in the study report.

The user explicitly chose a manual trial after discussing this distinction.
Preparing this CSV does not change the rule or promote the model. The scored
anchored leader remains unchanged pending a reported Kaggle result.

## Verification and retention

[Verification](verification.json) independently checks CSV columns, unique IDs,
sample ordering, allowed nonempty labels, component and output hashes, and
reproduces the blend with the algebraically equivalent rational transform
`0.7*p / (0.7*p + 0.3*(1-p))`. Every decoded prediction matches. All 53 pinned
source/artifact hashes passed. The smallest score distance from 0.5 is
approximately .000407, well above the numerical verification tolerance.

Run from the repository root:

```sh
python part1/research/anchored_lda_blend_submission_v1/verify.py
```

Keep the exact ready CSV, [manifest](manifest.json), frozen recipe, preparation
and independent verification scripts, full component/blend score arrays, and
[five-row change report](changes_vs_anchored.csv). Existing fitted component
artifacts are referenced without duplication. Preparation-time manifests remain
immutable after a score is reported; record the score in a separate result
commit and annotated score tag under the normal submission workflow.
