# Fixed complementary-head submission

**User-reported Kaggle score: 0.89316**, below the .89935 leader by .00619 and standalone anchored's .89636 by .00320. The [exact-file result record](../../submissions/results/anchored80-diverse-kernel20-trainval235.json) supersedes the frozen preparation-time status.

The user requested the best candidate to try despite the negative advancement result, then explicitly chose the fixed blend. The selected recipe is `complementary-heads-v2/blend_diverse30`: **80% anchored raw scores +20% diverse kernel probabilities**, with diversity penalty .3, a .3 label cutoff and argmax fallback. This is the fixed blend, with no learned combiner applied to its predictions.

[Submission CSV](../../submissions/candidates/anchored80_diverse_kernel20_trainval235.csv) contains 312 unique IDs in the supplied sample order. SHA-256:

`97f311bc840843876e42f203a9c47d350e5c9df6a0ceb2c84ff4e35e6f79ff54`

The full refit uses all **235 supplied labels** and **1,014 originally masked rows**, in original train-then-validation order. Actual test embeddings enter prediction only. The second head's supervised loss uses gold labels; feature scaling and the kernel map use permitted fitting inputs. Its residual objective uses anchored predictions from three group-safe folds. The exact benchmark entrypoint is reused, including its nested diagnostic fits; only the selected fixed blend is exported. No settings were selected from test outputs.

The [export protocol](export_protocol.json) was frozen before fitting and inspecting test predictions. [Benchmark source and results](../complementary_heads_v2/README.md) are committed at `735ad7a3`. The user's exploratory authorization does not change the failed automatic advancement status:

| Recipe | Train157 CV | Pooled235 CV | Validation78 | Chronology98 |
|---|---:|---:|---:|---:|
| Current .89935 submission | .935244 | .921631 | .891026 | .887755 |
| Selected fixed diverse blend | .933386 | .920922 | .886752 | .870748 |

The selected recipe ties the strongest Train157 result and has the highest Pooled235 result among v2's combined candidates. All local checks are reused development evidence; the previous validation score belongs to the 157-label fit, not independent validation of this final235 artifact.

The export reproduced the unchanged anchored test scores exactly and reproduced the exact .89935 parent CSV hash using its retained Cand2 scores. Those reference scores never enter candidate fitting. The new file changes **12 of 312 label sets** versus that parent: adds Safety4, Efficacy2 and Recruitment2; removes Recruitment3 and Other4. These changes do not establish whether any test label is correct. See [changed rows](changed_predictions.csv) and the [preparation record](prepared_submission.json).

Independent [verification](verification.json) checks exact CSV bytes, sample order, binary/nonempty label sets, the arithmetic blend and both decoder representations, protected source/data hashes, and all 12 nested full235 fitting boundaries.

```sh
python part1/research/complementary_submission_v1/verify_fixed.py
```

The submitted recipe regressed in the direction indicated by all four local benchmark comparisons against the .89935 leader. This does not isolate the diversity penalty: its matched ordinary-kernel control was not submitted. The learned combiner and joint neural-head proposal were not tested by this upload.

Do not overwrite this export or transfer its score to different bytes. The result has a separate commit and immutable annotated tag `part1-anchored80-diverse-kernel20-trainval235-kaggle-0.89316`. Submission was manual; the agent performed no upload, push or merge. Frozen preparation manifests retain their original status.
