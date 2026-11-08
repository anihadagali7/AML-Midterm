# Part 1 report handoff

The user has closed this model-search phase and selected **80% anchored /20% strict-inductive five-seed Cand2** as the report winner. Its user-reported Kaggle sample-wise F1 is **0.89935**. The complementary kernel-head experiment is complete at **0.89316** and belongs in the unsuccessful-experiment discussion. Further model experiments are not queued.

## Exact winning checkpoint

- Result commit: `cde3d500e1304f2afeeb6dc4c7f50e49190c6a4b`.
- Annotated score tag: `part1-anchored80-cand2-inductive20-trainval235-kaggle-0.89935`.
- Submission path at that tag: `part1/submissions/candidates/anchored80_cand2_inductive20.csv`.
- CSV SHA-256: `c20aa937fa8b1e5aea55bcd6564c6acd952f0d74030cc7c1a437eac54a1ee257`.
- Result record at that tag: `part1/submissions/results/anchored80-cand2-inductive20-trainval235.json`.
- Frozen export recipe: `part1/experiments/cand2_followup_v1/blend_export_protocol.json`, SHA-256 `8b4e2e9e69c2fddf026038a883411aa3c4e59cb870025d01c5fe2729a4b76cb4`.
- Frozen preparation source: `ef4d7e0e`; the later score record supersedes its original unsubmitted status.

The complete winner source/evidence checkout is `/Users/matthew/Desktop/michael/AML-Midterm-cand2-followup`, branch `codex/part1-cand2-followup`. The current complementary-head branch preserves this later failed experiment; it does not contain a merge of that winner branch. The immutable winner tag remains accessible in the shared repository.

Useful local links: [winning CSV](/Users/matthew/Desktop/michael/AML-Midterm-cand2-followup/part1/submissions/candidates/anchored80_cand2_inductive20.csv), [winning recipe, benchmark and J audit](/Users/matthew/Desktop/michael/AML-Midterm-cand2-followup/part1/experiments/cand2_followup_v1/README.md), [exact result](/Users/matthew/Desktop/michael/AML-Midterm-cand2-followup/part1/submissions/results/anchored80-cand2-inductive20-trainval235.json).

Portable lookup from any worktree in this repository:

```sh
git show part1-anchored80-cand2-inductive20-trainval235-kaggle-0.89935:part1/submissions/results/anchored80-cand2-inductive20-trainval235.json
git show part1-anchored80-cand2-inductive20-trainval235-kaggle-0.89935:part1/experiments/cand2_followup_v1/README.md
```

## Model and evaluation facts for the report

The winning final artifact fits on **235 supplied gold labels plus 1,014 originally masked training rows**. Its predictors use the 150 supplied stopping-reason embeddings. Actual Kaggle test inputs are prediction-only; IDs align rows, and chronology uses years for evaluation partitions. No external trial labels or deep learning are used.

Anchored combines logistic regression, cosine neighbor votes and shrinkage LDA, with clustering supporting confident pseudo-label generation. Cand2 uses the existing strict-inductive five-seed pipeline and produces five label marginals. The final rule is the fixed arithmetic combination `0.8*anchored_scores + 0.2*Cand2_marginals`, followed by threshold .3 and argmax fallback. These are selection scores, not demonstrated calibrated probabilities. The winner uses neither the newly tested diverse kernel head nor a learned final combiner.

| Model | Train157 CV | Pooled235 CV | Validation78 | Chronology98 | Reported Kaggle |
|---|---:|---:|---:|---:|---:|
| **Selected anchored80/Cand2_20** | **.935244** | .921631 | .891026 | **.887755** | **.89935** |
| Previous anchored75/LDA25 | .932059 | **.923404** | .891026 | .884354 | .89850 |
| Anchored | .932059 | .921365 | .891026 | .884354 | .89636 |
| Fixed anchored80/diverse-kernel20 | .933386 | .920922 | .886752 | .870748 | .89316 |

Both grouped CV pools use eight saved repeats of five folds, embedding-group separation and masked-duplicate purges. Report the mean sample-wise F1 computed on each complete repeat, not a mean of individual fold F1 values. Original validation is a separate 157-label fit; the final235 artifact includes those former validation labels. Chronology pools the three fixed earlier-to-later windows, totaling 98 predictions. See [TESTING_POLICY.md](TESTING_POLICY.md).

The winning blend's benchmark combines retained held-out component scores from these exact plans. It does not represent fresh refits of both components for every blend comparison. Keep that distinction separate from the later complementary-head studies, which did refit their anchored and kernel models within each nested fitting partition.

The winner improved both grouped means versus anchored, but its wins/ties/losses were **5/1/2 on Train157 and 4/1/3 on Pooled235**. It did not satisfy the original >=5/8 wins in both pools rule. The user authorized an exploratory submission and selected the winner from the subsequently reported Kaggle result. Preserve that distinction; do not claim unanimous local superiority or retroactively rewrite the advancement rule. The previous LDA blend retains the higher Pooled235 mean.

All supplied labels and saved splits have influenced development. Kaggle scores are user-reported; the leaderboard partition and actual uploaded bytes were not independently inspected. Report the measured results and development limitations without claiming a new untouched holdout or statistical significance from overlapping repeats. The gain over anchored is .00299; the gain over the previous .89850 winner is .00085.

The existing `part1.ipynb` still explains and refits the anchored component. It is not a complete executable account of the final Cand2 blend. Use the winner's frozen exporter and source/evidence references when writing the final method section. Older root report and notebook statements must be reconciled during report writing rather than presented as an already completed report update.

## Completed complementary-head experiment

- `735ad7a3`: v1/v2 source, frozen settings, complete benchmark summaries, negative findings, verification and useful matched predictions.
- `a2b0aa62`: authorized fixed-blend final235 preparation, exact CSV, hash, recipe and prediction-only test access.
- `f9016b9b`: separate user-reported .89316 result.
- Annotated score tag: `part1-anchored80-diverse-kernel20-trainval235-kaggle-0.89316`.
- Exact artifact: [fixed diverse blend CSV](submissions/candidates/anchored80_diverse_kernel20_trainval235.csv), SHA-256 `97f311bc840843876e42f203a9c47d350e5c9df6a0ceb2c84ff4e35e6f79ff54`.
- [Experiment and interpretation](research/complementary_heads_v2/README.md), [preparation](research/complementary_submission_v1/README.md), [score record](submissions/results/anchored80-diverse-kernel20-trainval235.json), [closeout audit](research/complementary_submission_v1/closeout.json).

Each version completed all 84 outer cases. Independent verification covered 252 sample/macro metric records, 1,260 per-label records and 1,008 nested partitions per version. Matched predictions and exact submitted artifacts are retained; unsuccessful fold caches were removed after verification, with all compact summaries and recipes preserved. No fitted estimator dumps remain in this study.

The stronger diversity penalty improved its own ordinary-head fixed-blend control in both grouped pools, but no variant passed against the .89935 leader. Learned combination did not improve the result. The manual .89316 score agrees with the local direction against the leader. It does not isolate the penalty's Kaggle effect because the ordinary-kernel control was not submitted; it also does not test joint neural-head training or hidden-layer removal. This was a classical fixed-anchor adaptation.

The research commits and score tags remain local. This handoff does not merge research branches or change either immutable score tag. Continue with report writing around the selected winner.
