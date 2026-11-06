# Part 1 integrated research handoff

This branch combines the logistic/semi-supervised work at `7425345` with the independent `AMLMidterm` research. The [central testing policy](TESTING_POLICY.md) governs new comparisons. Frozen originals remain in their own directories; source code, recipes, and submission predictions have not been retuned during integration.

**Highest reported Kaggle result remains Ani's original 0.89423.** Our submitted 70% Ani / 15% classical tri-training / 15% SVM ensemble scored **0.89401**, despite stronger local comparisons against the inductive Ani adaptation. Those two submitted pipelines have different input-access policies. We have not established a model that beats Ani on Kaggle.

## Comparable local results

These are sample-wise F1 values on the same grouped eight-repeat protocols. Each repeat pools all OOF predictions. SD, paired comparisons, and exact configurations are in the linked artifacts. Validation is the original 78-row check using the 157-label fit; it is not the score of the final 235-label refit.

| Recipe | Train157 CV | Pooled235 CV | Validation78 | Chronological98 | Reported Kaggle |
|---|---:|---:|---:|---:|---:|
| Ani inductive control | .919586 | .909840 | .891026 | .860544 | Unknown for this access mode |
| 75% Ani / 25% tri-training | .926752 | .916401 | .888889 | .884354 | Unsubmitted |
| 70% Ani / 15% tri / 15% SVM | .925690 | .916489 | .910256 | .877551 | .89401, 235 labels, inductive |
| 75/25 plus half subset average | .928875 | .918794 | .880342 | .870748 | Unsubmitted |
| Classical tri-training | .908758 | .898901 | .903846 | .879252 | .88012, 235 labels, inductive |

Sources: [six-recipe comparison](research/independent/experiments/cross_repo_benchmark_v1/confirmation8/comparison.csv), [continued-search comparison](research/independent/experiments/continued_search_v1/final_comparison.csv), and [submission ledger](submissions/index.csv). The three-way ensemble remains a useful balanced reference. Subset averaging raises both CV means but loses on both supplementary checks; it is not a clear overall replacement. The historical `Current best hybrid` name refers to an older validation-selected model, not the winner of this table.

The logistic branch's older 40-fold benchmark used pooled235 ordinary KFold seeds 100–107. Ani .908865 and DBSCAN .888316 belong to that protocol. They are retained in [its historical benchmark](experiments/paired_recipe_cv/BENCHMARK.md) and must not be treated as the grouped scores above.

## Submission provenance and Ani branch changes

The [submission guide](submissions/README.md), [CSV index](submissions/index.csv), and `archive_manifest.json` now include the original Ani artifact and both independent scored submissions alongside the existing logistic/DBSCAN history. Scores were reported by the user/branch records; integration did not upload or query Kaggle.

Ani's reported .89423 corresponds to the original recipe/artifact at commit `a548721`. That recipe used all 235 gold labels and transductive input access. Its original CSV is byte-identical to `part1_first_submission.csv` in the newer Ani commit `8a8c3fc`. The newer commit changes `part1_best_model.ipynb`, generated labels, and `part1_best_submission.csv`. **No score for that changed CSV is established here.** Both source snapshots and exact hashes are preserved under [ani_snapshots](research/ani_snapshots/README.md). Do not attach the old .89423 score to the newer `best` filename or assume our inductive control represents that revision.

The user reported two remaining submissions after the .88012 and .89401 uploads. No additional submission was created by the transductive experiment. Historical Git tags remain unchanged.

## What the latest diagnostics found

The [refit diagnostic](research/independent/experiments/refit_diagnostic_v1/report.html) held pooled-CV evaluation rows fixed while adding eligible validation-source gold labels to fitting. Three-way mean F1 rose from .913032 to .916489, winning five of eight repeats, but gains differed by source cohort. Actual 157-label versus 235-label fits changed 29/312 test label sets. Refitting can materially change predictions; these results do not establish the direction of its Kaggle effect.

The [transductive diagnostic](research/independent/experiments/three_way_transductive_v1/report.html) kept the submitted recipe fixed and compared Ani-only versus all-component target-input fitting. Full transduction reached train CV .926486 and pooled CV .918085, but won only four of eight repeats in each pool. Validation fell to .897436; chronology rose to .884354. Its final fit changed 15/312 test label sets. Neither arm met the predeclared CV eligibility rule, so no submission CSV was exported. Local target batches had all labels erased; real test labels were never available. This is a separate access regime, not a replacement for the inductive benchmark.

The [architecture review](research/independent/experiments/architecture_review_v1/report.html) found 26/235 persistent exact-label-set errors shared by Ani and the three-way model across at least six of eight repeats. Other remains weak (F1 about .618, recall .55); Funding is also weaker than the common labels. The inventory contains 2,293 fit/evaluation records and 1,840 distinct configurations; its 68 family identifiers include variants, not 68 independent architectures.

## Next hypotheses, with existing work accounted for

1. **Structured eligibility text:** separate inclusion from exclusion, model eligibility burden, and validate numeric parsing before adding clinical bounds. First measure metadata-only signal under the shared folds. Plain whole-text features have already been tried.
2. **Separate embedding and metadata models:** retain a rotation-tolerant embedding model and test a small, regularized metadata model. Only if it adds complementary signal should we try a gate using annotation coverage/uncertainty, trained within inner CV. Raw/PCA tree models, co-training, and fixed blends are already represented in past trials.
3. **Target persistent multilabel errors:** test a new representation only when error patterns support it. Classifier chains, label powerset, and joint score decoders have already been explored; those names alone are not new experiments.

The independent review suggested correctly parsing ICD codes because its generic text tokenizer split code suffixes. **The logistic branch already unpacked nested ICD lists into three-character categories** and tested minimum-count 3/5 features with fixed DBSCAN targets. Its [feature screen](experiments/remaining_classifier_features/SUMMARY.md) found no improvement over .910256 validation F1 (best ICD variant .895299). Reuse that parser and evidence; full codes or alternate hierarchies would be a new, explicit hypothesis, not a claim that ICD parsing has never been tested. That older screen was validation-driven, so a useful new representation still needs the shared CV protocol.

No architecture/feature experiment from this next-step list has been run as part of integration. No extra data or deep learning is planned. The provided reason vectors are rotated embeddings; raw stopping-reason text is unavailable. Drug/action/target sets do not provide reliable pairwise links, and pathway depths alone do not provide missing parent edges.

## Working from the combined tree

- [Research README and reproduction instructions](research/independent/README.md)
- [Import manifest with per-file source hashes](research/independent/IMPORT_MANIFEST.json)
- [Integration verification](research/independent/integration_verification.json)
- [Current agent conventions](AGENTS.md)

Run scripts from `part1/research/independent` in the recorded environment. Keep future runs in new versioned directories; do not overwrite hashed historical sources or protocols. The integration preserves research and reproducibility, but final course packaging still needs a notebook/report update for the eventual selected model. Existing numbered notebooks describe earlier checkpoints.
