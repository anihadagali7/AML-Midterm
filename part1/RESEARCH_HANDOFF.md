# Part 1 integrated research handoff

This is the main-project entry point for the consolidated LDA research. Readable source code and reports are under `research/lda_ensemble/`; leader/reference predictions and frozen dependencies are retained directly. Unsuccessful runs retain settings, source and result summaries; their per-fold outputs and the compressed archive were discarded. Use the [retention and replay guide](research/lda_ensemble/EVIDENCE.html), including its disposable-output replay wrapper. The anchored-model files remain in their existing Part 1 locations. The redundant `AML-Midterm-integration` worktree has been removed after preservation and replay checks; use this main folder for further work. [All research entry points](RESEARCH_INDEX.html). No branches were merged.

This branch combines the logistic/semi-supervised work at `7425345` with the independent `AMLMidterm` research. The [central testing policy](research/lda_ensemble/TESTING_POLICY.md) governs new comparisons. Frozen originals remain in their own directories; source code, recipes, and submission predictions have not been retuned during integration.

**Latest recorded Kaggle leader: the separate anchored reason classifier, 0.89636.** The user reported submitting a different, better model; its [exact-file result record](experiments/paired_improvement/candidate/kaggle_result.json) is in this `AML-Midterm` project. Its local candidate and scored archive both hash to `1db6433dc5a5b900d440db16e5ef8375eb51dcf02d3f270217527d8b174319eb`. This reported result exceeds original Ani's .89423 by .00213 and this branch's submitted LDA ensemble's .89401 by .00235. The score was reported by the user, not independently queried from Kaggle. This is a separate model; it does not change the outcomes of this branch's experiments.

**Experiment documentation checkpoint:** the [complete study index](research/lda_ensemble/experiments/lda_research_checkpoint_v1/index.html) catalogs the feature work, LDA comparisons, negative results, failed versions, saved predictions, verification records and final submission provenance. The current retention manifest checks the files still kept; older verification records describe checks performed before unsuccessful-run outputs were discarded. The [machine-readable catalog](research/lda_ensemble/experiments/lda_research_checkpoint_v1/catalog.json) records each study's exact parent, selection procedure, evidence and status. These completed studies are historical evidence to carry forward to the anchored-model work, not experiments already rerun against that newer model.

**This branch's scored submission:** the final235 inductive LDA ensemble scored **0.89401**, reported by the user after uploading the prepared file. It ties the previous ensemble at displayed precision. The files differ on 13/312 predicted label sets; this was not an identical local CSV. See the [exact-file result record](research/lda_ensemble/experiments/final_lda_ensemble_trainval235_v1/kaggle_result.json) and [prediction changes](research/lda_ensemble/experiments/final_lda_ensemble_trainval235_v1/prediction_changes_vs_previous.csv). The user subsequently confirmed one daily slot remained, then reported submitting the different, better model. The earlier remaining-slot estimate is historical; do not assume a submission remains available.

**Follow-up after the Kaggle tie:** [14 frozen temporal and decision variants](research/lda_ensemble/experiments/temporal_generalization_review_v1/report.html), evaluated as 28 parent-specific comparisons on all eight repeats in both pools, produced **no confirmed ensemble upgrade** and no candidate meeting the predeclared temporal eligibility rule. A recent-weighted inner-learned shared cutoff raises chronology slightly to .880952, with pooled CV and validation unchanged, but the gain is confined to one window. No additional final235 fit, test prediction or submission was created.

**This branch's confirmed CV improvement:** an inductive adapter of Ani's later `8a8c3fc` LDA revision inside the 70/15/15 ensemble reaches **.931794 train / .921631 pooled**, winning 7/8 paired repeats in both pools against the earlier submitted ensemble. Validation falls slightly to .905983; chronology ties at .877551. See the [revision review](research/lda_ensemble/experiments/lda_revision_review_v1/report.html). Its later Kaggle result was a tie, not an improvement. This statement describes this branch's comparisons and does not rank the newer anchored classifier's local results.

**Direct comparison with original Ani:** the [fresh matched comparison](research/lda_ensemble/experiments/ani_direct_comparison_v1/report.html) verifies the `a548721` source functions behind the .89423 submission. Under identical inductive access, the current ensemble beats original Ani by **+.012208 train / +.011791 pooled**, winning **8/8 repeats in both pools**, with higher validation and chronology too. When both methods can use the same unlabeled target batch, the gains remain **+.010881 / +.010461**, with **7/8 and 8/8 wins**. The earlier omission of original Ani's local scores from the user-facing comparison was a presentation gap, not an absence of an inductive baseline. The new study freshly refits and verifies both access regimes.

**Final fit and scored artifact:** [final LDA ensemble CSV](research/lda_ensemble/experiments/final_lda_ensemble_trainval235_v1/part1_lda_ensemble_trainval235.csv), SHA-256 `871bd7faff1380c13b4e099f725372399b40bf3954c70854efe9f3bad7d3f7a2`. After explicit user authorization, the frozen inductive 70/15/15 recipe was fitted on 235 gold labels plus 1,014 original masked training rows and generated 312 predictions. Test inputs entered prediction only. Notebook parity, reproduction of the previous submitted ensemble, small-query-batch invariance, sample ID order, valid labels and protected hashes passed; see the historical preparation [manifest](research/lda_ensemble/experiments/final_lda_ensemble_trainval235_v1/manifest.json) and [verification](research/lda_ensemble/experiments/final_lda_ensemble_trainval235_v1/verification.json). The user subsequently reported **.89401 Kaggle**. The final235 fit still has no independent local validation score. Historical experiment statements about no final refit describe their original runs; the export and reported result are separate artifacts. Historical submission files, preparation manifests and the frozen ledger remain unchanged.

## Comparable local results

These are sample-wise F1 values on the same grouped eight-repeat protocols. Each repeat pools all OOF predictions. SD, paired comparisons, and exact configurations are in the linked artifacts. Validation is the original 78-row check using the 157-label fit; it is not the score of the final 235-label refit.

| Recipe | Train157 CV | Pooled235 CV | Validation78 | Chronological98 | Reported Kaggle |
|---|---:|---:|---:|---:|---:|
| Ani inductive control | .919586 | .909840 | .891026 | .860544 | Unknown for this access mode |
| 75% Ani / 25% tri-training | .926752 | .916401 | .888889 | .884354 | Unsubmitted |
| 70% Ani / 15% tri / 15% SVM | .925690 | .916489 | .910256 | .877551 | .89401, 235 labels, inductive |
| 70% latest Ani LDA / 15% tri / 15% SVM | .931794 | .921631 | .905983 | .877551 | .89401, 235 labels, inductive |
| 75/25 plus half subset average | .928875 | .918794 | .880342 | .870748 | Unsubmitted |
| Classical tri-training | .908758 | .898901 | .903846 | .879252 | .88012, 235 labels, inductive |

Sources: [six-recipe comparison](research/lda_ensemble/research/independent/experiments/cross_repo_benchmark_v1/confirmation8/comparison.csv), [continued-search comparison](research/lda_ensemble/research/independent/experiments/continued_search_v1/final_comparison.csv), and [submission ledger](research/lda_ensemble/submissions/index.csv). The three-way ensemble remains a useful balanced reference. Subset averaging raises both CV means but loses on both supplementary checks; it is not a clear overall replacement. The historical `Current best hybrid` name refers to an older validation-selected model, not the winner of this table.

The logistic branch's older 40-fold benchmark used pooled235 ordinary KFold seeds 100–107. Ani .908865 and DBSCAN .888316 belong to that protocol. They are retained in [its historical benchmark](research/lda_ensemble/experiments/paired_recipe_cv/BENCHMARK.md) and must not be treated as the grouped scores above.

## Submission provenance and Ani branch changes

The [submission guide](research/lda_ensemble/submissions/README.md), [CSV index](research/lda_ensemble/submissions/index.csv), and `archive_manifest.json` now include the original Ani artifact and both independent scored submissions alongside the existing logistic/DBSCAN history. Scores were reported by the user/branch records; integration did not upload or query Kaggle.

Ani's reported .89423 corresponds to the original recipe/artifact at commit `a548721`. That recipe used all 235 gold labels and transductive input access. Its original CSV is byte-identical to `part1_first_submission.csv` in the newer Ani commit `8a8c3fc`. The newer commit changes `part1_best_model.ipynb`, generated labels, and `part1_best_submission.csv`. **No score for that changed CSV is established here.** Both source snapshots and exact hashes are preserved under [ani_snapshots](research/lda_ensemble/research/ani_snapshots/README.md). Do not attach the old .89423 score to the newer `best` filename or assume our inductive control represents that revision.

The user reported two remaining submissions after the .88012 and .89401 uploads, subsequently reported the LDA .89401 result and one remaining slot, and then reported the separate better submission. Those earlier quota statements are historical. No additional submission was created by the transductive experiment. Historical Git tags remain unchanged.

## Commit and submission checkpoints

The consolidated work is committed locally on the existing `codex/part1-research-integration` branch. [Commit and tag index](research/consolidation_v1/commit_index.json) records the idea batches and five dedicated result checkpoints. The annotated tags identify the exact submitted CSV and recipe hashes. These are backfilled records of earlier submissions, not new uploads. [Verification from committed files](research/consolidation_v1/commit_verification.json) passed; no branch merge or push was performed. Follow [AGENTS.md](AGENTS.md) for future experiment commits and score tags.

## Retention policy

Keep source, fixed settings, evaluation protocols, failures and summary results for all experiments. Keep detailed predictions and useful audits for leaders, scored submissions and matched reference comparisons. Discard unsuccessful-run fold caches, per-trial tables, duplicated inputs and logs after checking their summaries. Replays write to a new disposable directory; delete those outputs after extracting any useful summary. Historical audit counts below describe checks completed before this cleanup, not files still stored.

## What the latest diagnostics found

### Generalization follow-up after the LDA submission tied

Three new, versioned studies retain the submitted LDA ensemble as the matched parent and use only provided Part 1 data. [Temporal feature changes](research/lda_ensemble/experiments/temporal_lda_cv_v1/report.html) test global and within-reason year drift, five/ten-year recency weighting, and time-aware neighbors at prediction or throughout teaching. [Recent-gold experts](research/lda_ensemble/experiments/recent_era_lda_cv_v1/report.html) test four fixed blends with specialists fitted on recent gold, preserving minimum per-reason support, embedding groups and excluded-gold boundaries. [Nested decision learning](research/lda_ensemble/experiments/nested_decision_lda_cv_v1/report.html) learns shared or per-reason cutoffs from three-fold grouped inner OOF predictions, with equal or recent-weighted sample-F1 objectives and a fixed penalty toward the native cutoff. The complete source and both ensemble partners are refitted within every inner training partition.

All 14 ensemble variants receive the original validation and three fixed chronological windows under an exception declared before supplementary scoring. Ordinary CV confirmation remains separate from temporal eligibility. The latter requires chronology gain ≥.005, improvement in at least two windows, validation loss ≤.005, and CV loss ≤.003 in each pool. None met either rule.

| Recipe | Train157 CV | Pooled235 CV | Validation78 | Chronological98 | Paired CV wins vs submitted LDA |
|---|---:|---:|---:|---:|---|
| Submitted LDA ensemble | .931794 | .921631 | .905983 | .877551 | Reference |
| Recent-weighted inner-learned shared cutoff | .932059 | .921631 | .905983 | .880952 | 2/8 train, 2/8 pooled |

The cutoff candidate's chronological gain is +.003401, all in the last window, with two trials improved and two worsened. It is a small lead to preserve, not a demonstrated robust improvement or a submission recommendation. The other year-based changes either tie or worsen chronology. Negative results apply to these implementations, not every possible use of year metadata.

A retrospective breakdown also shows that the previous LDA CV gain was not confined to early trials: on the 37 pooled-CV examples from 2008 onward, the old ensemble scores .849099 and the submitted LDA scores .865991, with six of eight paired wins. This is an overlapping subset of reused pooled CV, not a fresh chronological holdout, and did not select the new recipes.

The [consolidated review](research/lda_ensemble/experiments/temporal_generalization_review_v1/report.html) preserves all results and changed-decision counts. Independent rescoring checked **912 complete-OOF repeat records, 4,560 fold arrays and 92 supplementary records**. An additional audit checked 504 temporal feature records, 336 expert records and 336 cutoff records, reconstructed the learned cutoffs from saved inner predictions, and verified 1,008 inner-fold audit records (252 distinct inner fits shared across four cutoff recipes). Every study reproduced the saved latest-LDA control scores; all protected hashes remained unchanged. No Part 2 fields, external records, actual test inputs, final235 refit or submission were used.

The [refit diagnostic](research/lda_ensemble/research/independent/experiments/refit_diagnostic_v1/report.html) held pooled-CV evaluation rows fixed while adding eligible validation-source gold labels to fitting. Three-way mean F1 rose from .913032 to .916489, winning five of eight repeats, but gains differed by source cohort. Actual 157-label versus 235-label fits changed 29/312 test label sets. Refitting can materially change predictions; these results do not establish the direction of its Kaggle effect.

The [transductive diagnostic](research/lda_ensemble/research/independent/experiments/three_way_transductive_v1/report.html) kept the submitted recipe fixed and compared Ani-only versus all-component target-input fitting. Full transduction reached train CV .926486 and pooled CV .918085, but won only four of eight repeats in each pool. Validation fell to .897436; chronology rose to .884354. Its final fit changed 15/312 test label sets. Neither arm met the predeclared CV eligibility rule, so no submission CSV was exported. Local target batches had all labels erased; real test labels were never available. This is a separate access regime, not a replacement for the inductive benchmark.

The [architecture review](research/lda_ensemble/research/independent/experiments/architecture_review_v1/report.html) found 26/235 persistent exact-label-set errors shared by Ani and the three-way model across at least six of eight repeats. Other remains weak (F1 about .618, recall .55); Funding is also weaker than the common labels. The inventory contains 2,293 fit/evaluation records and 1,840 distinct configurations; its 68 family identifiers include variants, not 68 independent architectures.

### Latest Ani revision and matched follow-ups

The newer Ani source revision had been preserved but had not been evaluated under the shared grouped protocol. The [LDA-revision study](research/lda_ensemble/experiments/ani_lda_revision_cv_v1/report.html) now evaluates an **inductive adaptation**, matching its function definitions on the same allowed fitting data. It adds shrinkage LDA at weight .2, reduces logistic weight to .5 while keeping cosine neighbors at .3, changes logistic C from .3 to .1, and uses that mixture in pseudo-label teaching. The original notebook's test-inclusive transforms, evaluation cells and export cells were not executed.

Eight predeclared parent-specific recipes received all eight repeats in both pools, including C.1 without LDA, LDA with original C.3, and LDA added only at final prediction. The full revision inside the original ensemble improved by **+.006104 train / +.005142 pooled**, with **7/8 wins in each pool**. Against its matching C.1/no-LDA ensemble it improved by +.004777 / +.003901, so the gain is not solely regularization. Validation changed .910256 → .905983; chronology remained .877551. The CV gain is useful, but the later-year check does not yet support a larger generalization claim.

Pooled Funding F1 improved .769493 → .821196. Other precision improved .707766 → .758236 while recall fell .550000 → .539286; the persistent Other blind spot remains. The user reports a competing Kaggle score near .93. Our .931794 train-CV score is not a Kaggle score and does not close that demonstrated leaderboard gap.

Two further studies each tested eight fixed candidates against the **new LDA revision**, retaining original controls too: [gold-only and class-specific covariance heads](research/lda_ensemble/experiments/lda_covariance_cv_v1/report.html), and [direct sample-F1 utility regression versus independent-label regression](research/lda_ensemble/experiments/joint_utility_cv_v1/report.html). Neither produced a further two-pool confirmation. All negative results and supplementary diagnostics remain saved. The [consolidated revision report](research/lda_ensemble/experiments/lda_revision_review_v1/report.html) links all results, matched contrasts, source hashes and independent verification. No test predictions, final235 refit, submission, or promotion was made. The `8a8c3fc` historical submission CSV still has no established Kaggle score.

### Direct original-Ani comparison requested by the user

The [direct-comparison study](research/lda_ensemble/experiments/ani_direct_comparison_v1/report.html) freezes original Ani, the later LDA source alone, and the current 70/15/15 LDA ensemble, under two matched input-access settings. All six combinations are freshly fitted on the same 80 grouped CV partitions, followed by original validation and the three chronological windows. Both preserved notebooks' function outputs are reproduced on every fitting partition. Tri-training and SVM partners are also refitted, and inductive predictions reproduce the previous saved controls.

| Input access | Recipe | Train157 CV | Pooled235 CV | Validation78 | Chronological98 |
|---|---|---:|---:|---:|---:|
| Inductive | Original Ani, `a548721` | .919586 | .909840 | .891026 | .860544 |
| Inductive | Current LDA ensemble | .931794 | .921631 | .905983 | .877551 |
| Same unlabeled target batch for both | Original Ani, `a548721` | .921709 | .911702 | .888889 | .872449 |
| Same unlabeled target batch for both | LDA ensemble | .932590 | .922163 | .903846 | .877551 |

The current ensemble improves all four measurements against original Ani within each access regime. Inductive paired CV gains are +.012208/+ .011791 with 8/8 wins in both pools. Target-input paired gains are +.010881/+.010461 with 7/8 train wins and one tie, and 8/8 pooled wins. Comparing the **current inductive pipeline directly to original Ani's target-input policy** gives +.010085 train / +.009929 pooled, with 8/8 and 7/8 wins; validation improves +.017094 and chronology +.005102. That last contrast combines model and access differences, so the matched-access contrasts remain the model-effect evidence.

The source-only controls also improve CV over original Ani, but the inductive LDA source alone loses on original validation (.871795). The ensemble partners raise it to .905983. Adding target-input fitting to the current ensemble yields only +.000796 train / +.000532 pooled with 4/8 and 3/8 wins, reduces validation by .002137 and ties chronology; it does not establish an upgrade over the current inductive candidate.

The target-input arm models the **final submission's access pattern**, with the current evaluation batch appended as masked rows and every target label erased. It does not reproduce Ani's historical global-transform, ungrouped notebook CV. Actual Kaggle test inputs were not loaded; local CV target batches are smaller than the 312-row Kaggle batch. Chronological target inputs are visible only in that separately labeled arm. All original masked duplicates of held groups remain purged before adding the target batch. No Part 2 data, external records, final235 refit, test predictions, upload or promotion occurred.

Independent verification checked **96 complete-OOF repeat records, 480 fold arrays, 24 supplementary records, 80 fitting boundaries and 336 notebook-function parity comparisons**. Source/data hashes, gold labels, query-label erasure, row coverage and fresh-control reproduction passed. The reported .89423 remains attached to original Ani's historical artifact. The local improvement is established under these shared development tests; the later LDA submission scored .89401, tying the earlier ensemble rather than demonstrating a Kaggle gain. The separate anchored-model result is recorded at the top of this handoff.

### Completed follow-ups against the stronger LDA baseline

The [LDA follow-up review](research/lda_ensemble/experiments/lda_followup_review_v1/report.html) adds seven studies and **69 parent-specific recipes, including matched controls and related variants**, each evaluated on all eight saved repeats in both pools. **None confirmed a further improvement over its matching latest LDA parent.** The .931794/.921631 LDA ensemble remains the confirmed CV improvement from this research track. This statement does not rank the separately developed anchored classifier, whose later reported Kaggle result is higher.

The studies cover [binary and conditional Other heads](research/lda_ensemble/experiments/conditional_other_cv_v1/report.html), [neighbor geometry](research/lda_ensemble/experiments/lda_neighbor_geometry_cv_v1/report.html), [grouped subset averaging](research/lda_ensemble/experiments/lda_subset_cv_v1/report.html), [reference-distance features](research/lda_ensemble/experiments/recruitment_novelty_cv_v1/report.html), [ensemble mixes](research/lda_ensemble/experiments/lda_ensemble_mix_cv_v1/report.html), [joint-label-set LDA](research/lda_ensemble/experiments/joint_lda_reason_cv_v1/report.html), and [protein-target/fingerprint retries](research/lda_ensemble/experiments/lda_feature_retry_cv_v1/report.html). These are focused recipe comparisons, not 69 independent hypotheses. Their comparator is the LDA revision, not the subsequently submitted anchored classifier.

| Follow-up ensemble | Train157 CV | Pooled235 CV | Validation78 | Chronological98 | Interpretation |
|---|---:|---:|---:|---:|---|
| Current LDA 70/15/15 reference | .931794 | .921631 | .905983 | .877551 | Current confirmed CV candidate |
| LDA 75/25 tri mix | .933121 | .919947 | .876068 | .884354 | Higher train CV, worse pooled CV and validation; only 2/8 and 3/8 wins |
| Half grouped-subset average | .931529 | .921454 | .910256 | .877551 | Close to reference; only 2/8 and 3/8 wins |
| Reference-distance LR specialist | .931263 | .921277 | .905983 | .877551 | No paired wins in either pool |
| Augmented protein-target head | .931263 | .922518 | Not evaluated | Not evaluated | Pooled gain with train loss; only 2/8 and 4/8 wins |
| Augmented fingerprint head | .931794 | .921631 | Not evaluated | Not evaluated | Same means, with offsetting trial changes; no confirmation |

The feature retry preserves the new LDA and neighbors while changing only its final C.1 logistic head, using the new pseudo-labels. It compares augmented fitting with half replacement by a gold-only head, including no-feature controls for each. Fingerprint selection stays inside fitting gold. Neither protein targets nor fingerprints established a metadata-specific gain under the paired two-pool rule. This explicitly retests promising earlier fields on the stronger model rather than treating earlier negatives as universal exclusions.

The [label-set error diagnostic](research/lda_ensemble/experiments/lda_followup_review_v1/baseline_labelset_errors.csv) finds a sharper remaining gap: Other is predicted in **71.15%** of repeated OOF predictions for the 26 Other-only cases, but only **4.17%** for the nine Recruitment+Other cases. Other-only trials contribute .035284 of the overall .078369 F1 deficit, versus .012589 from Recruitment+Other and .009929 from seven Funding-only trials. These are descriptive error contributions on reused labels, not attainable improvement estimates. Pure Recruitment already averages .993613 sample F1; further broad Recruitment adjustments have little room and can harm many correct cases. The next feature hypothesis should address confusion among the less common reasons as well as missing second reasons, using these cohort counts to prioritize work.

The reference-distance specialist excluded each training row's entire embedding group from its gold reference sets. The joint LDA head treated observed combinations as distinct outcomes and used exact conditional sample-F1 decoding; neither helped reliably. The mix study reused hash-verified predictions on identical folds, recovered exact partner scores algebraically and reconstructed both original and revised controls. Independent rescoring across the seven studies checked **2,144 complete-OOF repeat records, 10,720 fold arrays and 116 supplementary records**; all protected hashes, fitting boundaries and saved-fold checks passed. No Part 2 fields, external records, test inputs, final235 refit, submission, or promotion were used.

## Next hypotheses, with existing work accounted for

### Completed Part-1-only feature round

The [feature-gap study](research/lda_ensemble/experiments/feature_gap_cv_v1/report.html) tested nine feature blocks separately in Ani's final multinomial classifier and in the unchanged 70/15/15 ensemble: four new representations (inclusion text, exclusion text, eligibility structure, full-plus-category ICD codes) and five retries (whole criteria, disease indicators, ICD categories, protein targets, matched drug names). Only supplied **Part 1** inputs were used; Part 2 phase/outcome data and external records are excluded. Pseudo-label generation, kNN, regularization, decoder, and the predeclared feature-block weight stayed fixed.

Three candidates passed the initial three-repeat screen in both pools, but **none met eight-repeat confirmation**:

| Addition and exact parent | Train157 delta / wins | Pooled235 delta / wins | Validation | Chronology |
|---|---:|---:|---:|---:|
| Protein targets → Ani | +.001062 / 4 of 8 | +.002571 / 6 of 8 | .884615 | .850340 |
| Diseases → three-way | +.000265 / 4 of 8 | −.001241 / 3 of 8 | .905983 | .870748 |
| Full + category ICD → three-way | −.000531 / 3 of 8 | −.002660 / 3 of 8 | Not evaluated | Not evaluated |

The protein-target and disease supplementary checks were predeclared **diagnostics**, not confirmed finalists. Native parent validation/chronology are .891026/.860544 for Ani and .910256/.877551 for three-way. Protein targets raised Ani's pooled Other F1 from .612507 to .626935, but recall only from .553571 to .560714. Corrections to the persistent Other errors were sporadic; the ensemble disease/ICD additions lowered Other recall. Do not promote or export these recipes.

All 80 fitting folds reproduced the saved native Ani control, and feature parsing, query-batch invariance, vocabulary/scaling boundaries, grouped purges and protected hashes passed. An [independent rescore](research/lda_ensemble/experiments/feature_gap_cv_v1/independent_verification.json) checked 284 complete-OOF repeat records, 1,420 fold prediction arrays and 16 supplementary records. [Protocol](research/lda_ensemble/experiments/feature_gap_cv_v1/protocol.json), [all screen results (summary)](research/lda_ensemble/experiments/feature_gap_cv_v1/summary.json), [confirmation](research/lda_ensemble/experiments/feature_gap_cv_v1/final_comparison.csv), [per-trial changes (summary)](research/lda_ensemble/experiments/feature_gap_cv_v1/summary.json), and negative results are retained. No submission, test prediction, full235 final refit, or model promotion was performed.

These results narrowed the next question to how metadata contributes independent corrections. In the initial pooled folds, the native final classifier received 633–748 embedding-generated pseudo-label rows versus 183–193 gold rows. The follow-up studies below now test gold-only corrections against matched gold-only embedding controls. The older raw features remain candidates for different representations or integration methods, not universally rejected fields.

### Completed feature and correction follow-ups

The [consolidated follow-up report](research/lda_ensemble/experiments/feature_followup_review_v1/report.html) records nine further studies and **152 parent-specific screened recipes, including matched controls and related variants**. These are not 152 independent hypotheses. They cover gold-only corrections, Other-only transfers, independent multi-reason neighbor votes, cross-fitted residual and sample-F1-weighted correctors, multi-reason pseudo-label teachers, finer text encodings, biochemical features, and metadata-assisted neighbor retrieval. **None established an upgrade to the three-way ensemble under the two-pool confirmation rule.**

| Recipe | Train157 CV | Pooled235 CV | Validation78 | Chronological98 | Interpretation |
|---|---:|---:|---:|---:|---|
| Half gold-only protein-target correction → Ani | .922240 | .914007 | .878205 | .853741 | Passed vs Ani, 5/8 wins in each pool; below ensemble and worse supplementary checks |
| Independent binary neighbor votes → Ani | .921444 | .911791 | .891026 | .863946 | Passed vs Ani; below ensemble |
| Transfer gold-target Other scores → ensemble | .926221 | .917908 | .897436 | .877551 | Only 4/8 wins in each pool; not confirmed |
| Half gold-only Morgan fingerprint correction → Ani | .921975 | .913032 | .880342 | .860544 | 6/8 train wins but 4/8 pooled wins; not confirmed |

Matched controls matter: protein targets added only +.001062 train / +.001064 pooled beyond the matching gold-only embedding head, with 4/8 and 5/8 wins. Morgan fingerprints added +.000796 / +.000089, with 4/8 wins in each pool. Neither establishes a metadata-specific gain. The gold-only training change explains much of the improvement over native Ani.

The [text study](research/lda_ensemble/experiments/selected_text_cv_v2/report.html) retested criteria word/character fragments, raw drug-name fragments, and disease phrases. Vocabulary and chi-square feature selection were fitted inside each outer training partition, with selection using gold fitting labels only. Disease phrases passed the initial screen but fell to .925425/.915426 after confirmation. The [biochemical study](research/lda_ensemble/experiments/biochemical_features_cv_v1/report.html) tested actions, all versus specific catalog pathways, and Morgan fingerprints from supplied SMILES. The earlier chemistry screen had used molecular descriptors, not fingerprints. No external chemical or trial records were fetched.

The [retrieval study](research/lda_ensemble/experiments/metadata_neighbors_cv_v1/report.html) tested criteria terms, raw drug fragments, molecular fingerprints and protein targets in the nearest-neighbor distance while preserving the original logistic head. Some pooled gains accompanied train-pool losses; none confirmed against the ensemble or demonstrated a consistent metadata contribution against matching embedding-only retrieval controls.

The [teacher ablation](research/lda_ensemble/experiments/multilabel_teacher_cv_v1/report.html) verified that native normalized teacher scores, confidence .8, and positive cutoff .3 cannot produce multiple positive pseudo-labels. Removing that restriction did not improve the ensemble reliably. Treat this as a tested structural property, not an established defect. [Cross-fitted Other corrections](research/lda_ensemble/experiments/other_residual_cv_v1/report.html) and [sample-F1-weighted corrections](research/lda_ensemble/experiments/other_utility_cv_v3/report.html) also failed confirmation.

Every completed study preserves its recipe, paired repeat results, fold predictions, supplementary checks and independent verification. Failed versions remain separately recorded: a zero-utility assertion, a numerical optimizer failure, and raw-drug bracket parsing. Corrected versions reran the required comparisons; no failed fold was discarded. The consolidated report includes verification totals and source hashes. Only supplied Part 1 inputs were used; no Part 2 fields, external records, test fitting/prediction, final235 refit, submission, or promotion occurred.

The numbered hypotheses below originated at integration; the completed study above supersedes their untested status for its specific encodings and classifier integration.

1. **Structured eligibility text:** separate inclusion from exclusion, model eligibility burden, and validate numeric parsing before adding clinical bounds. First measure metadata-only signal under the shared folds. Plain whole-text features have already been tried.
2. **Separate embedding and metadata models:** retain a rotation-tolerant embedding model and test a small, regularized metadata model. Only if it adds complementary signal should we try a gate using annotation coverage/uncertainty, trained within inner CV. Raw/PCA tree models, co-training, and fixed blends are already represented in past trials.
3. **Target persistent multilabel errors:** test a new representation only when error patterns support it. Classifier chains, label powerset, and joint score decoders have already been explored; those names alone are not new experiments.

The independent review suggested correctly parsing ICD codes because its generic text tokenizer split code suffixes. **The logistic branch already unpacked nested ICD lists into three-character categories** and tested minimum-count 3/5 features with fixed DBSCAN targets. Its [feature screen](research/lda_ensemble/experiments/remaining_classifier_features/SUMMARY.md) found no improvement over .910256 validation F1 (best ICD variant .895299). Reuse that parser and evidence; full codes or alternate hierarchies would be a new, explicit hypothesis, not a claim that ICD parsing has never been tested. That older screen was validation-driven, so a useful new representation still needs the shared CV protocol.

No architecture/feature experiment from this next-step list was run during integration itself. The later feature study above tested specific representations from this list. No extra data or deep learning is planned. The provided reason vectors are rotated embeddings; raw stopping-reason text is unavailable. Drug/action/target sets do not provide reliable pairwise links, and pathway depths alone do not provide missing parent edges.

## Working from the combined tree

- [Research README and reproduction instructions](research/lda_ensemble/research/independent/README.md)
- [Current retention manifest and source hashes](research/consolidation_v1/retention.json)
- [Integration verification](research/lda_ensemble/research/independent/integration_verification.json)
- [Current agent conventions](research/lda_ensemble/AGENTS.md)

Run scripts from `part1/research/independent` in the recorded environment. Keep future runs in new versioned directories; do not overwrite hashed historical sources or protocols. The integration preserves research and reproducibility, but final course packaging still needs a notebook/report update for the eventual selected model. Existing numbered notebooks describe earlier checkpoints.
