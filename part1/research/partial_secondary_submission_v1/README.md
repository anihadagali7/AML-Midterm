# Exploratory partial-Other submission

Prepared one candidate after Michael asked whether the unknown-secondary model was worth a manual test with five submissions remaining. **It has not been uploaded or scored. It is not a promoted leader.** The current best reported score remains .89935.

Use [partial_other_unknown80_cand2_inductive20_trainval235.csv](../../submissions/candidates/partial_other_unknown80_cand2_inductive20_trainval235.csv). SHA-256: `f6aed0af121d02364c346b38b7c4a6f83f2409b7e9e03fb49b74e5e0d25b353e`.

The exact recipe is the prior benchmark's `binary_unknown`: 80% altered anchored scores and20% unchanged strict-inductive five-seed Cand2 marginals, with .3 cutoff plus argmax. It keeps unselected Other pseudo-labels unknown in the binary Other classifiers, Other neighbor retrieval and Other cluster mapping. Its altered anchored component was refitted on235 supplied labels plus1,014 originally masked training rows. Test inputs entered prediction only. No settings were tuned during preparation.

This is an explicit exploratory follow-up to a failed advancement result. The prior benchmark showed Train157 .920382 versus .935244 for the current blend, pooled .914982 versus .921631, validation tied at .891026, and chronology .901020 versus .887755. The main variant lost every grouped repeat in both pools. Its later-year and Other-recall gains motivate one informative trial; they do not establish an expected Kaggle improvement. The original frozen advancement protocol remains unchanged.

The file changes47 of312 label sets relative to the .89935 submission. It adds Other on37 rows and removes Other on2; other labels also change through pseudo-label feedback. These differences were inspected only after the recipe was frozen and do not reveal correctness. No alternate final weight, threshold or per-row reversal was selected.

An unchanged anchored refit reproduced its saved final scores exactly, and reconstructing the parent80/20 CSV reproduced the .89935 artifact's exact SHA-256. Verification checked the candidate's312 unique IDs, sample-submission order, allowed nonempty label sets, arithmetic, registered benchmark values and protected source hashes. [Preparation record](prepared_submission.json), [frozen export](export_protocol.json), [component provenance](component_provenance.json), [verification receipt](verification.json).

Run `python part1/research/partial_secondary_submission_v1/verify_candidate.py` from the repository root to verify without fitting. The export refuses to overwrite an existing protocol or CSV; use a disposable checkout for replay. A future user-reported Kaggle result requires its own result record, commit and immutable annotated score tag. No score tag is created for this unsubmitted artifact.
