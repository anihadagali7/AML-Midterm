# Part 1 working conventions

Read `TESTING_POLICY.md` and `RESEARCH_HANDOFF.md` before starting new model work.
They are the current cross-branch policy and evidence summary. Older experiment
documents, notebook headings, and `current_best` filenames describe their own
historical checkpoints, not the current global winner.

Use the saved grouped train157 and pooled235 folds, paired repeat comparisons,
shortlist-before-validation procedure, and supplementary chronological checks.
Preserve the default inductive fitting boundary. Keep explicitly requested
transductive diagnostics separately labeled. Never fit on hidden labels.

Do not modify frozen research code, recipes, protocols, raw gold data, or scored
submission CSVs in place. Their hashes are checked by later studies. Add versioned
studies/adapters and run matched controls. Retain failures and provenance.

No deep learning, external trial data/labels, ID predictors, or test-label reverse
engineering. Do not automatically refit, upload, promote, or retag a model because
its CV score is higher. Existing user authorization for a particular action still
applies. Record reported Kaggle scores against the exact file hash and access mode.
