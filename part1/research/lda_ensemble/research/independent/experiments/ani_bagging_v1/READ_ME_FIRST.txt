Purpose
Follow up the eight single-change combinations, which did not yield a supported
improvement after the full eight-repeat check. Test whether averaging Ani fits
on different grouped gold subsets improves stability with a small labeled set.

Four frozen recipes
For each existing parent ensemble (75% Ani + 25% tri-training; or 70% Ani +
15% tri-training + 15% SVM), replace the Ani component with one of:
1. 50% original full-fitting-set Ani + 50% subset-model average.
2. 100% subset-model average.
The subset average uses five equally weighted Ani fits, each trained on roughly
80% of the outer-training gold labels. Subsets come from grouped multilabel
five-fold assignments at fixed seed270118. Members use original native Ani
settings. Average raw Ani scores before the original decision-score transform.
The tri-training/SVM partner fits and their ensemble weights remain unchanged.

Boundaries
Each subset fit excludes the omitted gold groups and outer held-out groups from
both labeled and original masked training pools. Omitted gold is not repurposed
as unlabeled data. Validation/test inputs are absent from fitting. The audit
checks all subset fitting pools, held-out-label/feature mutation, matching
masked duplicates, and query-batch independence.

Evaluation
Reuse exactly the same original grouped train157/pooled235 three-repeat screen,
then the same five extra repeats for shortlisted recipes and controls. Every
repeat has five outer folds. Advancement follows the unchanged matched-parent
rule from the preceding experiment. Freeze the supplementary shortlist before
checking validation or chronology. Reuse prior validation/chronology results for
unchanged controls. Registry, protocol, sources and protected files are hashed.

Implementation
run_ani_bagging.py reuses functions from run_ani_combinations.py while selecting
this study's own output directory, model dispatch, source hashes and registry.
It does not edit the frozen preceding scripts or their results. The subset
models are implemented separately in ani_bagging.py. No Kaggle upload or final
train+validation submission fit is part of this study.

Reproduction
From /Users/matthew/Desktop/AMLMidterm, use .venv/bin/python with:
run_ani_bagging.py prepare
run_ani_bagging.py parity
audit_ani_bagging.py
run_ani_bagging.py train
run_ani_bagging.py pooled
run_ani_bagging.py shortlist
run_ani_bagging.py extra_train
run_ani_bagging.py extra_pooled
run_ani_bagging.py confirm
run_ani_bagging.py supplementary
run_ani_bagging.py report
run_ani_bagging.py verify

Interpretation
All CV scores are sample-wise F1 averaged over complete out-of-fold vectors.
Repeat SD describes variation across reused splits; it is not an independent
test confidence interval. These are repeatedly reused development labels, and
local gains cannot establish a new Kaggle leaderboard score. Keep prior failed
and successful trials for a complete record.
