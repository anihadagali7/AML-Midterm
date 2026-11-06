# Four-submission iteration after file 10 scored 0.58609

Start: 2026-11-06 22:49:35 UTC. User has up to four further submissions over
roughly 90 minutes. Use file 10 as the scored baseline. Keep the original Ani
references, full older ensembles, and frozen predictions intact.

Round one is frozen in `sprint_models.py` before scoring: nearby fast-neighbor
blend weights, mixtures with existing complementary estimators, and ten small
meta-model adjustments. All base-model OOF features are reused only with matching
source hashes and fitting/evaluation IDs. Fit any new meta-model on fitting-fold
OOF data only. No test labels, external labeled data, IDs, stopping reasons,
deep learning, or reverse engineering of outcomes.

Evaluate every recipe on the original ten splits and all three longer-gap
splits, under strict and original published-recipe policies. Equal-phase log loss
and within-repeat pooled grouped CV remain unchanged. These labels have been
consulted previously; results and bootstrap intervals are adaptive research
evidence, not untouched validation or guaranteed leaderboard improvements.

Prefer candidates that reduce both chronological averages against file 10, with
at least 0.0002 gain in one average, no more than 0.001 grouped-CV regression,
and corroborating direction under the published policy. Inspect every period,
worst-window loss, official validation and paired uncertainty. Keep every failed
proposal. Select a small, diverse shortlist rather than spending all four slots
on nearly identical probabilities. Do not force a submission if evidence is poor.

Prepare the best supported first candidate promptly. The user uploads via SCP
and returns the score. Preserve that exact file/hash; record later observations
separately. Subsequent choices may adapt to the aggregate score but may not infer
individual test outcomes. At most four additional submissions are recommended
in this window; leave room to react to the first result. Never count an unknown
score as an improvement.

Final ensembles use full train+validation fits (9,050 rows). If a legacy tree
component is selected, refit its original recipe on those same 9,050 rows before
exporting. Verify final predictions against native-model replay, sample ID order,
finite probabilities and file hashes. Keep the 0.58609 winner intact.
