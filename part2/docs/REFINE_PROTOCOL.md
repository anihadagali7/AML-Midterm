# Three remaining uploads after file 11 reached first place

Declared 2026-11-06, after the user reported file 11 at 0.58474 and before this
batch was evaluated. Start 23:36:49 UTC; the user's revised 30-minute window ends
2026-11-07 00:06:49 UTC. At most three further uploads, with manual Kaggle results
provided by the user. File 11 is the scored control. Preserve all earlier CSVs.

File 13 was already evaluated and exported before file 11's Kaggle score. Its
30% original-ensemble weight improves both chronological averages and grouped CV
against file 11 in both preprocessing tracks, but worsens two of six time windows.
Recommend it first now that the 20% direction improved Kaggle. Do not assume that
larger weights or other components improve Kaggle until their scores are returned.

Freeze 19 new fixed probability blends: original-ensemble weights 25%, 35%, 40%,
and 50%; original-ensemble weights 20%, 30%, and 40% combined with recent-tree
weights 5%, 10%, and 15%; four mixtures combining original-ensemble weights 20%
or 30%, recent-tree weights 5% or 10%, and 10% Ani phase trend; and two mixtures
with 30% original ensemble plus 10% or 20% sharper neighbors. Remaining weight
always goes to file 10. No per-phase optimization or new base-model fitting.

Every component prediction must match the exact evaluation IDs/labels and saved
source run on all original ten splits and all three longer-gap splits. Reuse both
existing feature policies; original-ensemble and recent-tree components retain
training-only preprocessing. Equal-phase log loss and pooled repeated grouped CV
remain unchanged. Compare every candidate directly against file 11, retain every
result, and inspect the latest forward window and worst gap as well as means.

Prefer both chronological averages below file 11, with at least 0.0002 gain in
one, CV regression no greater than 0.001, corroborating direction in both tracks,
and worst-gap regression no greater than 0.002. Report paired trial bootstrap
intervals for the shortlist. These labels have been repeatedly used; intervals
do not correct for adaptive selection and are not independent validation.

Use returned aggregate Kaggle scores to choose which locally supported direction
to submit next, without inferring individual test outcomes. No external labeled
data, test labels, stopping reasons, IDs as predictors, or deep learning. Export
from the existing full 9,050-row native models, verify source/dependency hashes,
exact winning-file reconstruction, independent inference with labels/IDs changed,
sample order and finite probabilities. Frozen receipts retain unknown scores;
later scores belong in the live ledger and queue.
