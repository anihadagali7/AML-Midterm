# Final-slot checks while file 15 is awaiting Kaggle

Declared after file 13 scored 0.58465, while file 15's score is unknown. File 15
is 60% file 10, 20% original ensemble, 10% recent tree and 10% phase trend. This
batch makes nine small, fixed variations before consulting any new Kaggle score.

Five candidates follow the line from file 13 to file 15 with alpha 0.25, 0.5,
0.75, 1.25 or 1.5. Two change file 15's trend weight to 5% or 15%; two change its
tree weight to 7.5% or 12.5%. Remaining mass stays on file 10. All component
weights stay nonnegative and sum to one, including the line extensions.

Use the same 13 splits, two preprocessing tracks, equal-phase log loss and
pooled grouped CV. Retain every result and compare against both files 13 and 15.
Use paired bootstrap intervals for any selected candidate. Historical labels are
reused; this is adaptive selection, with no claim of an untouched validation set.

Wait for the actual file 15 score before choosing the final upload. If it improves
the score, require both chronological averages below file 15 and corroborating
direction in the other track where possible. Avoid a CV regression above 0.001
or worst-gap regression above 0.001. If no candidate supports a further extension,
prefer a locally supported interpolation or retain the best known file. If file
15 loses, examine interpolation against file 13 and the earlier independent
reserve. Do not force a submission from a single favorable split.

No base-model refitting, test labels, external labeled data, or outcome inference
from individual test rows. Only aggregate returned Kaggle scores inform direction.
All exports must replay the native 9,050-row fits, reconstruct files 13 and 15,
and retain exact dependencies and CSV hashes. At most one upload remains after
file 15; the active user deadline remains 2026-11-07 00:06:49 UTC.
