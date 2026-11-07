Purpose
Continue the search before using another Kaggle submission. This study tests
eight changes: four additions separately applied to each of the two previously
leading ensembles. The native Ani adaptation and both unchanged parents are
controls. All three controls reproduced their previous decision scores before
the CV runs began.

Recipes
Parent A: 75% Ani decision score + 25% classical tri-training score.
Parent B: 70% Ani + 15% tri-training + 15% local SVM hybrid vote.
Additions: halve Ani pseudo-label trust; average three GMM seeds; learn the
Other decision threshold in nested CV; or mix a 50% binary Other specialist
and learn its threshold in nested CV. Parent ensemble weights stay fixed.

Evaluation
Use the original saved grouped five-fold assignments in both train157 and
pooled235. Screen every candidate on the same three repeats, then extend
shortlisted candidates and controls with the same five additional repeats.
protocol.json freezes the paired comparison and advancement rules before fits.
registry.json records exact recipes and source hashes. Model changes are
compared with their own parent, as well as the fixed previous leaders in each
CV pool. No tuning uses outer-fold labels. Original validation and chronological
checks are reserved for the frozen supplementary shortlist. Unchanged control
validation/chronology results are reused with provenance.

Auditing
audit_ani_combinations.py checks held-out label/embedding mutation, matching
masked-duplicate exclusion, query-batch independence, and the Ani fitting pools
inside nested CV for each of the four new code paths. Final verification
rescoring checks saved predictions, IDs and labels, and reproduces previous
control CV outputs exactly. Protected inputs and prior submission files are
hashed and checked before/after stages.

Interpretation
CV means are sample-wise F1 over a complete out-of-fold prediction vector for
each repeat, averaged across repeats. Repeat standard deviations are descriptive;
repeats reuse the same people/trials and are not independent datasets. These
development labels have been used in earlier searches. Positive CV changes
do not prove an improvement on Kaggle's later trials. Test inputs never enter
fitting, and this study creates no submission or final train+validation model.

Outputs
train3/, pooled3/: all initial predictions and per-fold/repeat metrics.
train5/, pooled5/: additional repeats for the frozen shortlist and controls.
screen_comparison.csv: all eight attempted changes, including failures.
confirmation.csv/json: eight-repeat results and supplementary shortlist.
supplementary/results.json: original validation and chronological checks.
final_comparison.csv, summary.json, report.html: final comparison when complete.
boundary_audit.json, verification.json: audit results when complete.

Reproduction (from /Users/matthew/Desktop/AMLMidterm)
.venv/bin/python run_ani_combinations.py prepare
.venv/bin/python run_ani_combinations.py parity
.venv/bin/python audit_ani_combinations.py
.venv/bin/python run_ani_combinations.py train
.venv/bin/python run_ani_combinations.py pooled
.venv/bin/python run_ani_combinations.py shortlist
.venv/bin/python run_ani_combinations.py extra_train
.venv/bin/python run_ani_combinations.py extra_pooled
.venv/bin/python run_ani_combinations.py confirm
.venv/bin/python run_ani_combinations.py supplementary
.venv/bin/python run_ani_combinations.py report
.venv/bin/python run_ani_combinations.py verify

These commands reuse cached completed CV results after source/data checks. Do
not edit a frozen recipe or protocol in place; register a separate study instead.
