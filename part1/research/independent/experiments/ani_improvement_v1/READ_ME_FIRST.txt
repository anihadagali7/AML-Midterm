ANI IMPROVEMENT ROUND 1

Start with report.html. It contains the complete initial screen and the
confirmed finalists, including all unsuccessful variants.

The initial cohort had fourteen variants plus the native Ani control:
- Three Ani/tri-training mixtures, two Ani/SVM mixtures, and one three-way mix.
- Two reduced pseudo-label weights, two agreement-filter variants.
- One average over three predetermined GMM seeds.
- One nested Other-threshold adjustment and two Other specialists.

All candidates used the original fixed 5-fold x 3-repeat train157 and pooled235
splits. Four ensembles passed the predeclared shortlist rule; they and the
control received the same five additional repeats previously frozen for the
cross-repo comparison. No new split seeds or favorable subsets were chosen.
The three strongest confirmed candidates and control were then evaluated on
the original validation and the three unchanged chronological windows.
No weights, thresholds, features or configurations changed after validation.

Retained candidates:
- cv_leader.json: 75% Ani plus 25% tri-training, with decision scales aligned.
  Train CV8 0.9267515924; pooled CV8 0.9164007092.
  Beats Ani in 8/8 training repeats and 7/8 pooled repeats.
  Validation 0.8888888889; chronological 0.8843537415.
- all_four_checks_candidate.json: 70% Ani, 15% tri, 15% local SVM hybrid.
  Train CV8 0.9256900212; pooled CV8 0.9164893617.
  Beats Ani in 7/8 repeats in each CV pool.
  Validation 0.9102564103; chronological 0.8775510204.
- Native training-only Ani: train CV8 0.9195859873; pooled CV8 0.9098404255;
  validation 0.8910256410; chronological 0.8605442177.

The mixture weights apply to decision scores. Ani scores are shifted in
log-odds so its native .3 cutoff maps to .5. Tri keeps its native .5 scores;
the local SVM hybrid contributes hard label votes. Final cutoff is .5 with
highest-score fallback for an otherwise empty prediction. These scores are
not asserted to be calibrated probabilities.

Pseudo-label weighting acts on the logistic loss, neighbor votes and GMM
label mapping. The logistic total loss scale is normalized to its native row
count. Agreement filters gave no aggregate sample-F1 gain in the initial
screen; a few individual label predictions changed. Other thresholds were
selected only in grouped inner CV, excluding inner held-out masked duplicates.
Averaging seeds used all three predetermined seeds, never the best-scoring one.

Integrity:
- Native control predictions exactly match the prior benchmark in all eight
  repeats in both pools.
- All saved CV, validation and chronological predictions were independently
  rescored; IDs and truth checked against the supplied CSVs.
- Seven representative fitting paths pass held-out label/feature mutation,
  masked duplicate exclusion (72 rows), and query-batch invariance checks.
- The boundary audit's first metadata write hit a Path/string checksum error
  after all fitting checks passed. Metadata serialization was fixed; the
  checks and model implementation were unchanged. See boundary_audit.json.
- The provided data, prior benchmarks, submitted tri-training CSV and its
  reported Kaggle score, and saved current-best submission are unchanged.

These are development comparisons on previously used labels, not independent
future tests. Multiple tested candidates can introduce selection optimism.
Both retained models remain unsubmitted. No final235 submission refit or test
input fitting was performed. Three Kaggle submissions remain from the user's
reported budget; this experiment used none. The Kaggle reference is Ani's
original transductive submission, whereas all local fits here are inductive.

Reproduce from the AMLMidterm root with the existing Python environment:
  .venv/bin/python run_ani_improvements.py prepare
  .venv/bin/python run_ani_improvements.py parity
  .venv/bin/python run_ani_improvements.py train
  .venv/bin/python run_ani_improvements.py pooled
  .venv/bin/python run_ani_improvements.py shortlist
  .venv/bin/python run_ani_improvements.py extra_train
  .venv/bin/python run_ani_improvements.py extra_pooled
  .venv/bin/python run_ani_improvements.py confirm
  .venv/bin/python run_ani_improvements.py supplementary
  .venv/bin/python run_ani_improvements.py report
  .venv/bin/python run_ani_improvements.py verify
  .venv/bin/python audit_ani_improvements.py

The source hashes and registry are frozen in protocol.json. Cached results
are keyed on the recipe, code and fold protocol. Recipe/source changes must
be recorded as a new experiment version rather than overwriting this cohort.
