"""Create the self-contained, executed Part Two notebook."""
import json
import os
import sys
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parent
MODULES=['experiments.py','boosted.py','refinements.py','recency_and_sparse_boost.py',
         'boosted_robustness.py','year_cap.py','ensemble_search.py','submission_models.py',
         'error_analysis.py','reporting.py','followup_ensembles.py']


def run():
    selected=json.loads((ROOT/'experiments/model_search/selected_ensembles.json').read_text())
    cells=[]
    def md(t):cells.append(nbf.v4.new_markdown_cell(t))
    def code(t,hidden=False):
        cell=nbf.v4.new_code_cell(t)
        if hidden:cell.metadata['jupyter']={'source_hidden':True}
        cells.append(cell)
    md('''# Part Two: clinical-trial outcomes

Predict the probability of success for Phases I, II and III. Kaggle evaluates log loss (lower is better).
The best reported reference score is **0.58688**. Our **03 scored 0.59562**, and **02 improved to 0.58993**. The local ranking did not transfer to Kaggle. Initial 01/04 remain unscored. Results are recorded against file hashes in `submissions/leaderboard_results.json`.

This notebook includes the complete experiment source, two local evaluation protocols, per-phase F1/ROC-AUC/average precision, the required stopping-reason analysis, and four checked submission files. It uses the supplied data and classical machine learning. IDs and Part One reason information never enter outcome prediction.

**Latest update: hold 05/06.** Ani's published model is reproduced, and the new ten-split benchmark favors Ani over our linear endpoint across time. See `local_benchmark.ipynb`, `LOCAL_TESTING.md` and `experiments/local_benchmark/RESULTS.md` before choosing another submission. This notebook preserves the initial search and original outcome-model fits.
''')
    code('''from pathlib import Path
import os, sys, json
ROOT = Path.cwd()
if (ROOT / "AML-Midterm/part2").is_dir(): ROOT = ROOT / "AML-Midterm/part2"
assert (ROOT / "outcome_train.csv").exists(), "Run in the repository or part2 directory"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ["MPLCONFIGDIR"] = str(ROOT / "experiments/.matplotlib")
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
''')
    md('''## Complete implementation

These collapsed cells contain all experiment and inference code as executable notebook source. Running them materializes the modules in this directory. Search execution is controlled separately below.
''')
    for name in MODULES:code('%%writefile '+name+'\n'+(ROOT/name).read_text(),True)
    md('''## Reproduction and evaluation design

The default reuses completed experiments and does not retrain. Set `RUN_TRAINING=True` to repeat the searches and produce the four submissions.

The supplied validation set has 149 Phase I, 501 Phase II and 363 Phase III rows. Selection uses the **equal mean of per-phase log losses**, since test has 500 rows per phase. A second, train-only temporal comparison fits through 2008 and evaluates on 2009-2013 trials. Combined-phase rows stay together. Every learned feature transform is fitted only on the corresponding fitting population. Final models use all 8,037 training rows and no validation labels.

Both local evaluations are used in model selection. They are not untouched estimates of generalization; the latest test cohort (2014-2021) can behave differently.
''')
    code('''RUN_TRAINING = False
import subprocess
from experiments import OUT, load_data, metrics, audit_data
SELECTED = '''+repr(selected)+'''
if RUN_TRAINING:
    for script in ["experiments.py", "refinements.py", "recency_and_sparse_boost.py", "year_cap.py",
                   "boosted.py", "boosted_robustness.py", "ensemble_search.py"]:
        command = [sys.executable, script]
        if script == "ensemble_search.py": command.append("--select")
        subprocess.run(command, check=True)
    (OUT / "selected_ensembles.json").write_text(json.dumps(SELECTED, indent=2))
    for script in ["submission_models.py", "error_analysis.py", "followup_ensembles.py", "reporting.py"]:
        subprocess.run([sys.executable, script], check=True)
import numpy as np
import pandas as pd
from IPython.display import display, Image
data = load_data()
display(pd.DataFrame([{ "split":s, "rows":len(d), "trials":d.trial_id.nunique(),
    "first_year":d.start_year.min(), "last_year":d.start_year.max(),
    "matched_drug_fraction":d.matched_drugs.notna().mean()} for s,d in data.items()]))
audit = audit_data(data)
assert all(n == 0 for n in audit["overlaps"].values())
''')
    md('''## Model and feature comparisons

Compare eligibility word/bigram and character TF-IDF, binary words, diseases, drugs, ICD groups, protein targets/actions/pathways, Morgan fingerprints/descriptors, shared and separate phase effects, newer-trial weighting, raw/capped start year, and NB log-ratio features. Algorithms include logistic regression, CatBoost, histogram gradient boosting, sparse LightGBM and Extra Trees. All settings and predictions are saved under `experiments/model_search`.

The baseline uses eligibility TF-IDF (35,000 unigram/bigram terms, minimum document frequency 3, sublinear TF), phase indicators, and pooled L2 logistic regression with C=2. Molecular features do not consistently help. Raw year extrapolation is fragile in the temporal check. Annotation coverage declines in test, which favors retaining a strong text component. The selected mixes also test whether phase-specific model balance improves over one global blend.
''')
    code('''components = pd.read_csv(OUT / "all_component_scores.csv")
ensemble_scores = pd.read_csv(OUT / "ensemble_scores.csv")
print(f"{len(components)} matching component configurations; {len(ensemble_scores)} fixed ensemble configurations")
display(components[["name", "validation_phase_mean_logloss", "temporal_phase_mean_logloss", "joint_logloss"]].head(15))
display(ensemble_scores[["name", "validation_phase_mean_logloss", "temporal_phase_mean_logloss", "joint_logloss"]].head(12))
display(Image(filename=str(OUT / "figures/model_comparison.png")))
''')
    md('''## Four manual Kaggle submissions

03 was selected first because it had the lowest local joint loss, but scored **0.59562**. 02 improved to **0.58993**, still behind the **0.58688** reference. Less tree influence in Phases II/III improved this comparison, but the cause of the generalization gap remains uncertain. The original four CSVs are preserved. Their predictions use models fitted on the full training file, never the reduced temporal fit.
''')
    code('''index = pd.read_csv(ROOT / "submissions/index.csv")
display(index[["submission", "validation_logloss", "temporal_logloss", "mean_test_probability", "kaggle_score"]])
manifest = json.loads((ROOT / "submissions/manifest.json").read_text())
display(manifest["ensembles"])
test_predictions = pd.DataFrame({row.submission[:2]:pd.read_csv(ROOT/"submissions"/row.submission).label
                                 for row in index.itertuples()})
display(pd.DataFrame([[np.mean(np.abs(test_predictions[a]-test_predictions[b]))
                      for b in test_predictions] for a in test_predictions],
                     index=test_predictions.columns, columns=test_predictions.columns).rename_axis("Mean absolute test-probability difference"))
''')
    md('''## Prospective follow-ups after the leaderboard feedback

05 keeps Phase I exactly as in 02 and removes trees in Phases II/III. 06 keeps Phase I unchanged and uses 12.5% trees in II/III. Both reuse the original fitted components without refitting or new outcome labels. **These are now on hold after the Ani comparison.** Their selection used the reported leaderboard direction; the new fixed benchmark gives a more informative comparison. Use the separate benchmark notebook for current Ani-anchored candidates and their uncertainty.
''')
    code('''display(pd.read_csv(ROOT / "submissions/followup_index.csv"))
display(pd.read_csv(OUT / "followup_phase_metrics.csv"))
display(json.loads((ROOT / "submissions/followup_manifest.json").read_text())["ensembles"])
display(json.loads((ROOT / "submissions/leaderboard_results.json").read_text()))
''')
    md('''## Per-phase performance

Success (`label=1`) is the positive class. F1 uses threshold 0.5. ROC-AUC and PR-AUC use success probabilities; PR-AUC is average precision. Compare all four candidates with the baseline, including trade-offs when log loss improves without an F1 improvement.
''')
    code('''phase_metrics = pd.read_csv(OUT / "phase_metrics.csv")
display(phase_metrics)
display(Image(filename=str(OUT / "figures/phase_roc_pr.png")))
''')
    md('''## Error analysis using Part One reasons

For unsuccessful validation rows, join `trial_id` to Part One `ID`. Use true visible train/validation reason labels when available; otherwise use the frozen original 157-label DBSCAN/logistic model (0.5 cutoff and argmax fallback). Trials absent from Part One form a separate no-recorded-stopping-reason group. Reasons can overlap. Count each trial once within each phase.

The baseline predicts success for 8 of 9 Phase III efficacy-tagged failures, suggesting that eligibility-only text misses important failure mechanisms. All nine tags are predicted, making this a small, uncertain subgroup. Clinical context, phase effects and nonlinear models are tested to broaden the predictor, while the tables below show their actual improvements and regressions. No reason label, embedding, or Part One membership is an outcome-model input.
''')
    code('''reasons = pd.read_csv(OUT / "reason_labels_for_analysis.csv")
display(reasons.reason_source.value_counts().rename_axis("Part One label source").to_frame("trials"))
display(json.loads((OUT / "reason_model_provenance.json").read_text()))
comparison = pd.read_csv(OUT / "reason_error_comparison.csv")
display(comparison[["phase", "reason", "n", "false_success_baseline", "false_success_improved",
                    "false_success_rate_baseline", "false_success_rate_improved",
                    "true_reason_rows_baseline", "predicted_reason_rows_baseline"]])
display(Image(filename=str(OUT / "figures/reason_errors.png")))
''')
    md('''Funding has no unsuccessful Phase III examples in this validation set, so its rate is undefined. Small group sizes and uncertainty in predicted reason labels limit conclusions about differences between categories. Wilson intervals and the complete tables for every candidate are saved in CSV files. The paired trial-level bootstrap below is descriptive because the same validation set was used for selection.
''')
    code('''display(pd.read_csv(OUT / "paired_validation_bootstrap.csv"))
display(pd.read_csv(OUT / "03_phase_specialized_reason_errors.csv"))
''')
    md('''## Verification and saved artifacts

Independent linear refits and saved tree-model reloads must reproduce the stored validation and test probabilities. All CSVs have exactly 1,500 unique IDs in sample-submission order, columns `ID,label`, and finite probabilities strictly between zero and one. `index.csv` is bookkeeping and should not be submitted.

The complete fitted bundle is `experiments/model_search/submission_models.joblib`. Reload inference with `python submission_models.py --input outcome_test.csv --output replayed_predictions`. Library versions are pinned in `requirements.txt`. No Kaggle upload is performed here.
''')
    code('''checks = pd.read_csv(OUT / "model_replay_verification.csv")
assert checks.max_probability_difference.max() < 1e-8
display(checks)
sample = pd.read_csv(ROOT / "part2_sample_submission.csv")
all_files = pd.concat([index, pd.read_csv(ROOT / "submissions/followup_index.csv")], ignore_index=True)
for row in all_files.itertuples():
    result = pd.read_csv(ROOT / "submissions" / row.submission)
    assert result.columns.tolist() == ["ID", "label"]
    assert len(result) == 1500 and result.ID.nunique() == 1500
    assert result.ID.equals(sample.ID)
    assert np.isfinite(result.label).all() and result.label.between(0, 1, inclusive="neither").all()
print(f"All {len(all_files)} submissions pass the format, ID, and probability checks.")
display(json.loads((OUT / "versions.json").read_text()))
''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
        'language_info':{'name':'python','version':sys.version.split()[0]}})
    nbf.write(nb,ROOT/'part2.ipynb')
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    os.environ['IPYTHONDIR']=str(ROOT/'experiments/.ipython')
    os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'experiments/.jupyter_runtime')
    os.environ['MPLCONFIGDIR']=str(ROOT/'experiments/.matplotlib')
    for key in ['IPYTHONDIR','JUPYTER_RUNTIME_DIR','MPLCONFIGDIR']:Path(os.environ[key]).mkdir(parents=True,exist_ok=True)
    client=NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
    client.execute()
    nbf.write(nb,ROOT/'part2.ipynb')
    errors=[output for cell in nb.cells if cell.cell_type=='code' for output in cell.get('outputs',[]) if output.output_type=='error']
    assert not errors
    print('Executed',len(nb.cells),'cells without errors:',ROOT/'part2.ipynb')

if __name__=='__main__':run()
