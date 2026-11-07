"""Create an executed notebook containing all local-benchmark source/results."""
import json,os,sys
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parent
MODULES=['experiments.py','ani_models.py','benchmark_models.py','benchmark_protocol.py','local_benchmark.py',
         'reproduce_ani.py','benchmark_ani_recipe.py','benchmark_candidates.py','benchmark_final.py',
         'benchmark_report.py','test_local_benchmark.py','verify_benchmark.py']


def run():
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s,hidden=False):
        c=nbf.v4.new_code_cell(s)
        if hidden:c.metadata['jupyter']={'source_hidden':True}
        cells.append(c)
    md('''# Part Two: repeatable local testing and Ani comparison

**Hold submission 05.** The known reference is Ani's Kaggle **0.58688**, compared with our **02: 0.58993** and **03: 0.59562**. A higher validation F1, or even lower validation log loss, did not establish a better Kaggle model.

This notebook freezes trial-level splits, reproduces Ani's published predictions, compares models on grouped CV and chronological holdouts, and retains uncertainty and source provenance. All implementation code is included below. No Kaggle upload is performed.
''')
    code('''from pathlib import Path
import os,sys,json
ROOT=Path.cwd()
if (ROOT/"AML-Midterm/part2").is_dir():ROOT=ROOT/"AML-Midterm/part2"
assert (ROOT/"outcome_train.csv").exists()
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
os.environ["OMP_NUM_THREADS"]="4"
os.environ["OPENBLAS_NUM_THREADS"]="4"
os.environ["MPLCONFIGDIR"]=str(ROOT/"experiments/.matplotlib")
''')
    md('''## Complete source

Collapsed cells materialize the exact implementation. Training is controlled separately and is off by default.
''')
    for file in MODULES:code('%%writefile '+file+'\n'+(ROOT/file).read_text(),True)
    md('''## Protocol and reproduction

Ten fixed comparisons: supplied validation, three nonoverlapping two-year future windows, and two repeats of trial-grouped three-fold CV. The primary metric is mean log loss with equal weights for Phases I/II/III. Combined-phase rows stay together. All fitting labels and preprocessing are isolated from the held-out trials. The strict stacking pipeline refits feature preprocessing inside its inner folds as well.

The reference recipe separately reproduces Ani's original transductive vocabulary/IDF and final supervised train+validation refit. These two tracks remain explicit; a feature-policy change is not silently attributed to the model alone.
''')
    code('''RUN_TRAINING=False
import subprocess
if RUN_TRAINING:
    for command in [["reproduce_ani.py"],["local_benchmark.py","--suite","full"],
                    ["benchmark_ani_recipe.py"],["benchmark_candidates.py"],
                    ["benchmark_final.py"],["benchmark_report.py"]]:
        subprocess.run([sys.executable]+command,check=True)
import pandas as pd
import numpy as np
from IPython.display import display,Image
from benchmark_protocol import DIRECTORY,initialize
universe,protocol=initialize()
OUT=ROOT/json.loads((DIRECTORY/"current_run.json").read_text())["directory"]
display(pd.DataFrame([{k:v for k,v in s.items() if k not in ["fit_ids","eval_ids"]} for s in protocol["splits"]]))
display(json.loads((OUT/"run_manifest.json").read_text()))
''')
    md('''## Exact Ani reproduction

Source: [Ani commit 1b8bd11](https://github.com/anihadagali7/AML-Midterm/tree/1b8bd11c47428fb5453e53cdc275542b92a3b347). Frozen notebooks and CSVs are preserved under `experiments/ani_comparison/source` with hashes. All three outcome data files match the branch byte for byte.

Ani uses eight pooled field-specific predictors with grouped-OOF logistic stacking and exponential recency tau=3. His final model fits 9,050 train+validation rows; our initial submissions fitted 8,037 training rows. His feature vocabulary/IDF uses all known covariates, including test text. Recency weights do not affect his similarity-weighted kNN component. No stopping-reason inputs enter either outcome classifier.
''')
    code('''display(pd.read_csv(ROOT/"experiments/ani_comparison/exact_reproduction.csv"))
display(json.loads((ROOT/"experiments/ani_comparison/provenance.json").read_text()))
reproduction=json.loads((ROOT/"experiments/ani_comparison/exact_reproduction.json").read_text())
assert reproduction[1]["max_submission_probability_difference"]<1e-7
''')
    md('''## Strict comparison

Read chronological mean, worst period, per-phase changes and grouped CV together. Group-CV scores pool all held-out rows within each repeat before averaging the repeats. F1, ROC-AUC, average precision, Brier score and predicted/observed success rates are secondary diagnostics in `fold_scores.csv`.

The six recipes were evaluated on the same saved splits. Do not select a favorable subset of folds. Small local differences do not establish performance on 2014–2021 trials.
''')
    code('''summary=pd.read_csv(OUT/"summary.csv")
assert summary.complete_comparison.all()
display(summary)
scores=pd.read_csv(OUT/"fold_scores.csv")
display(scores[scores.kind.eq("temporal")][["model","split","phase_mean_logloss","I_logloss","II_logloss","III_logloss"]])
display(Image(filename=str(OUT/"figures/chronological_windows.png")))
''')
    md('''## Paired uncertainty

Intervals resample trials jointly, keeping combined-phase rows together. Repeated-CV losses are averaged per trial before resampling. Temporal intervals average the three periods equally. Negative differences favor a candidate relative to Ani. Intervals spanning zero are inconclusive.

These are descriptive, conditional intervals. They do not correct fully for model selection or represent the unseen cohort shift. The most recent temporal window has only 94 Phase I and 168 Phase III rows.
''')
    code('''display(Image(filename=str(OUT/"figures/paired_comparisons.png")))
display(summary[["model","temporal_delta","temporal_delta_lower","temporal_delta_upper", "group_cv_delta","group_cv_delta_lower","group_cv_delta_upper"]])
''')
    md('''## Faithful published-recipe track

The same labels and trial splits are used, but Ani's original feature-only transduction is preserved. Evaluation labels never fit a predictor. Compare this sensitivity track separately from the strict prospective benchmark.
''')
    code('''published=pd.read_csv(OUT/"published_recipe_track/summary.csv")
assert published.complete_comparison.all()
display(published)
''')
    md('''## Optional conservative candidates

07 stays anchored to the winning published file and puts 25% weight on Ani's existing capped Phase III year-trend alternative. 08 blends 75% of that winning file with 25% of our linear component, now refitted on the same 9,050 labeled rows as Ani's final fit. Both were evaluated using matching component predictions on every saved split in both tracks. They are exploratory candidates with unknown Kaggle scores, not established improvements.

05 remains on hold because the new time-based comparisons favor Ani over our linear endpoint. Preserve the 0.58688 file until a new submitted score actually improves it. Full source recipes and hashes are in `submissions/ani_candidates_manifest.json`.
''')
    code('''for track in [OUT,OUT/"published_recipe_track"]:
    display(pd.read_csv(track/"derived_candidates/summary.csv"))
display(pd.read_csv(ROOT/"submissions/ani_candidates_index.csv"))
display(json.loads((ROOT/"submissions/ani_candidates_manifest.json").read_text()))
''')
    md('''## Repeatability checks and commands

The protocol tests check trial separation, test-set exclusion, repeated-CV coverage, chronological ordering, equal-phase loss, a known paired-bootstrap result and incomplete-run handling. Run the commands from the repository root:

```sh
make -C AML-Midterm/part2 benchmark-test
make -C AML-Midterm/part2 benchmark
make -C AML-Midterm/part2 benchmark-report
```

New code or package versions receive a separate result fingerprint. Dataset changes are rejected by the existing protocol. Final fitting and held-out benchmark scores remain separate. See `LOCAL_TESTING.md` for the full operating procedure and limitations.
''')
    code('''from benchmark_protocol import resolve
for split in protocol["splits"]:
    train,ev=resolve(universe,split)
    assert not set(train.trial_id)&set(ev.trial_id)
for path in (OUT/"models").glob("*/audit.json"):
    audit=json.loads(path.read_text())
    assert audit["trial_overlap"]==0
    assert all(f["trial_overlap"]==0 for f in audit["inner_folds"])
sample=pd.read_csv(ROOT/"part2_sample_submission.csv")
for row in pd.read_csv(ROOT/"submissions/ani_candidates_index.csv").itertuples():
    result=pd.read_csv(ROOT/"submissions"/row.submission)
    assert len(result)==1500 and result.ID.is_unique and result.ID.equals(sample.ID)
    assert result.label.between(0,1,inclusive="neither").all()
print("All ten split audits and both candidate CSVs pass.")
verification=ROOT/"experiments/local_benchmark/verification.json"
if verification.exists():display(json.loads(verification.read_text()))
''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python','version':sys.version.split()[0]}})
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    for key,folder in [('IPYTHONDIR','.ipython'),('JUPYTER_RUNTIME_DIR','.jupyter_runtime'),('MPLCONFIGDIR','.matplotlib')]:
        os.environ[key]=str(ROOT/'experiments'/folder);Path(os.environ[key]).mkdir(parents=True,exist_ok=True)
    NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbf.write(nb,ROOT/'local_benchmark.ipynb')
    assert not any(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]))
    print('Executed benchmark notebook:',ROOT/'local_benchmark.ipynb')

if __name__=='__main__':run()
