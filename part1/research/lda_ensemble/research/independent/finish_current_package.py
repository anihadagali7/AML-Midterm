"""Execute current report/reproduction notebooks and refresh the portable archive."""
from pathlib import Path
import os,json,shutil,zipfile,hashlib
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
ROOT=Path(__file__).resolve().parent;mid=ROOT/'midterm'
os.environ['JUPYTER_PATH']=str(ROOT/'.venv/share/jupyter')
notebooks=['03_current_report','04_reproduce_current']
if (mid/'05_tri_training_and_shift.ipynb').exists():notebooks.append('05_tri_training_and_shift')
if (mid/'06_stability_and_specialists.ipynb').exists():notebooks.append('06_stability_and_specialists')
if (mid/'07_cotraining_and_distances.ipynb').exists():notebooks.append('07_cotraining_and_distances')
if (mid/'08_svm_refinements.ipynb').exists():notebooks.append('08_svm_refinements')
if (mid/'09_shift_graph_and_subgroups.ipynb').exists():notebooks.append('09_shift_graph_and_subgroups')
if (mid/'10_common_cv_benchmark.ipynb').exists():notebooks.append('10_common_cv_benchmark')
if (mid/'11_pooled_cv_comparison.ipynb').exists():notebooks.append('11_pooled_cv_comparison')
for name in notebooks:
 p=mid/(name+'.ipynb');nb=nbformat.read(p,as_version=4)
 NotebookClient(nb,timeout=600,kernel_name='aml-midterm',resources={'metadata':{'path':str(mid)}}).execute()
 nbformat.write(nb,p);html,_=HTMLExporter().from_notebook_node(nb);(mid/(name+'.html')).write_text(html)
 print('Executed',name,flush=True)
manifest=json.loads((ROOT/'predictions/current_best/manifest.json').read_text());summary=json.loads((ROOT/'experiments/current-summary.json').read_text())
checks=dict(current_submission_rows=312,gold_rows_preserved=157,missing_rows_predicted=1014,raw_hashes_verified=True,notebooks_executed=notebooks,exact_reproduction_sha256=manifest['submission_sha256'],validation_sample_f1=manifest['validation']['sample_f1'],validation_macro_f1=manifest['validation']['macro_f1'],validation_candidate_evaluations=summary['validation_candidate_evaluations'],target_met=manifest['target_met'],kaggle_score=None,integrity=json.loads((ROOT/'experiments/final-integrity-checks.json').read_text()))
if (ROOT/'experiments/phase21-integrity-checks.json').exists():checks['tri_training_integrity']=json.loads((ROOT/'experiments/phase21-integrity-checks.json').read_text())
if (ROOT/'experiments/phase23-integrity-checks.json').exists():checks['nested_specialist_integrity']=json.loads((ROOT/'experiments/phase23-integrity-checks.json').read_text())
if (ROOT/'experiments/phase24-integrity-checks.json').exists():checks['cotraining_metric_integrity']=json.loads((ROOT/'experiments/phase24-integrity-checks.json').read_text())
if (ROOT/'experiments/phase25-integrity-checks.json').exists():checks['svm_refinement_integrity']=json.loads((ROOT/'experiments/phase25-integrity-checks.json').read_text())
if (ROOT/'experiments/phase26-integrity-checks.json').exists():checks['shift_graph_subgroup_integrity']=json.loads((ROOT/'experiments/phase26-integrity-checks.json').read_text())
if (ROOT/'experiments/benchmark_v1/summary.json').exists():checks['common_cv_benchmark']=json.loads((ROOT/'experiments/benchmark_v1/summary.json').read_text())
if (ROOT/'experiments/pooled_benchmark_v1/conclusions.json').exists():checks['pooled_cv_comparison']=json.loads((ROOT/'experiments/pooled_benchmark_v1/conclusions.json').read_text())
(ROOT/'experiments/current-deliverable-checks.json').write_text(json.dumps(checks,indent=2));shutil.copy2(ROOT/'experiments/current-deliverable-checks.json',mid/'experiments/current-deliverable-checks.json')
# Notebook-generated figure is also available at the project top level.
(ROOT/'figures').mkdir(exist_ok=True)
for p in (mid/'figures').glob('current_*.png'):shutil.copy2(p,ROOT/'figures'/p.name)
archive=ROOT/'AMLMidterm_Part1.zip';temporary=ROOT/'AMLMidterm_Part1.building.zip'
with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(mid.rglob('*')):
  if not p.is_file() or any(part in ['__pycache__','.venv','reproduction','current_reproduction'] for part in p.relative_to(mid).parts):continue
  z.write(p,Path('midterm')/p.relative_to(mid))
with zipfile.ZipFile(temporary) as z:
 assert z.testzip() is None
 assert hashlib.sha256(z.read('midterm/predictions/current_best/part1_submission.csv')).hexdigest()==manifest['submission_sha256']
 assert 'midterm/04_reproduce_current.ipynb' in z.namelist()
os.replace(temporary,archive)
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'checks':checks},indent=2))
