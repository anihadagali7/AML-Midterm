"""Update deliverables with the requested frozen-candidate validation measurement."""
from pathlib import Path
import json,shutil,zipfile
import nbformat as nbf
from nbconvert import HTMLExporter
r=json.loads(Path('experiments/final-validation.json').read_text());score=r['sample_f1']
notice=f'''LATEST FINAL-CANDIDATE VALIDATION RESULT
Sample-wise F1: {score:.8f}; macro F1: {r['macro_f1']:.8f}; 78 trials.
This is the exact delivered full-fit candidate with the same model settings and 0.35 threshold. Its test probabilities and submission predictions were verified unchanged. Its training CV F1 is 0.93631; that did not transfer to validation. Improvement over the user's prior model has not been demonstrated. The first bagged comparison coincidentally had the same best sample-wise F1; the per-label and macro scores differ.
Details: experiments/final-validation.json and predictions/fullfit_validation_audit.csv.
'''
p=Path('READ_ME_FIRST.txt');s=p.read_text()
if not s.startswith('LATEST FINAL-CANDIDATE'):s=notice+'\n'+s
s=s.replace('This is NOT a Kaggle score or a new external validation result.','This CV value is not a Kaggle score. The exact full-fit candidate subsequently scored 0.850427 on validation with its settings unchanged.')
s=s.replace('External validation was used once for a frozen 3-candidate comparison. Later work did not recheck it.','External validation was first used for a frozen 3-candidate comparison. At the user’s explicit request, the final full-fit candidate was then evaluated once with its settings unchanged; it scored 0.850427.')
p.write_text(s)
p=Path('midterm/01_experiment_report.ipynb');nb=nbf.read(p,as_version=4)
nb.cells[0].source=f'''# AML Midterm — Part 1 experiment report

**Final candidate validation F1: {score:.5f}.** Macro F1: {r['macro_f1']:.5f}, measured on all 78 validation rows. The model configuration and threshold were unchanged. Its test probabilities match the delivered submission.

The candidate's **0.93631 training-only grouped leave-one-out F1 did not transfer to validation**. No improvement over the user's existing model has been demonstrated, and a leaderboard win is not established. The initial bagged-model comparison also had a best sample-F1 of 0.85043; these are separate measurements. Historical search and first-comparison details follow.
'''
for cell in nb.cells:
 if cell.cell_type!='markdown':continue
 cell.source=cell.source.replace('These are the only three external-validation model evaluations.','These are the three models in the initial frozen external-validation comparison. The final full-fit candidate was evaluated separately at the user’s request; see the latest result above.')
 cell.source=cell.source.replace('After its disappointing result, later experiments remained training-only; no individual validation-error retuning or repeated validation scoring occurred.','After its disappointing result, subsequent model development remained training-only. The final candidate was later evaluated at the user’s request, with no threshold or model retuning.')
 cell.source=cell.source.replace('The final candidate was **not re-evaluated on the used validation set**. Its 0.93631 leave-group-out score remains a selection-biased development estimate, awaiting external confirmation.','The exact final candidate was subsequently evaluated on validation at the user’s request: **0.85043 sample-F1**. Its 0.93631 leave-group-out score remains a selection-biased development estimate and is substantially higher than the observed validation result.')
 cell.source=cell.source.replace('Submit `predictions/part1_fullfit_candidate.csv` once and record its Kaggle result before claiming improvement.','The final candidate does not demonstrate an improvement on validation; do not infer a leaderboard improvement from its CV score.')
 cell.source=cell.source.replace('The final full-fit regime addresses training-size differences but does not erase that evidence.','The full-fit candidate also scored 0.85043 on validation, confirming that the training-size change did not solve the measured gap.')
# Add the exact measurement and its source once, leaving historical outputs intact.
if not any(c.cell_type=='markdown' and c.source.startswith('## Requested final-candidate evaluation') for c in nb.cells):
 table='| Reason | Validation F1 |\n|---|---:|\n'+'\n'.join(f"| {l} | {r['f1_'+l]:.5f} |" for l in ['Safety','Funding','Efficacy','Recruitment','Other'])
 nb.cells.insert(1,nbf.v4.new_markdown_cell('## Requested final-candidate evaluation\n\n'+table+'\n\nSource: `experiments/final-validation.json`. Reproducible evaluation: `evaluate_final_validation.py`. This evaluation does not alter any training labels, settings, threshold, or Kaggle predictions.'))
nbf.validate(nb);nbf.write(nb,p)
body,_=HTMLExporter().from_notebook_node(nb);p.with_suffix('.html').write_text(body)
p=Path('midterm/02_reproduce.ipynb');nb=nbf.read(p,as_version=4)
if not any(c.cell_type=='markdown' and c.source.startswith('## Exact final validation evaluation') for c in nb.cells):
 nb.cells.append(nbf.v4.new_markdown_cell('## Exact final validation evaluation\n\nAt the user’s request, the frozen full-fit candidate was evaluated: sample-F1 0.85043, macro-F1 0.68841. The script below reproduces that measurement without changing the model or threshold and verifies the original test predictions. It is saved here but is not automatically executed.'))
 nb.cells.append(nbf.v4.new_code_cell('evaluation_source = '+repr(Path('evaluate_final_validation.py').read_text())+'\n(replica / "evaluate_final_validation.py").write_text(evaluation_source)\n'))
nbf.validate(nb);nbf.write(nb,p)
checks=json.loads(Path('experiments/deliverable-checks.json').read_text());checks.update(external_validation_comparisons=2,external_validation_candidate_count=4,final_candidate_validation_f1=score,final_candidate_test_predictions_verified_unchanged=True)
Path('experiments/deliverable-checks.json').write_text(json.dumps(checks,indent=2))
for name in ['READ_ME_FIRST.txt','evaluate_final_validation.py','update_validation_docs.py','make_candidate.py','build_deliverables.py']:
 shutil.copy2(name,Path('midterm')/name)
for name in ['final-candidate.json','final-validation.json','deliverable-checks.json']:
 shutil.copy2(Path('experiments')/name,Path('midterm/experiments')/name)
shutil.copy2('predictions/fullfit_validation_audit.csv','midterm/predictions/fullfit_validation_audit.csv')
with zipfile.ZipFile('AMLMidterm_Part1.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in Path('midterm').rglob('*'):
  if p.is_file() and not any(part in {'reproduction','__pycache__','.DS_Store'} for part in p.parts):z.write(p,p.as_posix())
print('Updated report, notebooks, result records and archive. Final validation sample-F1:',score)
