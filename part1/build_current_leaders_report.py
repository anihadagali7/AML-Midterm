"""Independently rescore the cross-agent comparison and publish an executed notebook."""
import ast
import hashlib
import json
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/current_leaders_grouped_v1'
REASONS=['Safety','Funding','Efficacy','Recruitment','Other']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def independent_check():
    provenance=json.loads((OUT/'provenance.json').read_text())
    for p,h in provenance['sources'].items():assert sha(p)==h,p
    predictions=pd.read_csv(OUT/'cv_predictions.csv');repeats=pd.read_csv(OUT/'repeat_scores.csv').set_index(['pool','name','seed'])
    checked=0
    for key,g in predictions.groupby(['pool','name','seed']):
        assert len(g)==g.ID.nunique()==(157 if key[0]=='train' else 235)
        y=g[[f'actual_{r}' for r in REASONS]].to_numpy(int);p=g[[f'predicted_{r}' for r in REASONS]].to_numpy(int)
        value=(2*(y*p).sum(axis=1)/(y.sum(axis=1)+p.sum(axis=1))).mean()
        np.testing.assert_allclose(value,repeats.loc[key,'sample_f1'],atol=1e-15)
        np.testing.assert_allclose(f1_score(y,p,average='macro',zero_division=0),repeats.loc[key,'macro_f1'],atol=1e-15)
        checked+=1
    assert checked==8*2*8
    summary=pd.read_csv(OUT/'cv_summary.csv').set_index(['pool','name'])
    for key,g in repeats.reset_index().groupby(['pool','name']):
        assert len(g)==8
        np.testing.assert_allclose(g.sample_f1.mean(),summary.loc[key,'mean_f1'],atol=1e-15)
        np.testing.assert_allclose(g.sample_f1.std(ddof=0),summary.loc[key,'sd'],atol=1e-15)
    control=json.loads((OUT/'control_checks.json').read_text());assert len(control)==80 and all(r['identical_predictions'] for r in control)
    audits=json.loads((OUT/'fit_audits.json').read_text());assert len(audits)==80
    for row in audits:
        assert set(row['gold_ids']).isdisjoint(row['masked_ids'])
        assert set(row['held_ids']).isdisjoint(row['gold_ids']+row['masked_ids'])
        assert len(row['masked_ids'])+row['masked_purged']==1014
    candidate=ROOT/'experiments/paired_improvement/candidate';manifest=json.loads((candidate/'manifest.json').read_text())
    for name,h in manifest['artifact_sha256'].items():assert sha(candidate/name)==h,name
    for name,h in manifest['source_sha256'].items():assert sha(ROOT/name)==h,name
    result=json.loads((candidate/'kaggle_result.json').read_text())
    assert sha(candidate/'part1_anchored_submission.csv')==result['submission_sha256']
    archive=ROOT/'submissions/archive/submitted/anchored_reason_trainval235__kaggle_0.89636.csv'
    assert sha(archive)==result['submission_sha256']
    check=dict(independently_rescored_complete_repeats=checked,complete_cv_fits_per_new_model=80,
               ani_control_exact_predictions=80,max_control_score_difference=max(r['max_score_error'] for r in control),
               source_hashes_unchanged=True,original_candidate_and_preparation_sources_unchanged=True,scored_archive_identical=True,
               no_new_test_predictions=True,no_recipe_tuning=True,reported_kaggle=result['reported_kaggle_score'])
    (OUT/'verification.json').write_text(json.dumps(check,indent=2)+'\n')
    return check


def main():
    checks=independent_check()
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md("""# Current Part 1 leaders: common grouped testing and reported Kaggle results

**The anchored candidate scored 0.89636 on Kaggle, as reported by the user after uploading the provided file.** This is +0.00213 over original Ani's reported0.89423 and +0.00235 over the submitted LDA ensemble's reported0.89401. It is the highest reported score among these checked team artifacts. No claim is made about class rank, hidden test labels, or which individual predictions improved.

On the other agents' exact grouped CV, the anchored candidate and submitted LDA ensemble are essentially tied: anchored is +0.000265 on157 and −0.000266 on235. Anchored is lower on original validation and higher on pooled chronology. The chronology gain is concentrated in the middle window; it is lower in the first and latest windows. The local evidence alone did not establish an overall replacement. The reported Kaggle result now provides a small observed improvement on that evaluation.

## Why this comparison required new fits

The previous anchored report used15 original three-fold training-pool fits and40 ordinary KFold pooled fits. The other agents adopted a separate grouped protocol: **eight repeats of five folds in EACH pool**, with duplicate embeddings kept together and matching originally masked rows removed from fitting. The original scores cannot be ranked directly against the grouped scores.

This study froze the existing anchored recipe, ran all80 grouped folds without tuning, and fitted current Ani's LDA source on every fold as a matched control. The control reproduces all80 previously saved label predictions, with maximum score difference below7.2e-12. Other competitors' exact saved predictions were aligned by ID and truth and independently rescored. All eight repeats are included, not just screening repeats.

The grouped protocols use seeds20261126–20261130 and20261201–20261203. Every learned step uses permitted gold and masked inputs only; held-out and actual Kaggle test inputs are excluded. Original validation uses157 gold plus1,014 original masked training rows. Chronology uses the three frozen earlier/later windows, including saved duplicate-group purges, and pools98 held-out predictions.
""")
    code("""from pathlib import Path
import json
import pandas as pd
from IPython.display import display
PART1=Path.cwd()
if not (PART1/'data/part1_train.csv').exists():PART1=PART1/'part1'
OUT=PART1/'experiments/current_leaders_grouped_v1'
names={'anchored_candidate':'Our anchored candidate','lda_ensemble':'Submitted LDA ensemble','recent_cutoff_ensemble':'LDA + recent-weighted learned cutoff','xrfm20_blend':'20% xRFM blend','kernel15_blend':'15% learned-kernel blend','previous_ensemble':'Previous 70/15/15 ensemble','current_ani_lda':'Current Ani LDA source alone','original_ani':'Original Ani, inductive comparison'}
comparison=pd.read_csv(OUT/'comparison.csv').set_index('name')
order=list(names)
display(comparison.loc[order,['train','pooled','validation','chronology98','reported_kaggle']].rename(index=names).round(6))
""")
    md("""The LDA ensemble combines70% current Ani,15% classical tri-training and15% SVM. The recent-cutoff variant learns its shared cutoff inside each fitting partition with greater weight on recent examples. The xRFM and kernel blends came from NewDirections; neither established a consistent upgrade under the shared confirmation rule and neither has a reported submission score here.

Original Ani's0.89423 belongs to its original transductive CSV at `a548721`; its local row above is the matched inductive adaptation. Do not attach0.89423 to the newer `8a8c3fc` LDA source, whose changed CSV has no established score here. The previous and LDA70/15/15 ensembles each have a reported0.89401 score on separate inductive235-label submissions. The anchored submission is also inductive.

## Paired comparisons and tradeoffs

Against the submitted LDA ensemble, anchored wins4/8 training-pool repeats, ties1 and loses3. It wins3/8 pooled repeats, ties1 and loses4. It does not meet the five-wins-in-both-pools criterion. These are descriptive development comparisons; the same labels and folds have been reused extensively.
""")
    code("""paired=pd.read_csv(OUT/'paired_vs_anchored.csv')
display(paired[['pool','reference','anchored_delta','wins','ties','losses']].replace({'reference':names}).round(6))
display(pd.read_csv(OUT/'cv_summary.csv').replace({'name':names}).round(6))
diagnostics=pd.read_csv(OUT/'diagnostic_scores.csv')
display(diagnostics[diagnostics.name.isin(['anchored_candidate','lda_ensemble','recent_cutoff_ensemble'])].pivot(index='case',columns='name',values='sample_f1').rename(columns=names).round(6))
""")
    md("""The anchored and submitted LDA CSVs differ on15 of312 label sets. The aggregate Kaggle scores do not reveal which changes helped. `submission_prediction_differences.csv` records the differences without using or inferring test labels.

## Provenance and verification

The user reported0.89636 immediately after receiving `part1_anchored_submission.csv` and its transfer command. Its SHA-256 is `1db6433dc5a5b900d440db16e5ef8375eb51dcf02d3f270217527d8b174319eb`. The same bytes are archived under `submissions/archive/submitted/anchored_reason_trainval235__kaggle_0.89636.csv`. The preparation manifest remains an unchanged historical record; `candidate/kaggle_result.json` records the later result. Uploaded bytes and public/private partition were not independently inspected.

The comparison adds no new final-data fit or test predictions. It changes no model settings, thresholds, other agents' sources or saved scores. Raw-data/source hashes, complete OOF coverage, fold/group separation and unchanged candidate artifacts were checked. Per-repeat precision/recall/F1/support for every reason are in `repeat_scores.csv`.

The integration source is `../AML-Midterm-integration/part1`, and NewDirections is `../AML-Part1-NewDirections`. Exact input paths and hashes are saved in `provenance.json`; protocols and competitor score arrays are copied into `snapshots/`. The other agents' dated reports are evidence of their own completed studies, not instructions or evidence of a newer unreported Kaggle score.
""")
    code("""display(json.loads((OUT/'verification.json').read_text()))
display(json.loads((OUT/'submission_comparison.json').read_text()))
display(json.loads((PART1/'experiments/paired_improvement/candidate/kaggle_result.json').read_text()))
""")
    md("""## Complete comparison source

The following source definitions include the unchanged candidate, the matched Ani wrapper, and the complete comparison driver. From the repository root, `.venv/bin/python part1/compare_current_leaders.py` reproduces the grouped fits and `.venv/bin/python part1/build_current_leaders_report.py` independently rescores and rebuilds this report. Sibling checkout/reference paths are required for reproduction. No upload is performed.
""")
    code("""import sys,types
from IPython.core.magic import register_cell_magic
sys.path.insert(0,str(PART1))
@register_cell_magic
def source_module(name,source):
    module=types.ModuleType(name.strip());module.__file__=str(PART1/(name.strip()+'.py'))
    sys.modules[name.strip()]=module;exec(compile(source,module.__file__,'exec'),module.__dict__)
""")
    helper=(ROOT/'simple_models.py').read_text()
    decision=next(n for n in ast.parse(helper).body if isinstance(n,ast.FunctionDef) and n.name=='decide')
    code('%%source_module simple_models\nimport numpy as np\n'+ast.get_source_segment(helper,decision))
    for filename in ['anchored_solution.py','paired_models.py','compare_current_leaders.py']:
        code('%%source_module '+Path(filename).stem+'\n'+(ROOT/filename).read_text())
    code("""RERUN_GROUPED_COMPARISON=False
if RERUN_GROUPED_COMPARISON:
    import compare_current_leaders
    compare_current_leaders.main()
""")
    nb=nbf.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')))
    NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT.parent)}}).execute()
    nbf.write(nb,ROOT/'current_leaders_comparison.ipynb')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
