"""Read-only audit of the two exports and all three retained study batches."""
import importlib.util
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('halfstep_audit',HERE.parent/'blend_halfstep_v1/study.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
p,b=h.p,h.b
for module in (b,p,h):module.guard()
data,partner=p.cv_components()
all_predictions={n:b.blend(data['scores'][0],data['scores'][2],s)[1] for n,s in b.SPECS.items()}
all_predictions.update({n:b.blend(data['scores'][0],partner,dict(lda_weight=w,cutoff=.5))[1] for n,w in p.WEIGHTS.items()})
all_predictions[h.NAME]=h.score(data['scores'][0],data['scores'][2],partner)[1]
count=0
for folder in (b.HERE,p.HERE,h.HERE):
    for table in ('repeat_scores','supplementary_scores'):
        for row in pd.read_csv(folder/'results'/f'{table}.csv').to_dict('records'):
            if table=='repeat_scores':mask=(data['pool']==row['pool'])&(data['seed']==row['seed'])
            else:mask=((data['pool']=='supplementary')&(data['case']!='validation')) if row['case']=='chronology98' else data['case']==row['case']
            metrics=b.reference.metrics(data['truth'][mask],all_predictions[row['mode']][mask])
            for key,value in metrics.items():
                assert abs(value-row[key])<1e-12,(folder,row['mode'],key)
                count+=1
    cv=pd.read_csv(folder/'results/repeat_scores.csv')
    for row in pd.read_csv(folder/'results/cv_summary.csv').to_dict('records'):
        values=cv[(cv.pool==row['pool'])&(cv['mode']==row['mode'])]
        assert len(values)==8
        for key,actual in [('mean_sample_f1',values.sample_f1.mean()),('population_sd',values.sample_f1.std(ddof=0)),('mean_macro_f1',values.macro_f1.mean())]:
            assert abs(actual-row[key])<1e-12
    for row in pd.read_csv(folder/'results/paired_summary.csv').to_dict('records'):
        mode=row.get('mode',h.NAME)
        pivot=cv[cv.pool==row['pool']].pivot(index='seed',columns='mode',values='sample_f1')
        delta=pivot[mode]-pivot[row['baseline']]
        assert abs(delta.mean()-row['mean_delta'])<1e-12
        assert (int((delta>1e-12).sum()),int((abs(delta)<=1e-12).sum()),int((delta< -1e-12).sum()))==(row['wins'],row['ties'],row['losses'])
    if (folder/'retention.json').exists():
        for path,digest in json.loads((folder/'retention.json').read_text())['files'].items():assert b.sha(b.ROOT/path)==digest,path
with np.load(b.FINAL/'prediction_scores.npz') as z:f={k:z[k] for k in z.files}
with np.load(p.FINAL) as z:
    np.testing.assert_array_equal(z['ids'],f['ids'])
    np.testing.assert_array_equal(z['scores'],f['lda_ensemble'])
    source=z['lda_source__inductive']
raw=np.clip(f['anchored_raw'],1e-9,1-1e-9)
a=.7*raw/(.7*raw+.3*(1-raw))
np.testing.assert_allclose(a,f['anchored_centered'],atol=1e-12,rtol=0)
partner=(f['lda_ensemble']-.7*source)/.3
sample=pd.read_csv(b.PART1/'data/part1_sample_submission.csv')
records=json.loads((HERE/'submissions.json').read_text())+[json.loads((p.HERE/'submission.json').read_text())]
outcomes=[json.loads((folder/'outcome.json').read_text()) for folder in (HERE,p.HERE)]
expected_scores=[.5*a+.5*f['lda_ensemble'],.9*a+.1*partner]
saved=[HERE/'blend50_50_final.npz',p.HERE/'finalist_test.npz']
outputs=[]
for record,outcome,scores,path in zip(records,outcomes,expected_scores,saved):
    assert outcome['status']=='not_submitted' and outcome['reported_kaggle_score'] is None
    assert outcome['submission_sha256']==record['sha256']
    pred=(scores>=.5).astype(int)
    for i in range(len(pred)):
        if not pred[i].any():pred[i,scores[i].argmax()]=1
    csv=b.ROOT/outcome['archived_submission_file'];frame=pd.read_csv(csv)
    assert b.sha(csv)==record['sha256'] and len(frame)==312 and frame.ID.is_unique
    np.testing.assert_array_equal(frame.ID,sample.ID)
    labels=[' '.join(label for label,flag in zip(b.LABELS,row) if flag) for row in pred]
    assert frame.Predicted.tolist()==labels
    assert not path.exists(), 'Closed trials retain reconstructed scores, not duplicate prediction dumps.'
    changed=int((pred!=f['predictions']).any(axis=1).sum())
    assert changed==record['changed_labelsets_vs_current']>0
    outputs.append(pred)
    cvpath=HERE/'finalist_cv.npz' if record['recipe']=='blend50_50' else p.HERE/'finalist_cv.npz'
    assert not cvpath.exists()
    retained=json.loads((cvpath.parent/'retention.json').read_text())
    assert retained['candidate_prediction_arrays_retained'] is False
distinct=int((outputs[0]!=outputs[1]).any(axis=1).sum());assert distinct>0
result_path='part1/submissions/results/anchored75-lda25-trainval235.json'
result=json.loads((b.ROOT/result_path).read_text())
tag=result['tag']
git=lambda *args:subprocess.check_output(['git','-C',str(b.ROOT),*args])
assert git('cat-file','-t','refs/tags/'+tag).strip()==b'tag'
commit=git('rev-parse',tag+'^{commit}').decode().strip()
assert commit=='39a5b46a58c650c29d921ae2dabb8cf2a29ab403'
assert git('show',commit+':'+result_path)==(b.ROOT/result_path).read_bytes()
for path,key in [(result['submission_file'],'submission_sha256'),(result['frozen_recipe_file'],'frozen_recipe_sha256')]:
    assert b.sha(b.ROOT/path)==result[key]
    assert git('show',commit+':'+path)==(b.ROOT/path).read_bytes()
print(json.dumps(dict(passed=True,metric_values_recomputed=count,all_cases=84,distinct_labelsets_between_candidates=distinct,export_rows_each=312,closed_trials=2,submitted_result_commit=commit,submitted_result_tag=tag),indent=2))
