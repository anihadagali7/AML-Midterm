"""Evaluate the already-selected full-fit candidate, without retuning or validation fitting."""
from pathlib import Path
import json, hashlib, datetime
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import experiment as e

manifest_path=Path('experiments/final-candidate.json')
manifest=json.loads(manifest_path.read_text())
assert hashlib.sha256(Path('experiment.py').read_bytes()).hexdigest()==manifest['engine_sha256']
assert hashlib.sha256(Path(manifest['submission']).read_bytes()).hexdigest()==manifest['submission_sha256']
selected=manifest['selected']
val=pd.read_csv('data/part1_val.csv',usecols=lambda c:c not in e.LABELS)
test=pd.read_csv('data/part1_test.csv')
unlabeled=e.DATA.iloc[e.UNLAB].drop(columns=e.LABELS)
query=pd.concat([val,test,unlabeled],ignore_index=True)
with threadpool_limits(limits=1):
    p=e.predict(np.arange(len(e.Y)),query,selected['config'])
# Check this is the same candidate that generated the delivered Kaggle predictions.
saved=pd.read_csv('predictions/fullfit_test_probabilities.csv')
np.testing.assert_allclose(p[len(val):len(val)+len(test)],saved[['p_'+l for l in e.LABELS]].to_numpy(),rtol=1e-10,atol=1e-12)
test_predictions=e.decisions(p[len(val):len(val)+len(test)],selected['rule'])
strings=[' '.join(l for l,v in zip(e.LABELS,row) if v) for row in test_predictions]
assert strings==pd.read_csv(manifest['submission']).Predicted.tolist()
# Validation labels are accessed only after fitting and prediction.
y=pd.read_csv('data/part1_val.csv',usecols=e.LABELS)[e.LABELS].to_numpy(int)
vp=p[:len(val)];hard=e.decisions(vp,selected['rule'])
result={
    'evaluated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'candidate':selected['name'],'n_validation':len(val),
    'rule':selected['rule'],'config':selected['config'],
    **e.metrics(y,vp,selected['rule']),
    'training_gold_rows':len(e.Y),'validation_used_for_fitting':False,
    'settings_or_threshold_changed':False,'same_test_probabilities_verified':True,
    'engine_sha256':manifest['engine_sha256'],'submission_sha256':manifest['submission_sha256'],
    'context':'User requested direct evaluation of the frozen full-fit candidate. This is a second validation evaluation occasion, with no retuning. Earlier bagged-model validation results remain separate.'
}
e.save_json('experiments/final-validation.json',result)
out=pd.DataFrame(vp,columns=['p_'+l for l in e.LABELS]);out.insert(0,'ID',val.ID.to_numpy())
for j,l in enumerate(e.LABELS):out['pred_'+l]=hard[:,j];out['true_'+l]=y[:,j]
out['sample_f1']=2*(y*hard).sum(1)/(y.sum(1)+hard.sum(1))
out.to_csv('predictions/fullfit_validation_audit.csv',index=False)
manifest['validation_status']='Evaluated on all 78 validation rows at user request, with model settings and threshold unchanged.'
manifest['validation_metrics']={k:result[k] for k in ['sample_f1','macro_f1']+['f1_'+l for l in e.LABELS]}
manifest['validation_result']='experiments/final-validation.json'
e.save_json(manifest_path,manifest)
print(json.dumps(result,indent=2))
