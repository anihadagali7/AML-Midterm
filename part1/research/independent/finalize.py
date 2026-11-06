"""Fit frozen candidates, evaluate validation once, and generate Kaggle artifacts."""
import experiment_v1 as e
import pandas as pd,numpy as np,json,datetime,hashlib
from pathlib import Path
from threadpoolctl import threadpool_limits
frozen=json.loads(Path('experiments/frozen-shortlist.json').read_text())
assert hashlib.sha256(Path('experiment_v1.py').read_bytes()).hexdigest()==frozen['engine_sha256']
# Read only predictor columns until all candidate predictions are committed to disk.
val=pd.read_csv('data/part1_val.csv',usecols=lambda c:c not in e.LABELS)
test=pd.read_csv('data/part1_test.csv');unlab=e.DATA.iloc[e.UNLAB].drop(columns=e.LABELS)
query=pd.concat([val,test,unlab],ignore_index=True)
seeds=[20261104,20261105,20261106];member_predictions={}
with threadpool_limits(limits=1):
 for member in frozen['members']:
  cid=e.config_id(member);out=Path('predictions')/f'locked-member-{cid}.npz'
  if out.exists():
   z=np.load(out);assert np.array_equal(z['ids'],query.ID.to_numpy(str));member_predictions[cid]=z['p'];continue
  preds=[]
  for seed in seeds:
   for fold,(a,b) in enumerate(e.make_splits(seed=seed)):
    print('FIT',cid,seed,fold,flush=True)
    preds.append(e.predict(a,query,member))
  p=np.mean(preds,axis=0);np.savez_compressed(out,p=p,ids=query.ID.to_numpy(str),seed_predictions=np.array(preds));member_predictions[cid]=p
 for candidate in frozen['shortlist']:
  c=candidate['config'];weights=np.array(c['weights'],float);weights/=weights.sum()
  p=sum(w*member_predictions[e.config_id(m)] for w,m in zip(weights,c['members']))
  np.savez_compressed('predictions/locked-'+candidate['name']+'.npz',p=p,ids=query.ID.to_numpy(str))
# Now and only now use validation labels for the frozen comparison.
result_path=Path('experiments/validation-once.json')
if result_path.exists():raise SystemExit('Validation comparison already exists. Refusing to run another comparison.')
y=pd.read_csv('data/part1_val.csv',usecols=e.LABELS)[e.LABELS].to_numpy(int)
records=[]
for candidate in frozen['shortlist']:
 p=np.load('predictions/locked-'+candidate['name']+'.npz')['p'][:len(val)]
 r={**candidate,**e.metrics(y,p,candidate['rule'])};records.append(r);print('VALIDATION',candidate['name'],r['sample_f1'],r['macro_f1'],flush=True)
maximum=max(r['sample_f1'] for r in records)
priority={'fingerprint_ssl':0,'embedding_ssl':1,'balanced_ensemble':2}
winner=min([r for r in records if r['sample_f1']>=maximum-.005],key=lambda r:priority[r['name']])
p=np.load('predictions/locked-'+winner['name']+'.npz')['p'];hard=e.decisions(p,winner['rule'])
# Cluster bootstrap by exact embedding group, keeping duplicate reasons together.
vg=pd.util.hash_pandas_object(val[e.EMB].round(10),index=False).to_numpy();groups=np.unique(vg)
row_f1=2*(y*hard[:len(val)]).sum(1)/(y.sum(1)+hard[:len(val)].sum(1))
rng=np.random.default_rng(8102026);boot=[]
for _ in range(10000):
 sampled=rng.choice(groups,len(groups),replace=True);idx=np.concatenate([np.flatnonzero(vg==g) for g in sampled]);boot.append(float(row_f1[idx].mean()))
ci=np.quantile(boot,[.025,.975]).tolist()
result=dict(evaluated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),validation_rows=len(val),candidates=records,selected=winner['name'],selected_sample_f1=winner['sample_f1'],selected_macro_f1=winner['macro_f1'],bootstrap_95_interval=ci,limitation='Single, frozen 3-candidate selection on 78 validation trials; the selected validation score has selection optimism. This is not a Kaggle score. No validation refit or error-specific tuning.',user_kaggle_baseline=.89423,user_kaggle_leader=.92735)
e.save_json(result_path,result)
# Save every frozen submission for reproducibility; main filename always points to selected candidate.
for candidate in frozen['shortlist']:
 pp=np.load('predictions/locked-'+candidate['name']+'.npz')['p'];hh=e.decisions(pp,candidate['rule'])
 sub=pd.DataFrame({'ID':test.ID,'Predicted':[' '.join(l for l,x in zip(e.LABELS,row) if x) for row in hh[len(val):len(val)+len(test)]]})
 sub.to_csv('predictions/submission_'+candidate['name']+'.csv',index=False)
 if candidate['name']==winner['name']:sub.to_csv('predictions/part1_submission.csv',index=False)
for name,frame,sl in [('validation',val,slice(0,len(val))),('test',test,slice(len(val),len(val)+len(test))),('unlabeled_train',unlab,slice(len(val)+len(test),None))]:
 out=pd.DataFrame(p[sl],columns=['p_'+l for l in e.LABELS]);out.insert(0,'ID',frame.ID.to_numpy());out.to_csv('predictions/'+name+'_probabilities.csv',index=False)
aug=e.DATA.copy();aug.loc[e.UNLAB,e.LABELS]=hard[len(val)+len(test):];aug['label_source']='original';aug.loc[e.UNLAB,'label_source']='pseudo_label_'+winner['name'];aug.to_csv('predictions/part1_train_augmented.csv',index=False)
# Validation outputs for report only; do not feed these back into model selection.
v=pd.DataFrame(hard[:len(val)],columns=['pred_'+l for l in e.LABELS]);v.insert(0,'ID',val.ID.to_numpy());v['sample_f1']=row_f1
for j,l in enumerate(e.LABELS):v['true_'+l]=y[:,j]
v.to_csv('predictions/validation_audit.csv',index=False)
sample=pd.read_csv('data/part1_sample_submission.csv');submission=pd.read_csv('predictions/part1_submission.csv')
assert list(submission.columns)==['ID','Predicted'] and len(submission)==312
assert submission.ID.tolist()==sample.ID.tolist() and submission.ID.is_unique
assert submission.Predicted.notna().all() and submission.Predicted.map(lambda s:bool(s) and set(s.split())<=set(e.LABELS)).all()
assert aug[e.LABELS].isin([0,1]).all().all() and np.array_equal(aug.loc[e.LAB,e.LABELS],e.Y)
e.save_json('experiments/final-checks.json',dict(submission_rows=312,submission_id_order_matches_sample=True,labels_valid=True,augmented_rows=len(aug),preserved_ground_truth_rows=len(e.LAB),generated_label_rows=len(e.UNLAB),submission_sha256=hashlib.sha256(Path('predictions/part1_submission.csv').read_bytes()).hexdigest()))
print('SELECTED',winner['name'],'SAMPLE F1',winner['sample_f1'],'95% interval',ci,flush=True)
