"""Nested shortlist audit and temporal stress test. Does NOT open validation or test labels."""
import experiment as e
import numpy as np,json,time
from pathlib import Path
from threadpoolctl import threadpool_limits
rank=json.loads(Path('experiments/robust-ranking.json').read_text())
ens=json.loads(Path('experiments/ensemble-ranking.json').read_text())
ssl=next(x['config'] for x in rank if x['config']['model']=='pseudo')
ssl_ensemble=next(x['config'] for x in ens if len(x['config']['members'])==3 and x['config']['members'][0]['model']=='svc')
candidates=[dict(model='logistic',C=.1,scale='global'),dict(model='logistic',C=.3,scale='global'),dict(model='svc',C=10,gamma=.001,scale='standard'),dict(model='svc',C=10,gamma=.001,scale='standard',features=['drug_targets'],feature_weight=1.),ssl,ens[0]['config'],ssl_ensemble]
e.save_json('experiments/nested-candidates.json',candidates)
outer=e.make_splits(seed=6172026,n_splits=5);pred=np.zeros_like(e.Y);prob=np.zeros_like(e.Y,float);records=[]
with threadpool_limits(limits=1):
 for fold,(a,b) in enumerate(outer):
  print('OUTER',fold,flush=True);inner=e.make_splits(e.Y[a],e.GROUP[e.LAB[a]],seed=1200+fold,n_splits=3)
  pool=[]
  for ci,c in enumerate(candidates):
   ip=np.zeros((len(a),5))
   for ia,ib in inner:
    exclude=np.r_[e.GROUP[e.LAB[b]],e.GROUP[e.LAB[a[ib]]]]
    ip[ib]=e.predict(a[ia],e.DATA.iloc[e.LAB[a[ib]]],c,excluded_groups=exclude)
   rule,score=e.choose_rule(e.Y[a],ip);pool.append(dict(config=c,rule=rule,score=score))
   print('INNER',fold,ci,score,rule,flush=True)
  # A small tie favors an earlier, simpler candidate.
  maximum=max(r['score'] for r in pool);best=next(r for r in pool if r['score']>=maximum-.003)
  pp=e.predict(a,e.DATA.iloc[e.LAB[b]],best['config'],excluded_groups=e.GROUP[e.LAB[b]])
  prob[b]=pp;pred[b]=e.decisions(pp,best['rule'])
  rec=dict(fold=fold,selected=best,inner=pool,outer=e.metrics(e.Y[b],pp,best['rule']))
  records.append(rec);e.save_json('experiments/nested-progress.json',records);print('OUTER RESULT',rec['outer'],flush=True)
 from sklearn.metrics import f1_score
 result=dict(sample_f1=f1_score(e.Y,pred,average='samples'),macro_f1=f1_score(e.Y,pred,average='macro'),per_label=dict(zip(e.LABELS,f1_score(e.Y,pred,average=None))),folds=records,limitation='Conditional shortlist audit. Candidate shortlist was informed by full-training exploratory CV; not an unbiased nested rerun of the complete adaptive search. External validation remains unopened.')
 e.save_json('experiments/nested-audit.json',result);np.savez_compressed('experiments/nested-predictions.npz',p=prob,pred=pred,y=e.Y)
 # Chronology tests are deliberately hard: most extra rare-reason labels occur in later years.
 temporal=[]
 for cutoff in [2002,2005]:
  years=e.DATA.iloc[e.LAB].start_year.to_numpy();a=np.flatnonzero(years<=cutoff);b=np.flatnonzero(years>cutoff)
  a=a[~np.isin(e.GROUP[e.LAB[a]],e.GROUP[e.LAB[b]])]
  for c in candidates:
   pp=e.predict(a,e.DATA.iloc[e.LAB[b]],c,excluded_groups=e.GROUP[e.LAB[b]])
   # Fixed default decision: no tuning on temporal holdout.
   rec=dict(cutoff=cutoff,config=c,n_train=len(a),n_test=len(b),train_positive=e.Y[a].sum(0).tolist(),test_positive=e.Y[b].sum(0).tolist(),**e.metrics(e.Y[b],pp,'t0.5'))
   temporal.append(rec);print('TEMPORAL',rec,flush=True)
 e.save_json('experiments/temporal-audit.json',temporal)
print('DONE',flush=True)
