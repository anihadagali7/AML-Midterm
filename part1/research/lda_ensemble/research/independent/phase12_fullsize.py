"""Leave-one-embedding-group-out selection for training-size-sensitive SSL.
No validation access until every full-size CV prediction and the shortlist are frozen.
"""
import phase11,experiment as e
import json,datetime,numpy as np,pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threadpoolctl import threadpool_limits
configs=[dict(model='lowdensity_v2',C=C,gamma=.001,pseudo_weight=.03) for C in [.4,.6,.8,1,1.5,2]]
configs += [dict(model='lowdensity_v2',C=1,gamma=g,pseudo_weight=.03) for g in [.0006,.002]]
configs += [dict(model='lowdensity_v2',C=1,gamma=.001,pseudo_weight=.03,cluster_gate=True,clusters=10)]
e.save_json('experiments/phase12-configs.json',configs)
groups=np.unique(e.GROUP[e.LAB]);positions=np.arange(len(e.Y))
def run(c):
 cid=e.config_id(c);path=Path(f'experiments/phase12-loo-{cid}.npz')
 if path.exists():return np.load(path,allow_pickle=True)['record'].item()
 p=np.zeros(e.Y.shape,float)
 for gi,g in enumerate(groups):
  b=np.flatnonzero(e.GROUP[e.LAB]==g);a=positions[~np.isin(positions,b)]
  p[b]=e.predict(a,e.DATA.iloc[e.LAB[b]],c,excluded_groups=[g])
  if gi%30==0:print('LEAVE-GROUP-OUT',cid,gi,len(groups),flush=True)
 rule,score=e.choose_rule(e.Y,p);record=dict(config=c,id=cid,rule=rule,cv=score,cv_metrics=e.metrics(e.Y,p,rule),n_groups=len(groups))
 np.savez_compressed(path,p=p,record=record);print('DONE',json.dumps(record),flush=True);return record
with threadpool_limits(limits=1):
 with ThreadPoolExecutor(max_workers=2) as pool:summary=list(pool.map(run,configs))
e.save_json('experiments/phase12-ranking.json',sorted(summary,key=lambda r:r['cv'],reverse=True))
# One sample-F1 champion and one macro-F1 champion. All decisions training-only.
cs={}
for r in [max(summary,key=lambda r:r['cv']),max(summary,key=lambda r:.7*r['cv']+.3*r['cv_metrics']['macro_f1'])]:cs[r['id']]=r
candidates=list(cs.values());e.save_json('experiments/phase12-validation-frozen.json',dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidates=candidates))
V=pd.read_csv('data/part1_val.csv',usecols=lambda col:col not in e.LABELS)
with threadpool_limits(limits=1):preds=[e.predict(positions,V,r['config']) for r in candidates]
np.savez_compressed('experiments/phase12-validation-predictions.npz',p=preds)
y=pd.read_csv('data/part1_val.csv')[e.LABELS].to_numpy();results=[]
for r,p in zip(candidates,preds):
 out={**r,'validation':e.metrics(y,p,r['rule'])};results.append(out);print('VALIDATION',json.dumps(out),flush=True)
e.save_json('experiments/phase12-validation-results.json',results)
