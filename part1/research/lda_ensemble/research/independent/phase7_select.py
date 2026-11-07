import phase7,experiment as e
import json,datetime,numpy as np,pandas as pd
from pathlib import Path
from threadpoolctl import threadpool_limits
rs=[json.loads(s) for s in Path('experiments/trials.jsonl').read_text().splitlines()];rs=[r for r in rs if r['tag']=='phase7' and r['status']=='ok']
def family(c):return ('transductive' if c.get('transductive') else 'inductive')+('_cluster' if c.get('cluster_gate') else '')
chosen={}
for fam in sorted({family(r['config']) for r in rs}):
 rows=[r for r in rs if family(r['config'])==fam]
 for r in [max(rows,key=lambda r:r['score']),max(rows,key=lambda r:r['macro_f1'])]:chosen[r['id']]=r
summary=[]
with threadpool_limits(limits=1):
 for r in chosen.values():
  ps=[]
  for seed in [20261104,20261105,20261106]:
   tag='phase7' if seed==20261104 else 'phase7_repeat';e.evaluate(r['config'],seed=seed,tag=tag)
   ps.append(np.load(f"experiments/{tag}-{r['id']}-{seed}.npz",allow_pickle=True)['p'])
  p=np.mean(ps,axis=0);rule,score=e.choose_rule(e.Y,p)
  rr=dict(name=family(r['config']),config=r['config'],rule=rule,cv=score,cv_mean=float(np.mean([e.metrics(e.Y,z,rule)['sample_f1'] for z in ps])),cv_macro=float(np.mean([e.metrics(e.Y,z,rule)['macro_f1'] for z in ps])))
  summary.append(rr);print('SUMMARY',json.dumps(rr),flush=True)
e.save_json('experiments/phase7-selection.json',summary)
candidates=[max([r for r in summary if r['name']==fam],key=lambda r:.8*r['cv_mean']+.2*r['cv_macro']) for fam in sorted({r['name'] for r in summary})]
e.save_json('experiments/phase7-validation-frozen.json',dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidates=candidates))
V=pd.read_csv('data/part1_val.csv',usecols=lambda c:c not in e.LABELS)
with threadpool_limits(limits=1):preds=[e.predict(np.arange(len(e.Y)),V,c['config']) for c in candidates]
np.savez_compressed('experiments/phase7-validation-predictions.npz',p=preds)
y=pd.read_csv('data/part1_val.csv')[e.LABELS].to_numpy();results=[]
for c,p in zip(candidates,preds):
 r={**c,'validation':e.metrics(y,p,c['rule'])};results.append(r);print('VALIDATION',json.dumps(r),flush=True)
e.save_json('experiments/phase7-validation-results.json',results)
