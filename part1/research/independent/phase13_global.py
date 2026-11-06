"""Extend the global decision threshold range using training OOF predictions only."""
import phase11,experiment as e
import numpy as np,pandas as pd,json,datetime
from pathlib import Path
from threadpoolctl import threadpool_limits
# Prefer the tied validation winner with an explicit unsupervised clustering component.
c={'model':'lowdensity_v2','cluster_gate':True,'clusters':10,'C':1,'gamma':.001,'pseudo_weight':.03};cid=e.config_id(c)
ps=[]
for seed in [20261104,20261105,20261106]:
 tag='phase6' if seed==20261104 else 'phase6_repeat'
 path=Path(f'experiments/{tag}-{cid}-{seed}.npz')
 if not path.exists():e.evaluate(c,tag=tag,seed=seed)
 ps.append(np.load(path,allow_pickle=True)['p'])
# Same global threshold on all five labels; tie-breaking favors the default 0.5.
rules=['t'+str(t) for t in [.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8]]+['argmax','expected_f1']
p=np.mean(ps,axis=0);rows=[]
for rule in rules:
 ms=[e.metrics(e.Y,z,rule) for z in ps];rows.append(dict(rule=rule,cv_mean=float(np.mean([m['sample_f1'] for m in ms])),cv_macro=float(np.mean([m['macro_f1'] for m in ms])),averaged_oof=e.metrics(e.Y,p,rule)['sample_f1']))
r=max(rows,key=lambda r:(r['cv_mean'],r['averaged_oof'],-abs(float(r['rule'][1:])-.5) if r['rule'].startswith('t') else -1));candidate=dict(config=c,**r)
e.save_json('experiments/phase13-threshold-cv.json',rows)
e.save_json('experiments/phase13-validation-frozen.json',dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidates=[candidate]))
V=pd.read_csv('data/part1_val.csv',usecols=lambda col:col not in e.LABELS)
with threadpool_limits(limits=1):pred=e.predict(np.arange(len(e.Y)),V,c)
np.savez_compressed('experiments/phase13-validation-predictions.npz',p=pred)
y=pd.read_csv('data/part1_val.csv')[e.LABELS].to_numpy();result={**candidate,'validation':e.metrics(y,pred,r['rule'])}
e.save_json('experiments/phase13-validation-results.json',[result]);print(json.dumps(result,indent=2))
