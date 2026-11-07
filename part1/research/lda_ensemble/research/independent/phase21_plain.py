"""Classical majority-vote tri-training ablation without added cluster safeguards."""
import phase21 as tri,experiment as e
import numpy as np,pandas as pd,json,datetime
from pathlib import Path
from threadpoolctl import threadpool_limits
cs=[dict(model='tri_training',committee=group,cluster_gate=False,pseudo_weight=1.,aggregation='vote',stratified_bootstrap=False) for group in ['rbf','mixed','scales','kernels']]
e.save_json('experiments/phase21c-configs.json',cs)
rows=[]
for c in cs:
 r=e.evaluate(c,tag='phase21c');p=np.load(f"experiments/phase21c-{e.config_id(c)}-20261104.npz",allow_pickle=True)['p']
 rows.append(dict(config=c,metrics=e.metrics(e.Y,p,'t0.5')))
short={}
for key in ['sample_f1','macro_f1']:
 r=max(rows,key=lambda r:r['metrics'][key]);short[e.config_id(r['config'])]=r
summary=[]
for cid,r in short.items():
 ms=[]
 for seed in [20261107,20261108,20261109]:
  e.evaluate(r['config'],tag='phase21c_repeat',seed=seed)
  p=np.load(f'experiments/phase21c_repeat-{cid}-{seed}.npz',allow_pickle=True)['p'];ms.append(e.metrics(e.Y,p,'t0.5'))
 summary.append(dict(name='classical_tri_majority',config=r['config'],rule='t0.5',ordinary_cv=float(np.mean([m['sample_f1'] for m in ms])),cv_std=float(np.std([m['sample_f1'] for m in ms]))))
e.save_json('experiments/phase21c-selection.json',summary)
c=max(summary,key=lambda r:r['ordinary_cv']-.2*r['cv_std'])
e.save_json('experiments/phase21c-validation-frozen.json',dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose='Check whether conservative modifications suppressed classical tri-training: full-weight pseudo-labels, ordinary bootstrap, no cluster gate, and fixed majority decisions.',candidates=[c]))
V=pd.read_csv('data/part1_val.csv',usecols=lambda col:col not in e.LABELS)
with threadpool_limits(limits=1):p=e.predict(np.arange(len(e.Y)),V,c['config'])
np.savez_compressed('experiments/phase21c-validation-predictions.npz',p=p)
y=pd.read_csv('data/part1_val.csv')[e.LABELS].to_numpy();r={**c,'validation':e.metrics(y,p,c['rule'])}
e.save_json('experiments/phase21c-validation-results.json',[r]);print('VALIDATION',json.dumps(r),flush=True)
