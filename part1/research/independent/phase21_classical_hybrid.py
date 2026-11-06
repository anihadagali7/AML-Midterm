"""Apply the preregistered reason-combination rule unchanged to classical tri-training."""
import phase21 as tri,experiment as e
import numpy as np,pandas as pd,json,datetime
from pathlib import Path
from threadpoolctl import threadpool_limits
chosen=json.loads(Path('experiments/phase21c-validation-frozen.json').read_text())['candidates'][0]
base=json.loads(Path('experiments/phase21-prior-incumbent.json').read_text())['candidate']
weights=dict(np.load('experiments/phase21-weights.npz'));names=['propensity_logistic','propensity_tree','kmm'];w=np.array([weights[k]/weights[k].sum() for k in names])
ps=[];baseline=[]
for seed in [20261107,20261108,20261109]:
 bp=np.load(f"experiments/phase21_refit-{e.config_id(base['config'])}-{seed}.npz",allow_pickle=True)['p']
 tp=np.load(f"experiments/phase21c_repeat-{e.config_id(chosen['config'])}-{seed}.npz",allow_pickle=True)['p']
 hb=e.decisions(bp,base['rule']);ht=e.decisions(tp,chosen['rule']);ps.append((hb,ht))
 row=2*(hb*e.Y).sum(1)/(hb.sum(1)+e.Y.sum(1));baseline.append(float(row.mean()))
records=[]
for number in range(32):
 mask=((number>>np.arange(5))&1).astype(bool);weighted=[];ordinary=[]
 for hb,ht in ps:
  h=np.where(mask[None,:],ht,hb);empty=h.sum(1)==0;h[empty]=hb[empty]
  row=2*(h*e.Y).sum(1)/(h.sum(1)+e.Y.sum(1));ordinary.append(float(row.mean()));weighted.append(w@row)
 ordinary=np.array(ordinary);weighted=np.array(weighted);robust=.75*weighted.min(1)+.25*ordinary
 eligible=bool(np.all(ordinary>=np.array(baseline)-.005));score=float(robust.mean()-.2*robust.std()-.001*mask.sum())
 records.append(dict(use_tri=mask.astype(int).tolist(),ordinary_cv=float(ordinary.mean()),weighted_cv=dict(zip(names,weighted.mean(0).tolist())),selection_score=score,eligible=eligible))
e.save_json('experiments/phase21d-hybrid-screen.json',records)
best=max([r for r in records if r['eligible']],key=lambda r:r['selection_score'])
candidates=[]
if any(best['use_tri']):
 config=dict(model='tri_labelwise',base=base['config'],tri=chosen['config'],base_rule=base['rule'],tri_rule=chosen['rule'],use_tri=best['use_tri'])
 candidates.append(dict(name='classical_tri_reason_hybrid',config=config,rule='t0.5',**best))
e.save_json('experiments/phase21d-validation-frozen.json',dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),hybrid_selection=best,candidates=candidates))
print('SELECTED',json.dumps(best),flush=True)
if candidates:
 V=pd.read_csv('data/part1_val.csv',usecols=lambda col:col not in e.LABELS)
 with threadpool_limits(limits=1):preds=[e.predict(np.arange(len(e.Y)),V,r['config']) for r in candidates]
 np.savez_compressed('experiments/phase21d-validation-predictions.npz',p=preds)
 y=pd.read_csv('data/part1_val.csv')[e.LABELS].to_numpy();results=[]
 for r,p in zip(candidates,preds):
  out={**r,'validation':e.metrics(y,p,r['rule'])};results.append(out);print('VALIDATION',json.dumps(out),flush=True)
else:results=[]
e.save_json('experiments/phase21d-validation-results.json',results)
