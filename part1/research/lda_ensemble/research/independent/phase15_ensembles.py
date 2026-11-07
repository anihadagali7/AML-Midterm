"""Small-support convex ensembles, selected solely on repeated training OOF outputs."""
import phase14,experiment as e
import numpy as np,pandas as pd,json,datetime,itertools
from pathlib import Path
from threadpoolctl import threadpool_limits
base=dict(model='lowdensity_v2',cluster_gate=True,clusters=10,C=1,gamma=.001,pseudo_weight=.03)
rows=[json.loads(s) for s in Path('experiments/trials.jsonl').read_text().splitlines()];by={}
for r in rows:
 if r['status']=='ok' and r['seed'] in [20261104,20261105,20261106]:by.setdefault(r['id'],{})[r['seed']]=r
# One high-performing model per broad mechanism. Freeze candidates without reading validation results.
pool=[]
for family in ['logistic','svc','pseudo','margin_svm','lowdensity_svm','structured_ssl','subspace_ssl','kernel_lowdensity','calibrated_ssl','dense_graph']:
 candidates=[]
 for cid,seeds in by.items():
  if len(seeds)<3:continue
  r=seeds[20261104]
  if r['config']['model']!=family:continue
  ps=[]
  for seed,rr in sorted(seeds.items()):
   z=np.load(f"experiments/{rr['tag']}-{cid}-{seed}.npz",allow_pickle=True)['p'];ps.append(z)
  if any(z.shape!=(157,5) or not np.isfinite(z).all() for z in ps):continue
  p=np.mean(ps,axis=0);rule,_=e.choose_rule(e.Y,p);score=float(np.mean([e.metrics(e.Y,z,rule)['sample_f1'] for z in ps]));candidates.append((score,r['config'],ps))
 if candidates:pool.append(max(candidates,key=lambda x:x[0]))
cid=e.config_id(base);ps=[]
for seed in [20261104,20261105,20261106]:
 rr=by[cid][seed];ps.append(np.load(f"experiments/{rr['tag']}-{cid}-{seed}.npz",allow_pickle=True)['p'])
members=[base]+[x[1] for x in pool];preds=[ps]+[x[2] for x in pool]
configs=[]
for j in range(1,len(members)):
 for w in [.25,.5,.75]:configs.append(([0,j],[1-w,w]))
for j,k in itertools.combinations(range(1,len(members)),2):configs.append(([0,j,k],[.5,.25,.25]))
results=[]
for inds,weights in configs:
 pp=[sum(w*preds[i][seed] for i,w in zip(inds,weights)) for seed in range(3)];p=np.mean(pp,axis=0);rule,_=e.choose_rule(e.Y,p)
 scores=[e.metrics(e.Y,z,rule) for z in pp];r=dict(config=dict(model='ensemble',members=[members[i] for i in inds],weights=weights),rule=rule,cv_mean=float(np.mean([m['sample_f1'] for m in scores])),cv_macro=float(np.mean([m['macro_f1'] for m in scores])),cv_std=float(np.std([m['sample_f1'] for m in scores])))
 results.append(r)
e.save_json('experiments/phase15-ensemble-screen.json',results)
cs={}
for metric in ['sample','balanced']:
 r=max(results,key=lambda r:((r['cv_mean'] if metric=='sample' else .7*r['cv_mean']+.3*r['cv_macro'])-.2*r['cv_std']))
 cs[e.config_id(r['config'])]=r
candidates=list(cs.values());e.save_json('experiments/phase15-validation-frozen.json',dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidates=candidates))
V=pd.read_csv('data/part1_val.csv',usecols=lambda col:col not in e.LABELS)
with threadpool_limits(limits=1):p=[e.predict(np.arange(len(e.Y)),V,r['config']) for r in candidates]
np.savez_compressed('experiments/phase15-validation-predictions.npz',p=p)
y=pd.read_csv('data/part1_val.csv')[e.LABELS].to_numpy();out=[]
for r,z in zip(candidates,p):
 x={**r,'validation':e.metrics(y,z,r['rule'])};out.append(x);print('VALIDATION',json.dumps(x),flush=True)
e.save_json('experiments/phase15-validation-results.json',out)
