"""Repeated grouped CV selection and frozen validation for a specified search phase."""
import phase20,experiment as e
import json,datetime,numpy as np,pandas as pd,argparse
from pathlib import Path
from threadpoolctl import threadpool_limits
parser=argparse.ArgumentParser();parser.add_argument('tag');args=parser.parse_args();tag=args.tag
rs=[json.loads(s) for s in Path('experiments/trials.jsonl').read_text().splitlines()];rs=[r for r in rs if r['tag']==tag and r['status']=='ok']
def family(c):
 if tag=='phase8':return c['model']
 if tag=='phase9':
  if c.get('white'):return 'whitening'
  if c.get('kernel','rbf')!='rbf':return 'non_rbf_kernel'
  return 'embedding_geometry'
 if tag=='phase10':return 'structured_cluster' if c.get('cluster_gate') else 'structured_joint'
 if tag=='phase11':return 'random_subspace' if c.get('members',1)>1 else ('prior_constraint' if c.get('constrain_prior') else 'confidence_filter')
 if tag=='phase14':return 'label_thresholds' if c['calibration']=='thresholds' else ('label_patterns' if c['calibration']=='patterns' else 'margin_calibration')
 if tag=='phase16':return 'alternative_loss'
 if tag=='phase17':return c['student']
 if tag=='phase18':return 'time_weights' if c['weighting'] in ['year','late'] else 'distribution_weights'
 if tag=='phase19':return 'size_matched_thresholds' if c['calibration']=='thresholds' else 'size_matched_calibration'
 if tag=='phase20':return 'separate_feature_kernels' if c.get('aux_blocks') else ('repeated_vector_weighting' if c.get('deduplicate') else 'effective_class_balance')
 raise ValueError(tag)
chosen={}
for fam in sorted({family(r['config']) for r in rs}):
 rows=[r for r in rs if family(r['config'])==fam]
 for r in [max(rows,key=lambda r:r['score']),max(rows,key=lambda r:r['macro_f1'])]:chosen[r['id']]=r
summary=[]
with threadpool_limits(limits=1):
 for r in chosen.values():
  ps=[]
  for seed in [20261104,20261105,20261106]:
   t=tag if seed==20261104 else tag+'_repeat';e.evaluate(r['config'],seed=seed,tag=t)
   ps.append(np.load(f"experiments/{t}-{r['id']}-{seed}.npz",allow_pickle=True)['p'])
  p=np.mean(ps,axis=0);rule,score=e.choose_rule(e.Y,p)
  rr=dict(name=family(r['config']),config=r['config'],rule=rule,cv=score,cv_mean=float(np.mean([e.metrics(e.Y,z,rule)['sample_f1'] for z in ps])),cv_macro=float(np.mean([e.metrics(e.Y,z,rule)['macro_f1'] for z in ps])))
  summary.append(rr);print('SUMMARY',json.dumps(rr),flush=True)
e.save_json(f'experiments/{tag}-selection.json',summary)
candidates=[max([r for r in summary if r['name']==fam],key=lambda r:.8*r['cv_mean']+.2*r['cv_macro']) for fam in sorted({r['name'] for r in summary})]
e.save_json(f'experiments/{tag}-validation-frozen.json',dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidates=candidates))
V=pd.read_csv('data/part1_val.csv',usecols=lambda c:c not in e.LABELS)
with threadpool_limits(limits=1):preds=[e.predict(np.arange(len(e.Y)),V,c['config']) for c in candidates]
np.savez_compressed(f'experiments/{tag}-validation-predictions.npz',p=preds)
y=pd.read_csv('data/part1_val.csv')[e.LABELS].to_numpy();results=[]
for c,p in zip(candidates,preds):
 r={**c,'validation':e.metrics(y,p,c['rule'])};results.append(r);print('VALIDATION',json.dumps(r),flush=True)
e.save_json(f'experiments/{tag}-validation-results.json',results)
