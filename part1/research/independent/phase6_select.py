"""Training-only selection of low-density variants and label-specific specialists."""
import phase6,experiment as e
from pathlib import Path
from collections import defaultdict
import json,numpy as np,pandas as pd,datetime
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
rows=[json.loads(s) for s in Path('experiments/trials.jsonl').read_text().splitlines()];rs=[r for r in rows if r['status']=='ok' and r['tag'] in ['phase6','phase6_features']]
def family(c):
 if c.get('features'):return 'auxiliary_features'
 if c.get('pca'):return 'pca'
 if c.get('cluster_gate'):return 'cluster_gate'
 if c.get('balance_power') is not None:return 'class_balance'
 if c.get('confidence_weight'):return 'confidence_weight'
 return 'plain'
chosen={}
for fam in sorted({family(r['config']) for r in rs}):
 subset=[r for r in rs if family(r['config'])==fam]
 for r in [max(subset,key=lambda r:r['score']),max(subset,key=lambda r:r['macro_f1'])]:chosen[r['id']]=r
# Include the champions at a fixed threshold for each reason to explore independent settings.
for j,label in enumerate(e.LABELS):
 eligible=[]
 for r in rs:
  if r['score']<.85:continue
  p=np.load(f"experiments/{r['tag']}-{r['id']}-20261104.npz",allow_pickle=True)['p'];eligible.append((f1_score(e.Y[:,j],p[:,j]>=.5,zero_division=0),r))
 r=max(eligible,key=lambda x:x[0])[1];chosen[r['id']]=r
summary=[];probabilities={}
with threadpool_limits(limits=1):
 for r in chosen.values():
  c=r['config'];cid=r['id'];ps=[]
  for seed in [20261104,20261105,20261106]:
   tag=r['tag'] if seed==20261104 else 'phase6_repeat';rr=e.evaluate(c,seed=seed,tag=tag)
   ps.append(np.load(f'experiments/{tag}-{cid}-{seed}.npz',allow_pickle=True)['p'])
  p=np.mean(ps,axis=0);probabilities[cid]=p;rule,score=e.choose_rule(e.Y,p)
  f1s=np.array([[f1_score(e.Y[:,j],z[:,j]>=.5,zero_division=0) for j in range(5)] for z in ps])
  record=dict(id=cid,config=c,family=family(c),rule=rule,averaged_oof_sample=score,mean_cv_sample=float(np.mean([e.metrics(e.Y,z,rule)['sample_f1'] for z in ps])),mean_cv_macro=float(np.mean([e.metrics(e.Y,z,rule)['macro_f1'] for z in ps])),label_mean=f1s.mean(0).tolist(),label_std=f1s.std(0).tolist())
  summary.append(record);print('SUMMARY',json.dumps(record),flush=True)
e.save_json('experiments/phase6-selection.json',summary)
# Freeze five mechanism-diverse candidates. Rules come only from mean OOF predictions.
candidates=[]
for fam in ['plain','cluster_gate','pca','class_balance','auxiliary_features']:
 rr=max([r for r in summary if r['family']==fam],key=lambda r:.8*r['mean_cv_sample']+.2*r['mean_cv_macro'])
 candidates.append(dict(name=fam,config=rr['config'],rule=rr['rule'],cv=rr['averaged_oof_sample']))
# Labelwise specialist voting; each label uses top three stable candidates, avoiding one winner per eight-positive label.
bylabel=[];p=np.zeros_like(e.Y,float)
for j in range(5):
 rr=sorted(summary,key=lambda r:r['label_mean'][j]-.5*r['label_std'][j],reverse=True)[:3]
 bylabel.append([r['config'] for r in rr]);p[:,j]=np.mean([probabilities[r['id']][:,j] for r in rr],axis=0)
rule,score=e.choose_rule(e.Y,p);candidates.append(dict(name='labelwise_specialists',members_by_label=bylabel,rule=rule,cv=score))
e.save_json('experiments/phase6-validation-frozen.json',dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidates=candidates))
V=pd.read_csv('data/part1_val.csv',usecols=lambda c:c not in e.LABELS);cache={};preds=[]
with threadpool_limits(limits=1):
 def get(c):
  cid=e.config_id(c)
  if cid not in cache:cache[cid]=e.predict(np.arange(len(e.Y)),V,c)
  return cache[cid]
 for candidate in candidates:
  if 'config' in candidate:preds.append(get(candidate['config']))
  else:preds.append(np.column_stack([np.mean([get(m)[:,j] for m in ms],axis=0) for j,ms in enumerate(candidate['members_by_label'])]))
np.savez_compressed('experiments/phase6-validation-predictions.npz',p=preds)
y=pd.read_csv('data/part1_val.csv')[e.LABELS].to_numpy();results=[]
for c,p in zip(candidates,preds):
 r={**c,'validation':e.metrics(y,p,c['rule'])};results.append(r);print('VALIDATION',json.dumps(r),flush=True)
e.save_json('experiments/phase6-validation-results.json',results)
