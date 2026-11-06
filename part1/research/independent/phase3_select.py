"""Repeat training CV, temporal audits, then freeze a small family-diverse validation batch."""
import phase3 as p3
import experiment as e
import numpy as np,pandas as pd,json,datetime
from pathlib import Path
from threadpoolctl import threadpool_limits
rows=[json.loads(s) for s in Path('experiments/trials.jsonl').read_text().splitlines()]
rs=[r for r in rows if r['tag']=='phase3' and r['status']=='ok'];models=sorted(set(r['model'] for r in rs));candidates=[]
for model in models:
 matches=[r for r in rs if r['model']==model]
 candidates.extend([max(matches,key=lambda r:r['score']),max(matches,key=lambda r:r['macro_f1'])])
candidates=list({r['id']:r for r in candidates}.values())
summary=[]
with threadpool_limits(limits=1):
 for base in candidates:
  c=base['config'];cid=base['id'];preds=[]
  for seed in [20261104,20261105,20261106]:
   tag='phase3' if seed==20261104 else 'phase3_repeat'
   r=e.evaluate(c,seed=seed,tag=tag);preds.append(np.load(f'experiments/{tag}-{cid}-{seed}.npz',allow_pickle=True)['p'])
  mean=np.mean(preds,axis=0);rule,score=e.choose_rule(e.Y,mean)
  avg=np.mean([e.metrics(e.Y,x,rule)['sample_f1'] for x in preds]);macro=np.mean([e.metrics(e.Y,x,rule)['macro_f1'] for x in preds])
  times=[]
  for cutoff in [2002,2005]:
   yr=e.DATA.iloc[e.LAB].start_year.to_numpy();a=np.flatnonzero(yr<=cutoff);b=np.flatnonzero(yr>cutoff)
   a=a[~np.isin(e.GROUP[e.LAB[a]],e.GROUP[e.LAB[b]])]
   pp=e.predict(a,e.DATA.iloc[e.LAB[b]],c,excluded_groups=e.GROUP[e.LAB[b]])
   times.append(e.metrics(e.Y[b],pp,'t0.5'))
  temporal=np.mean([x['sample_f1'] for x in times])
  record=dict(config=c,id=cid,rule=rule,mean_cv_sample=avg,mean_cv_macro=macro,averaged_oof_sample=score,temporal=times,selection_score=.45*avg+.20*macro+.35*temporal)
  summary.append(record);np.savez_compressed('experiments/phase3-mean-'+cid+'.npz',p=mean)
  print('SUMMARY',json.dumps(record),flush=True)
e.save_json('experiments/phase3-selection.json',summary)
chosen=[]
for model in models:chosen.append(max([x for x in summary if x['config']['model']==model],key=lambda x:x['selection_score']))
# One training-only calibrated threshold alternative for the strongest sample-F1 candidate.
from sklearn.metrics import f1_score
best=max(summary,key=lambda x:x['mean_cv_sample']);P=np.load('experiments/phase3-mean-'+best['id']+'.npz')['p'];thresholds=[]
for j in range(5):
 grid=np.arange(.1,.651,.05);fs=np.array([f1_score(e.Y[:,j],P[:,j]>=t,zero_division=0) for t in grid]);optimal=float(np.median(grid[fs>=fs.max()-1e-12]))
 # Shrink rare-label cutoffs toward 0.35; never use validation labels.
 weight=e.Y[:,j].sum()/(e.Y[:,j].sum()+10);thresholds.append(float(weight*optimal+(1-weight)*.35))
chosen.append({**best,'thresholds':thresholds,'rule':'per_label_training_f1_shrunk'})
e.save_json('experiments/phase3-validation-frozen.json',dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection_criterion='0.45 repeated grouped CV sample F1 + 0.20 macro F1 + 0.35 fixed-threshold temporal sample F1; one representative per family, plus one training-only per-label calibration alternative.',candidates=chosen))
V=pd.read_csv('data/part1_val.csv',usecols=lambda c:c not in e.LABELS)
with threadpool_limits(limits=1):predictions=[e.predict(np.arange(len(e.Y)),V,c['config']) for c in chosen]
np.savez_compressed('experiments/phase3-validation-predictions.npz',p=predictions)
y=pd.read_csv('data/part1_val.csv')[e.LABELS].to_numpy();results=[]
for candidate,pp in zip(chosen,predictions):
 if 'thresholds' in candidate:
  hh=(pp>=candidate['thresholds']).astype(int);empty=hh.sum(1)==0;hh[np.flatnonzero(empty),pp[empty].argmax(1)]=1
 else:hh=e.decisions(pp,candidate['rule'])
 result={**candidate,'validation_sample_f1':float(f1_score(y,hh,average='samples')),'validation_macro_f1':float(f1_score(y,hh,average='macro'))}
 results.append(result);print('VALIDATION',json.dumps(result),flush=True)
e.save_json('experiments/phase3-validation-results.json',results)
