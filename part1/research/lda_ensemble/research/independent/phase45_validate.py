import phase5,experiment as e
import json,numpy as np,pandas as pd,datetime
from pathlib import Path
from threadpoolctl import threadpool_limits
rows=[json.loads(s) for s in Path('experiments/trials.jsonl').read_text().splitlines()]
rows=[r for r in rows if r['status']=='ok' and r['tag'] in ['phase4','phase5']]
def family(r):return r['model']+'_'+r['config']['mix'] if r['model']=='composition' else r['model']
chosen=[]
with threadpool_limits(limits=1):
 for fam in sorted(set(family(r) for r in rows)):
  rr=[r for r in rows if family(r)==fam];short=list({r['id']:r for r in [max(rr,key=lambda r:r['score']),max(rr,key=lambda r:r['macro_f1'])]}.values());summaries=[]
  for r in short:
   c=r['config'];ps=[]
   for seed in [20261104,20261105,20261106]:
    tag=r['tag'] if seed==20261104 else 'phase45_repeat';res=e.evaluate(c,seed=seed,tag=tag);ps.append(np.load(f"experiments/{tag}-{r['id']}-{seed}.npz",allow_pickle=True)['p'])
   avg=np.mean(ps,axis=0);rule,score=e.choose_rule(e.Y,avg);sample=np.mean([e.metrics(e.Y,p,rule)['sample_f1'] for p in ps]);macro=np.mean([e.metrics(e.Y,p,rule)['macro_f1'] for p in ps])
   record=dict(family=fam,config=c,id=r['id'],rule=rule,averaged_oof_sample=score,mean_cv_sample=sample,mean_cv_macro=macro,selection=.7*sample+.3*macro);summaries.append(record)
  chosen.append(max(summaries,key=lambda r:r['selection']))
e.save_json('experiments/phase45-validation-frozen.json',dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidates=chosen))
V=pd.read_csv('data/part1_val.csv',usecols=lambda c:c not in e.LABELS)
with threadpool_limits(limits=1):ps=[e.predict(np.arange(len(e.Y)),V,r['config']) for r in chosen]
np.savez_compressed('experiments/phase45-validation-predictions.npz',p=ps)
y=pd.read_csv('data/part1_val.csv')[e.LABELS].to_numpy();results=[]
for r,p in zip(chosen,ps):
 result={**r,'validation':e.metrics(y,p,r['rule'])};results.append(result);print('VALIDATION',json.dumps(result),flush=True)
e.save_json('experiments/phase45-validation-results.json',results)
