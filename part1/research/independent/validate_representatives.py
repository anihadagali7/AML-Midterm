import experiment as e,json,numpy as np,pandas as pd,datetime
from pathlib import Path
from threadpoolctl import threadpool_limits
out=Path('experiments/phase3-reference-validation.json')
if out.exists():raise SystemExit('Already evaluated')
rows=[json.loads(s) for s in Path('experiments/trials.jsonl').read_text().splitlines()]
models=['logistic','svc','knn','cluster','graph','softmax','cluster_features','catboost']
chosen=[]
for model in models:
 eligible=[r for r in rows if r['status']=='ok' and r['config']['model']==model and not r['config'].get('features') and r['seed']==20261104]
 r=max(eligible,key=lambda r:r['score'])
 chosen.append({'config':r['config'],'rule':r['rule'],'cv_score':r['score'],'model':model})
e.save_json('experiments/phase3-reference-frozen.json',chosen)
V=pd.read_csv('data/part1_val.csv',usecols=lambda c:c not in e.LABELS);preds=[]
with threadpool_limits(limits=1):
 for r in chosen:
  p=e.predict(np.arange(len(e.Y)),V,r['config']);preds.append(p);print('PREDICTED',r['model'],flush=True)
np.savez_compressed('experiments/phase3-reference-predictions.npz',p=np.array(preds))
y=pd.read_csv('data/part1_val.csv')[e.LABELS].to_numpy()
results=[]
for r,p in zip(chosen,preds):
 result={**r,**e.metrics(y,p,r['rule'])};results.append(result);print(json.dumps(result),flush=True)
e.save_json(out,dict(evaluated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),results=results,protocol='Frozen configurations and decision rules chosen using training CV; one representative per mechanism, no validation retuning.'))
