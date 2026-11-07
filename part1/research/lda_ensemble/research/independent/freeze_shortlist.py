import json, numpy as np, hashlib, datetime
from pathlib import Path
import experiment as e
path=Path('experiments/frozen-shortlist.json')
if path.exists():raise SystemExit('Shortlist already frozen; refusing overwrite.')
fp=json.loads(Path('experiments/ssl_features-ranking.json').read_text())[0]['config']
combo=json.loads(Path('experiments/combinations-ranking.json').read_text())[0]['config']
target=dict(model='svc',C=10,gamma=.001,scale='standard',features=['drug_targets'],feature_weight=1.)
plain=next(x['config'] for x in json.loads(Path('experiments/robust-ranking.json').read_text()) if x['config']['model']=='pseudo')
models=[fp,combo,target,plain];cache={}
for f in Path('experiments').glob('*.npz'):
 z=np.load(f,allow_pickle=True)
 if 'record' not in z:continue
 r=z['record'].item()
 if 'seed' in r:cache[(r['id'],r['seed'])]=z['p']
shortlist=[]
for name,weights in [('fingerprint_ssl',[1,0,0,0]),('balanced_ensemble',[.5,.25,.25,0]),('embedding_ssl',[0,0,0,1])]:
 ps=[]
 for seed in [20261104,20261105,20261106]:ps.append(sum(w*cache[(e.config_id(m),seed)] for w,m in zip(weights,models) if w))
 p=np.mean(ps,axis=0);rule,score=e.choose_rule(e.Y,p)
 c=dict(model='ensemble',members=[m for m,w in zip(models,weights) if w],weights=[w for w in weights if w])
 rec=dict(name=name,config=c,rule=rule,repeated_oof_f1=score,mean_split_f1=float(np.mean([e.metrics(e.Y,z,rule)['sample_f1'] for z in ps])),prediction_strategy='Mean of 15 models: 5 grouped folds x seeds 20261104,20261105,20261106. No validation labels used in fits.')
 shortlist.append(rec);np.savez_compressed('experiments/frozen-oof-'+name+'.npz',p=p,y=e.Y);print(rec,flush=True)
obj=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),members=models,shortlist=shortlist,validation_selection='Choose the highest sample-F1 among these three frozen models; if tied within 0.005, choose fingerprint_ssl, then embedding_ssl, then balanced_ensemble. No subsequent validation-error-driven tuning.',training_data='Official training file only; validation remains held out even for final submission.',engine_sha256=hashlib.sha256(Path('experiment.py').read_bytes()).hexdigest())
e.save_json(path,obj)
