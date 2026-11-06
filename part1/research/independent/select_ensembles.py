import experiment as e,json,numpy as np
from pathlib import Path
from collections import defaultdict
by=defaultdict(dict)
for f in Path('experiments').glob('*.npz'):
 z=np.load(f,allow_pickle=True);r=z['record'].item()
 if r['tag'] in ['screen','refine','features','consensus','robustness','combinations']:by[r['id']][r['seed']]=(r,z['p'])
rank=json.loads(Path('experiments/robust-ranking.json').read_text())
# Broad mechanisms rather than many near-identical hyperparameter fits.
proto=[
 dict(model='logistic',scale='global',C=.2),
 dict(model='logistic',scale='global',C=.3),
 dict(model='svc',C=10,gamma=.001,scale='standard',features=['drug_targets'],feature_weight=1.),
 dict(model='svc',C=10,gamma=.001,scale='standard',features=['fingerprint'],feature_weight=.5),
 dict(model='svc',C=10,gamma=.001,scale='standard',features=['criteria'],feature_weight=1.),
 next(x['config'] for x in rank if x['config']['model']=='pseudo'),
 next(x['config'] for x in rank if x['config']['model']=='pseudo' and x['config']['teacher']['model']=='graph'),
]
ensembles=[]
for j in [2,3,4,5,6]:
 for w in [.25,.5,.75]:ensembles.append(dict(model='ensemble',members=[proto[0],proto[j]],weights=[1-w,w]))
for ids in [[0,2,5],[0,3,5],[0,2,4,5],[0,2,3,4,5],[0,5,6],[2,4,5],[0,1,2,3,4,5,6]]:
 ensembles.append(dict(model='ensemble',members=[proto[j] for j in ids]))
results=[]
for c in ensembles:
 ids=[e.config_id(m) for m in c['members']];seeds=sorted(set.intersection(*(set(by[cid]) for cid in ids)))
 ps=[];weights=np.asarray(c.get('weights',np.ones(len(ids))),float);weights/=weights.sum()
 for seed in seeds:ps.append(sum(w*by[cid][seed][1] for cid,w in zip(ids,weights)))
 p=np.mean(ps,axis=0);rule,score=e.choose_rule(e.Y,p)
 r=dict(config=c,id=e.config_id(c),rule=rule,averaged_oof_score=score,mean_split_score=float(np.mean([e.metrics(e.Y,x,rule)['sample_f1'] for x in ps])),n_seeds=len(seeds))
 results.append(r)
 np.savez_compressed('experiments/ensemble-'+r['id']+'.npz',p=p,record=r)
results.sort(key=lambda r:(r['averaged_oof_score'],r['mean_split_score']),reverse=True)
e.save_json('experiments/ensemble-ranking.json',results)
for r in results[:8]:print(r['averaged_oof_score'],r['mean_split_score'],r['rule'],r['config'])
# Direct SSL candidate with feature additions, evaluated across fresh seeds.
base=proto[5];configs=[]
for block in ['drug_targets','drugs','criteria','fingerprint','pathway_levels']:
 for w in [.15,.5]:configs.append({**base,'features':[block],'feature_weight':w})
e.save_json('experiments/ssl-features-input.json',configs)
