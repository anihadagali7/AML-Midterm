"""Fresh-seed checks and predeclared feature combinations; no external validation."""
import json
from pathlib import Path
import experiment as e
rs=[json.loads(s) for s in Path('experiments/trials.jsonl').read_text().splitlines()]
ok=[r for r in rs if r['status']=='ok' and r['tag'] in ['screen','refine','features','consensus']]
chosen=[]
for model in ['svc','logistic','lda','chain','powerset','graph','cluster','pseudo']:
 matches=[r for r in ok if r['model']==model and not r['config'].get('features')]
 chosen.extend(sorted(matches,key=lambda x:x['score'],reverse=True)[:(3 if model in ['svc','logistic','pseudo'] else 1)])
for block in ['drugs','pathway_levels','pathway_level0','descriptors','fingerprint','criteria','drug_pathways','drug_targets','year']:
 matches=[r for r in ok if r['config'].get('features')==[block]]
 chosen.append(max(matches,key=lambda x:x['score']))
configs=list({e.config_id(r['config']):r['config'] for r in chosen}.values())
e.save_json('experiments/robustness-input.json',configs)
for seed in [20261105,20261106,20261107]:
 for i,c in enumerate(configs):
  print('REPEAT',seed,i+1,len(configs),c,flush=True);e.evaluate(c,seed=seed,tag='robustness')
base=dict(model='svc',C=10,gamma=.001,scale='standard')
combos=[['drugs','pathway_levels'],['drugs','descriptors'],['drugs','pathway_level0'],['drugs','criteria'],['pathway_levels','descriptors'],['drugs','pathway_levels','descriptors'],['drugs','pathway_levels','criteria'],['drugs','pathway_levels','descriptors','criteria'],['criteria','diseases_cleaned','drugs','icdcodes','matched_drugs','drug_targets','drug_actions','drug_pathways','fingerprint','descriptors','year','missingness','counts']]
configs=[{**base,'features':b,'feature_weight':w} for b in combos for w in [.15,.5]]
e.save_json('experiments/combinations-input.json',configs)
for seed in [20261104,20261105,20261106]:
 for c in configs:e.evaluate(c,seed=seed,tag='combinations')
