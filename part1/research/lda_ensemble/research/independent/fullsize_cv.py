"""Nearly-full-training-size audit using leave-one-embedding-group-out CV. No validation access."""
import experiment as e,numpy as np,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threadpoolctl import threadpool_limits
frozen=json.loads(Path('experiments/frozen-shortlist.json').read_text());members=frozen['members']
# The selection candidates are unchanged in structure; this audit addresses training-size and decoding.
unique=np.unique(e.GROUP[e.LAB]);allpos=np.arange(len(e.Y))
def run_member(c):
 cid=e.config_id(c);out=Path('experiments')/('fullsize-'+cid+'.npz')
 if out.exists():return
 p=np.zeros_like(e.Y,float)
 for gi,g in enumerate(unique):
  b=np.flatnonzero(e.GROUP[e.LAB]==g);a=allpos[~np.isin(allpos,b)]
  p[b]=e.predict(a,e.DATA.iloc[e.LAB[b]],c,excluded_groups=[g])
  if gi%25==0:print('LEAVE-GROUP-OUT',cid,gi,len(unique),flush=True)
 rule,score=e.choose_rule(e.Y,p)
 e.save_json('experiments/fullsize-'+cid+'.json',dict(config=c,rule=rule,score=score,n_groups=len(unique),mean_training_size=float(np.mean([len(e.Y)-np.sum(e.GROUP[e.LAB]==g) for g in unique]))))
 np.savez_compressed(out,p=p);print('DONE',cid,score,rule,flush=True)
with threadpool_limits(limits=1):
 with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run_member,members))
results=[]
for candidate in frozen['shortlist']:
 c=candidate['config'];w=np.array(c['weights'],float);w/=w.sum()
 p=sum(wi*np.load('experiments/fullsize-'+e.config_id(m)+'.npz')['p'] for wi,m in zip(w,c['members']))
 rule,score=e.choose_rule(e.Y,p);r=dict(name=candidate['name'],config=c,rule=rule,fullsize_oof_f1=score,metrics=e.metrics(e.Y,p,rule));results.append(r)
 np.savez_compressed('experiments/fullsize-oof-'+candidate['name']+'.npz',p=p,y=e.Y)
e.save_json('experiments/fullsize-ranking.json',sorted(results,key=lambda r:r['fullsize_oof_f1'],reverse=True))
print('RESULTS',results,flush=True)
