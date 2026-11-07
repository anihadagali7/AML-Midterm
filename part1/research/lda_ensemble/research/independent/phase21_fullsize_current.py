"""Fixed-policy leave-one-embedding-group-out audit for the new frozen hybrid."""
import phase21,experiment as e
import json,numpy as np
from pathlib import Path
from threadpoolctl import threadpool_limits
frozen=json.loads(Path('experiments/current-best.json').read_text());candidate=frozen['candidate'];c=candidate['config'];cid=e.config_id(c);groups=np.unique(e.GROUP[e.LAB]);allpos=np.arange(len(e.Y));p=np.zeros_like(e.Y,float)
with threadpool_limits(limits=1):
 for i,g in enumerate(groups):
  b=np.flatnonzero(e.GROUP[e.LAB]==g);a=allpos[~np.isin(allpos,b)]
  p[b]=e.predict(a,e.DATA.iloc[e.LAB[b]],c,excluded_groups=[g])
  if i%25==0:print('LEAVE-GROUP-OUT',i,len(groups),flush=True)
metrics=e.metrics(e.Y,p,candidate['rule']);record=dict(config=c,rule=candidate['rule'],metrics=metrics,n_groups=len(groups),interpretation='Development audit of the already selected fixed policy; not an unbiased rerun of the full search.')
np.savez_compressed(f'experiments/phase21-fullsize-{cid}.npz',p=p,record=record)
e.save_json('experiments/phase21-fullsize-current.json',record)
latest=json.loads(Path('experiments/current-best.json').read_text());assert e.config_id(latest['candidate']['config'])==cid
latest['leave_one_embedding_group_out_f1']=metrics['sample_f1'];e.save_json('experiments/current-best.json',latest);print(json.dumps(record,indent=2))
