"""Targeted class-prior correction; prompted by v3's rare-label failure, logged as development."""
import sys,json,time,types
from pathlib import Path
import numpy as np
import study as e
import models,models_v3
BASE=Path(__file__).resolve().parent;OUT=BASE/'studies/v4'
def configs():
    return [dict(name=f'new_blend_prior_p{power:g}',family='prior_corrected_blend',power=power) for power in [.25,.5]]
def predict(gold,masked,query,c,labels,emb):
    p,_,audit=models_v3.predict(gold,masked,query,dict(family='rfm_fisher',rfm_weight=.75),labels,emb)
    y=gold[labels].to_numpy(int);target=y/y.sum(1)[:,None]
    prior=(target.sum(0)+1)/(len(y)+5)
    score=p/prior[None,:]**c['power'];score/=np.maximum(score.sum(1,keepdims=True),1e-12)
    return score,models.threshold(score),dict(**audit,prior=prior.tolist(),power=c['power'])
def freeze():
    OUT.mkdir(exist_ok=True,parents=True)
    for name in ['data','protocols','references']:
        if not (OUT/name).exists():(OUT/name).symlink_to(BASE/name,target_is_directory=True)
    assert not (OUT/'registry.json').exists()
    e.save(OUT/'registry.json',dict(version='new-directions-v4',frozen_at=time.time(),recipes=configs(),
        sources={str(p.relative_to(BASE)):e.sha(p) for p in [BASE/x for x in ['models.py','study.py','models_v2.py','models_v3.py','study_v4.py']]},
        raw_data={p.name:e.sha(p) for p in (BASE/'data').glob('*.csv')},
        protocols={p.name:e.sha(p) for p in (BASE/'protocols').glob('*.json')},
        hypothesis='Kernel regression estimates source-prevalence-weighted probabilities. v3 failed Funding recall in supplied validation. Two fixed tempering strengths reduce source-prior influence; priors learned exclusively on each fold gold. This is development informed by reused supplementary data.',
        selection='Same first-three-repeats screen against Ani; also require positive mean improvement against exact new_blend_75 parent in BOTH pools before extension. Stop this line if neither passes. Do not reevaluate validation for failed arms.',
        feature_access='inductive',external_data=False,deep_learning=False))
def registry():
    r=json.loads((OUT/'registry.json').read_text())
    for p,h in r['sources'].items():assert e.sha(BASE/p)==h,p
    for p,h in r['raw_data'].items():assert e.sha(BASE/'data'/p)==h,p
    for p,h in r['protocols'].items():assert e.sha(BASE/'protocols'/p)==h,p
    return r
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    else:
        e.ROOT=OUT;e.registry=registry;e.models=types.SimpleNamespace(predict=predict,threshold=models.threshold)
        e.main()
