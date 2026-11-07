"""New version, same frozen benchmark helpers and exact splits."""
import sys,json,time
from pathlib import Path
import study as e
import models_v2

BASE=Path(__file__).resolve().parent
OUT=BASE/'studies/v2'

def configs():
    cs=[]
    for k,t in [(10,.5),(10,1.),(30,.5),(30,1.)]:
        cs.append(dict(name=f'quotient_poisson_k{k}_t{t:g}',family='poisson_unique',neighbors=k,temperature=t))
    for steps,reg,power,aug in [(0,.01,.5,False),(2,.01,.5,False),(4,.01,.5,False),
                                (2,.1,.5,False),(2,.01,1.,False),(2,.01,.5,True)]:
        cs.append(dict(name=f'rfm_i{steps}_r{reg:g}_p{power:g}_aug{int(aug)}',family='rfm',steps=steps,
                       ridge=reg,power=power,shrink=.25,bandwidth=1.,augment=aug))
    return cs

def freeze():
    OUT.mkdir(exist_ok=True,parents=True)
    for name in ['data','protocols','references']:
        if not (OUT/name).exists():(OUT/name).symlink_to(BASE/name,target_is_directory=True)
    assert not (OUT/'registry.json').exists()
    e.save(OUT/'registry.json',dict(version='new-directions-v2',frozen_at=time.time(),recipes=configs(),
        sources={str(p.relative_to(BASE)):e.sha(p) for p in [BASE/'models.py',BASE/'study.py',BASE/'models_v2.py',BASE/'study_v2.py']},
        raw_data={p.name:e.sha(p) for p in (BASE/'data').glob('*.csv')},
        protocols={p.name:e.sha(p) for p in (BASE/'protocols').glob('*.json')},
        selection='Same dual-pool 3-repeat screen and 8-repeat confirmation versus Ani, >=2/3 then >=5/8 wins and positive means in both. Validation/chronology only after shortlisting.',
        hypotheses={'poisson_unique':'Unsupervised graph diagnostic found 4 disconnected components caused by repeat embeddings; quotient graph has 1. Coalesce duplicates before Poisson solve.',
                    'rfm':'Learn a full regularized Mahalanobis metric through kernel gradient outer products, using only allowed fitting rows; classical kernel ridge regression, no neural network. Include zero-iteration control and explicit K-means augmentation variant.'},
        feature_access='inductive',external_data=False,deep_learning=False))
    print('Frozen 10 follow-up recipes before scoring.',flush=True)

def registry():
    r=json.loads((OUT/'registry.json').read_text())
    for p,h in r['sources'].items():assert e.sha(BASE/p)==h,p
    for p,h in r['raw_data'].items():assert e.sha(BASE/'data'/p)==h,p
    for p,h in r['protocols'].items():assert e.sha(BASE/'protocols'/p)==h,p
    return r

if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    else:
        e.ROOT=OUT;e.registry=registry;e.models.predict=models_v2.predict
        e.main()
