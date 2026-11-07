"""Recent-method study 1: uncertainty-preserving label distribution propagation."""
import sys,json,time,types
from pathlib import Path
import study as e,models,models_v7,study_v5
BASE=Path(__file__).resolve().parent;OUT=BASE/'studies/v7'
def configs():
    cs=[]
    for k in [10,30,60]:cs.append(dict(name=f'ldp_k{k}',family='ldp',neighbors=k))
    for k in [10,30]:cs.append(dict(name=f'ldp_binary_k{k}',family='ldp',neighbors=k,binary=True))
    cs.append(dict(name='ldp_student_gold_control',family='ldp',neighbors=30,student=True,pseudo_mass=0.))
    for k in [10,30]:cs.append(dict(name=f'ldp_student_k{k}',family='ldp',neighbors=k,student=True,pseudo_mass=.5))
    cs.append(dict(name='hybrid_ldp_k30',family='ldp',neighbors=30,blend=.2))
    cs.append(dict(name='hybrid_ldp_student_k30',family='ldp',neighbors=30,student=True,pseudo_mass=.5,blend=.2))
    return cs
def freeze():
    OUT.mkdir(exist_ok=True,parents=True)
    for name in ['data','references','protocols']:
        if not (OUT/name).exists():(OUT/name).symlink_to(BASE/name,target_is_directory=True)
    assert not (OUT/'registry.json').exists()
    sources=study_v5.registry()['sources'].copy();sources.update({p:e.sha(BASE/p) for p in ['models_v7.py','study_v7.py']})
    e.save(OUT/'registry.json',dict(version='recent-method-v7',frozen_at=time.time(),recipes=configs(),sources=sources,
      raw_data={p.name:e.sha(p) for p in (BASE/'data').glob('*.csv')},protocols={p.name:e.sha(p) for p in (BASE/'protocols').glob('*.json')},
      comparator='Submitted 70/15/15 ensemble; legacy ani_delta is delta versus this reference.',
      selection='Positive mean and >=2/3 repeat wins in both pools advances; confirm positive mean and >=5/8 wins in both before supplementary checks.',
      source='Gottcke, Zimek and Campello, Information Systems 129 (2025), 102507. Algorithm 1 and inductive extension. Earlier conference version 2021.',
      adaptations='Multilabel normalized or independent binary evidence; duplicate-node pooling; fixed nonempty decoder; nearest-gold fallback for zero known mass; optional uncertainty-weighted linear student. Query rows never enter graph.',
      external_data=False,deep_learning=False,feature_access='inductive'))
    print('Frozen',len(configs()),'kNN-LDP recipes.',flush=True)
def registry():
    r=json.loads((OUT/'registry.json').read_text())
    for p,h in r['sources'].items():assert e.sha(BASE/p)==h,p
    for p,h in r['raw_data'].items():assert e.sha(BASE/'data'/p)==h,p
    for p,h in r['protocols'].items():assert e.sha(BASE/'protocols'/p)==h,p
    return r
def configure():
    e.ROOT=OUT;e.registry=registry;e.baseline=study_v5.baseline;e.models=types.SimpleNamespace(predict=models_v7.predict,threshold=models.threshold)
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    else:configure();e.main()
