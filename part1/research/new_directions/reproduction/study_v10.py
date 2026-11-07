"""Follow observed inner-selection preference without tuning on supplementary labels."""
import sys,json,time,types
from pathlib import Path
import study as e,models,models_v10,study_v8,study_v5
BASE=Path(__file__).resolve().parent;OUT=BASE/'studies/v10'
def configs():
    cs=[]
    settings=[('raw',False,1.,False),('unit',True,1.,False),('unit_q08',True,.8,False),('unit_binary',True,1.,True)]
    for name,unit,q,binary in settings:
        for blend in [0.,.2]:
            cs.append(dict(name=('hybrid_' if blend else '')+'leaf_one_'+name,family='xrfm_one',unit=unit,q=q,binary=binary,blend=blend))
    return cs
def freeze():
    OUT.mkdir(exist_ok=True,parents=True)
    for name in ['data','references','protocols']:
        if not (OUT/name).exists():(OUT/name).symlink_to(BASE/name,target_is_directory=True)
    assert not (OUT/'registry.json').exists()
    sources=study_v8.registry()['sources'].copy();sources.update({p:e.sha(BASE/p) for p in ['models_v10.py','study_v10.py']})
    e.save(OUT/'registry.json',dict(version='recent-method-v10',frozen_at=time.time(),recipes=configs(),sources=sources,
      raw_data={p.name:e.sha(p) for p in (BASE/'data').glob('*.csv')},protocols={p.name:e.sha(p) for p in (BASE/'protocols').glob('*.json')},
      comparator='Submitted 70/15/15 ensemble; legacy ani_delta is delta versus this reference.',
      selection='Positive mean and >=2/3 repeat wins in both pools advances; confirm positive mean and >=5/8 wins in both before supplementary checks.',
      hypothesis='V8 inner folds usually select one AGOP update; a fixed simpler learner may reduce selection variance. Unit normalization respects semantic embedding geometry; raw one-update arm is matched control. Same xRFM algorithm, not another new method family.',
      evidence_before_plan={'v8_screen':e.sha(BASE/'studies/v8/comparison_3.csv'),'v8_confirmation':e.sha(BASE/'studies/v8/comparison_8.csv')},
      external_data=False,deep_learning=False,feature_access='inductive'))
    print('Frozen',len(configs()),'one-step follow-up recipes.',flush=True)
def registry():
    r=json.loads((OUT/'registry.json').read_text())
    for p,h in r['sources'].items():assert e.sha(BASE/p)==h,p
    for p,h in r['raw_data'].items():assert e.sha(BASE/'data'/p)==h,p
    for p,h in r['protocols'].items():assert e.sha(BASE/'protocols'/p)==h,p
    return r
def configure():
    e.ROOT=OUT;e.registry=registry;e.baseline=study_v5.baseline;e.models=types.SimpleNamespace(predict=models_v10.predict,threshold=models.threshold)
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    else:configure();e.main()
