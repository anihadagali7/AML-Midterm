"""Frozen independent RFM follow-up, with identical external evaluation protocol."""
import sys,json,time
from pathlib import Path
import study as e
import models_v3
BASE=Path(__file__).resolve().parent
OUT=BASE/'studies/v3'
def configs():
    return [dict(name='rfm_joint',family='rfm_structured',joint=True),
            dict(name='rfm_unique_gradient',family='rfm_structured',unique_gradient=True),
            dict(name='rfm_group_bag',family='rfm_bag'),
            dict(name='rfm_nested',family='rfm_nested'),
            dict(name='new_blend_50',family='rfm_fisher',rfm_weight=.5),
            dict(name='new_blend_75',family='rfm_fisher',rfm_weight=.75)]
def freeze():
    OUT.mkdir(exist_ok=True,parents=True)
    for name in ['data','protocols','references']:
        if not (OUT/name).exists():(OUT/name).symlink_to(BASE/name,target_is_directory=True)
    assert not (OUT/'registry.json').exists()
    e.save(OUT/'registry.json',dict(version='new-directions-v3',frozen_at=time.time(),recipes=configs(),
        sources={str(p.relative_to(BASE)):e.sha(p) for p in [BASE/x for x in ['models.py','study.py','models_v2.py','models_v3.py','study_v3.py']]},
        raw_data={p.name:e.sha(p) for p in (BASE/'data').glob('*.csv')},
        protocols={p.name:e.sha(p) for p in (BASE/'protocols').glob('*.json')},
        selection='Same Ani dual-pool screen (positive means, >=2/3 wins); 8-repeat confirmation >=5/8. Also compare changes to exact RFM parent. No old model included in new blends. Validation and chronology after shortlist only.',
        hypotheses={'joint':'Train kernel on label-set posteriors, decode expected sample F1.',
            'unique_gradient':'Prevent repeated embeddings from dominating learned geometry.',
            'bagging':'Average full fit with five 80% grouped-gold fits to reduce metric estimation variance.',
            'nested':'Select bandwidth, ridge and cutoff inside each training fold using 3-fold grouped CV with masked-group purges.',
            'blend':'Combine complementary new learned-kernel and Fisher-residual representations; predeclared weights .5 and .75.'},
        feature_access='inductive',external_data=False,deep_learning=False))
    print('Frozen six follow-up recipes before scoring.',flush=True)
def registry():
    r=json.loads((OUT/'registry.json').read_text())
    for p,h in r['sources'].items():assert e.sha(BASE/p)==h,p
    for p,h in r['raw_data'].items():assert e.sha(BASE/'data'/p)==h,p
    for p,h in r['protocols'].items():assert e.sha(BASE/'protocols'/p)==h,p
    return r
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    else:
        e.ROOT=OUT;e.registry=registry
        # Override the caller's model namespace, leaving imported model implementations intact.
        import types
        e.models=types.SimpleNamespace(predict=models_v3.predict,threshold=models_v3.models.threshold)
        e.main()
