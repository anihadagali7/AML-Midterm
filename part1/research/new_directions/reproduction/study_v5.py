"""Continuation requested by user: new objectives, then clustering consensus in a later study."""
import sys,json,time,types
from pathlib import Path
import numpy as np
import study as e
import models,models_v5
BASE=Path(__file__).resolve().parent;OUT=BASE/'studies/v5'
def configs():
    cs=[]
    for rep in ['linear','kernel']:
        for strength in [0.,.1,.5]:cs.append(dict(name=f'entropy_{rep}_{strength:g}',family='entropy',representation=rep,strength=strength))
    for k,mass in [(1,.5),(3,0.),(3,.5),(3,2.)]:cs.append(dict(name=f'mixture_k{k}_u{mass:g}',family='multimixture',components=k,unlabeled_mass=mass))
    for name,w in [('all15',[.15]*5),('all30',[.3]*5),('safety50',[.5,0,0,0,0]),('safety_funding50',[.5,.5,0,0,0]),('first3_50',[.5,.5,.5,0,0])]:
        cs.append(dict(name=f'rfm_complement_{name}',family='complement',weights=w))
    return cs
def baseline(pool):
    z=[np.load(BASE/'references'/f'threeway_{pool}_{p}.npz',allow_pickle=True) for p in ['first','extra']]
    scores=np.concatenate([v['scores'] for v in z]);preds=np.concatenate([v['predictions'] for v in z]);y=z[0]['truth']
    return dict(scores=scores,predictions=preds,truth=y,ids=z[0]['ids'],repeat_f1=np.array([e.metric(y,p)['sample_f1'] for p in preds]))
def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['data','references','protocols']:
        if not (OUT/name).exists():(OUT/name).symlink_to(BASE/name,target_is_directory=True)
    assert not (OUT/'registry.json').exists()
    sources={p:e.sha(BASE/p) for p in ['study.py','models.py','models_v2.py','models_v5.py','study_v5.py']}
    sources.update({'references/legacy_engine/'+p:h for p,h in json.loads((BASE/'references/legacy_engine_manifest.json').read_text()).items()})
    e.save(OUT/'registry.json',dict(version='new-directions-v5',frozen_at=time.time(),recipes=configs(),sources=sources,
        raw_data={p.name:e.sha(p) for p in (BASE/'data').glob('*.csv')},protocols={p.name:e.sha(p) for p in (BASE/'protocols').glob('*.json')},
        comparator='Submitted 70/15/15 ensemble; legacy ani_delta fields in shared helper refer to THIS declared comparator.',
        selection='First three saved repeats in both pools: positive mean and >=2/3 wins versus submitted ensemble. Qualifiers receive 5 remaining repeats; positive mean >=5/8 wins in both required. Then fixed validation and chronology. Standalone methods also compared to exact supervised/no-unlabeled controls.',
        hypothesis='Direct entropy minimization learns low-density boundaries; multiple mixture components represent semantic subtypes per reason; selective learned-kernel use retains complementary Safety/Funding corrections.',
        next_requested='After this work, test averages of several clustering types with separate individual-clustering controls.',
        external_data=False,deep_learning=False,feature_access='inductive'))
    print('Frozen 15 recipes against the stronger submitted-ensemble reference.',flush=True)
def registry():
    r=json.loads((OUT/'registry.json').read_text())
    for p,h in r['sources'].items():assert e.sha(BASE/p)==h,p
    for p,h in r['raw_data'].items():assert e.sha(BASE/'data'/p)==h,p
    for p,h in r['protocols'].items():assert e.sha(BASE/'protocols'/p)==h,p
    return r
def configure():
    e.ROOT=OUT;e.registry=registry;e.baseline=baseline;e.models=types.SimpleNamespace(predict=models_v5.predict,threshold=models.threshold)
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    else:configure();e.main()
