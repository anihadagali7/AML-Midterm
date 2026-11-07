"""Geometry-first label evidence: fixed graph, adaptive gate, linear student."""
import sys,json,time,types
from pathlib import Path
import study as e,models,models_v12,study_v11,current_reference
BASE=Path(__file__).resolve().parent;OUT=BASE/'studies/v12'
def configs():
    settings=[('gold',False,'forward',0.),('seeds_fwd',False,'forward',.3),
      ('prop_fwd',True,'forward',.3),('prop_reverse',True,'reverse',.3),('prop_reverse64',True,'reverse',64.)]
    cs=[dict(name=name,family='vast_linear',propagate=prop,direction=direction,strength=strength,blend=0.) for name,prop,direction,strength in settings]
    cs.extend([{**c,'name':'hybrid_'+c['name'],'blend':.25} for c in cs if c['propagate']])
    return cs
def freeze():
    OUT.mkdir(exist_ok=True,parents=True)
    for name in ['data','references','protocols']:
        if not (OUT/name).exists():(OUT/name).symlink_to(BASE/name,target_is_directory=True)
    assert not (OUT/'registry.json').exists()
    sources=study_v11.registry()['sources'].copy();sources.update({p:e.sha(BASE/p) for p in ['models_v12.py','study_v12.py']})
    sources.update({str(p.relative_to(BASE)):e.sha(p) for p in (BASE/'references/vast_2026').iterdir() if p.is_file()})
    e.save(OUT/'registry.json',dict(version='vast-classical-v12',frozen_at=time.time(),recipes=configs(),sources=sources,
      raw_data={p.name:e.sha(p) for p in (BASE/'data').glob('*.csv')},protocols={p.name:e.sha(p) for p in (BASE/'protocols').glob('*.json')},
      comparator='Current C.1/LDA ensemble; legacy ani_delta refers to this reference.',
      selection='Initial positive mean and >=2/3 wins in BOTH pools; confirm positive mean >=5/8 wins in BOTH; supplementary only after confirmation.',
      source='https://arxiv.org/abs/2609.14451',status='September 2026 preprint, not established clinical-trial performance.',
      adaptations='Only classical graph mechanism. Supplied vectors standardized and unit-normalized; identical vectors collapse to nodes, fractional gold-label seeds. Minimal connected union-kNN, log degree correction, symmetric prior 1/5, receiver .75, synchronous accumulated-evidence updates until frozen set stops growing. Gold roles frozen per paper; true classifier labels unchanged. Adaptive gate paper constants. Replace neural student by regularized linear softmax; compare explicit reverse KL with ordinary soft-target cross entropy; .3 mass and paper 64. No external encoder.',
      external_data=False,deep_learning=False,feature_access='inductive'))
    print('Frozen',len(configs()),'classical VAST adaptations.',flush=True)
def registry():
    r=json.loads((OUT/'registry.json').read_text())
    for p,h in r['sources'].items():assert e.sha(BASE/p)==h,p
    for p,h in r['raw_data'].items():assert e.sha(BASE/'data'/p)==h,p
    for p,h in r['protocols'].items():assert e.sha(BASE/'protocols'/p)==h,p
    return r
def configure():
    e.ROOT=OUT;e.registry=registry;e.baseline=current_reference.baseline;e.models=types.SimpleNamespace(predict=models_v12.predict,threshold=models.threshold)
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    else:configure();e.main()
