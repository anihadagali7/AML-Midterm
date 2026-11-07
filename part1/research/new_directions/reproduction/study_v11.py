"""Reason-specific metrics tested against the stronger current LDA reference."""
import sys,json,time,types
from pathlib import Path
import study as e,models,models_v11,study_v10,current_reference
BASE=Path(__file__).resolve().parent;OUT=BASE/'studies/v11'
def configs():
    return [dict(name='current_lda_control',family='reason_metric',kind='control'),
      *[dict(name=f'shared_norm_w{int(w*100)}',family='reason_metric',kind='shared',weights=[w]*5) for w in [.25,.5]],
      dict(name='shared_binary_w35',family='reason_metric',kind='shared',binary=True,weights=[.35]*5),
      dict(name='separate_w35',family='reason_metric',kind='separate',steps=1,balance=False,weights=[.35]*5),
      dict(name='separate_balanced_w35',family='reason_metric',kind='separate',steps=1,balance=True,weights=[.35]*5),
      dict(name='separate_balanced_other60',family='reason_metric',kind='separate',steps=1,balance=True,weights=[0,0,0,0,.6]),
      dict(name='separate_balanced_two_w35',family='reason_metric',kind='separate',steps=2,balance=True,weights=[.35]*5)]
def freeze():
    OUT.mkdir(exist_ok=True,parents=True)
    for name in ['data','references','protocols']:
        if not (OUT/name).exists():(OUT/name).symlink_to(BASE/name,target_is_directory=True)
    assert not (OUT/'registry.json').exists()
    sources=study_v10.registry()['sources'].copy()
    sources.update({p:e.sha(BASE/p) for p in ['current_reference.py','models_v11.py','study_v11.py']})
    sources.update({str(p.relative_to(BASE)):e.sha(p) for p in (BASE/'references/lda_current').iterdir() if p.is_file()})
    e.save(OUT/'registry.json',dict(version='reason-metrics-v11',frozen_at=time.time(),recipes=configs(),sources=sources,
      raw_data={p.name:e.sha(p) for p in (BASE/'data').glob('*.csv')},protocols={p.name:e.sha(p) for p in (BASE/'protocols').glob('*.json')},
      comparator='Current C.1/LDA 70/15/15 ensemble .931794/.921631; legacy ani_delta refers to this reference.',
      selection='Initial positive mean and >=2/3 wins in BOTH pools, then positive mean >=5/8 wins in BOTH. Supplementary checks for confirmed candidates only.',
      hypothesis='Previous kernel blend changes few predictions. Each reason gets a distinct AGOP metric, optionally class-balanced gradient averaging; .1 identity shrinkage prevents rank collapse. Shared kernel transfer to new stronger parent is a matched control, not a new method.',
      sources_reviewed=['https://arxiv.org/abs/2508.10053'],external_data=False,deep_learning=False,feature_access='inductive'))
    print('Frozen',len(configs()),'reason-metric recipes against latest LDA reference.',flush=True)
def registry():
    r=json.loads((OUT/'registry.json').read_text())
    for p,h in r['sources'].items():assert e.sha(BASE/p)==h,p
    for p,h in r['raw_data'].items():assert e.sha(BASE/'data'/p)==h,p
    for p,h in r['protocols'].items():assert e.sha(BASE/'protocols'/p)==h,p
    return r
def configure():
    e.ROOT=OUT;e.registry=registry;e.baseline=current_reference.baseline;e.models=types.SimpleNamespace(predict=models_v11.predict,threshold=models.threshold)
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    else:configure();e.main()
