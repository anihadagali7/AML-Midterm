"""Focused pair-specific discrimination after the three new-family screens."""
import sys,json,time,types
from pathlib import Path
import study as e,models,models_v14,study_v11,current_reference
BASE=Path(__file__).resolve().parent;OUT=BASE/'studies/v14'
def configs():
    cs=[dict(name=f'pair_lda_{population}_{scope}',family='pairwise_lda',head='lda',population=population,scope=scope,weight=.2) for population in ['gold','augmented'] for scope in ['final','both']]
    cs.extend([dict(name=f'pair_lr_augmented_{scope}',family='pairwise_lda',head='lr',population='augmented',scope=scope,weight=.2) for scope in ['final','both']])
    cs.extend([dict(name=f'pair_lda_augmented_{scope}_w40',family='pairwise_lda',head='lda',population='augmented',scope=scope,weight=.4) for scope in ['final','both']])
    return cs
def freeze():
    OUT.mkdir(exist_ok=True,parents=True)
    for name in ['data','references','protocols']:
        if not (OUT/name).exists():(OUT/name).symlink_to(BASE/name,target_is_directory=True)
    assert not (OUT/'registry.json').exists()
    sources=study_v11.registry()['sources'].copy();sources.update({p:e.sha(BASE/p) for p in ['models_v14.py','study_v14.py']})
    sources.update({str(p.relative_to(BASE)):e.sha(p) for p in (BASE/'references/pairwise').iterdir() if p.is_file()})
    e.save(OUT/'registry.json',dict(version='pair-discrimination-v14',frozen_at=time.time(),recipes=configs(),sources=sources,
      raw_data={p.name:e.sha(p) for p in (BASE/'data').glob('*.csv')},protocols={p.name:e.sha(p) for p in (BASE/'protocols').glob('*.json')},
      comparator='Current C.1/LDA ensemble; legacy ani_delta refers to this reference.',
      selection='Initial positive mean and >=2/3 wins in BOTH pools; confirm positive mean >=5/8 wins in BOTH; supplementary only after confirmation.',
      source='https://www.jmlr.org/papers/v5/wu04a.html',
      hypothesis='LDA was the strongest observed improvement. Pair-specific covariance may resolve heterogeneity hidden by its single shared covariance. Use ten pairwise heads and probability coupling, keeping all ensemble partners fixed. This is a focused classical adaptation, not a claim of a newly published algorithm.',
      adaptations='Each observed positive reason gives a class instance; multilabel positives appear in both applicable pair classes. Uniform pairwise LDA priors or balanced C.1 logistic controls. Wu-Lin-Weng method-2 quadratic coupling, fixed .3 source decoder / .5 ensemble decoder. Final-only or both pseudo rounds; gold-only versus augmented head. Weights .2, or .4 replacing additional logistic weight. Other components unchanged.',
      external_data=False,deep_learning=False,feature_access='inductive'))
    print('Frozen',len(configs()),'pairwise-reason recipes.',flush=True)
def registry():
    r=json.loads((OUT/'registry.json').read_text())
    for p,h in r['sources'].items():assert e.sha(BASE/p)==h,p
    for p,h in r['raw_data'].items():assert e.sha(BASE/'data'/p)==h,p
    for p,h in r['protocols'].items():assert e.sha(BASE/'protocols'/p)==h,p
    return r
def configure():
    e.ROOT=OUT;e.registry=registry;e.baseline=current_reference.baseline;e.models=types.SimpleNamespace(predict=models_v14.predict,threshold=models.threshold)
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    else:configure();e.main()
