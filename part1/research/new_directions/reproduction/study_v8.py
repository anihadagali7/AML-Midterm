"""Recent-method study 2: xRFM leaf kernels, nested selection, LDP augmentation."""
import sys,json,time,types
from pathlib import Path
import study as e,models,models_v8,study_v7
BASE=Path(__file__).resolve().parent;OUT=BASE/'studies/v8'
def configs():
    cs=[]
    for i,(name,c) in enumerate(zip(['l2','l2_q08','pca_l1','pca_l1_q08','binary_l2','pca_l1_r01','pca_diag_l1'],models_v8.candidates())):
        cs.append(dict(name='leaf_'+name,family='xrfm_leaf',steps=3,**c))
    cs.append(dict(name='leaf_kernel_control',family='xrfm_leaf',p=2.,q=1.,ridge=.01,steps=0))
    cs.append(dict(name='leaf_nested',family='xrfm_leaf',nested=True))
    cs.append(dict(name='leaf_nested_ldp',family='xrfm_leaf',nested=True,pseudo_mass=.3))
    cs.append(dict(name='hybrid_leaf_nested',family='xrfm_leaf',nested=True,blend=.2))
    cs.append(dict(name='hybrid_leaf_nested_ldp',family='xrfm_leaf',nested=True,pseudo_mass=.3,blend=.2))
    return cs
def freeze():
    OUT.mkdir(exist_ok=True,parents=True)
    for name in ['data','references','protocols']:
        if not (OUT/name).exists():(OUT/name).symlink_to(BASE/name,target_is_directory=True)
    assert not (OUT/'registry.json').exists()
    sources=study_v7.registry()['sources'].copy();sources.update({p:e.sha(BASE/p) for p in ['models_v8.py','study_v8.py','models_v3.py']})
    literature={str(p.relative_to(BASE)):e.sha(p) for p in (BASE/'references/recent_literature').iterdir() if p.is_file()}
    e.save(OUT/'registry.json',dict(version='recent-method-v8',frozen_at=time.time(),recipes=configs(),sources=sources,literature=literature,
      raw_data={p.name:e.sha(p) for p in (BASE/'data').glob('*.csv')},protocols={p.name:e.sha(p) for p in (BASE/'protocols').glob('*.json')},
      comparator='Submitted 70/15/15 ensemble; legacy ani_delta is delta versus this reference.',
      selection='Positive mean and >=2/3 repeat wins in both pools advances; confirm positive mean and >=5/8 wins in both before supplementary checks.',
      source='Beaglehole et al., xRFM, ICLR 2026, Algorithm A.1 and author Lp,q kernels/AGOP. Dataset smaller than normal leaf capacity; one leaf, no tree needed.',
      adaptations='NumPy CPU implementation; fixed three iterations or inner grouped selection across 7 kernels and 0..4 iterations; normalized multilabel .3 / binary .5 fixed decoders. Unsupervised PCA only fitting inputs. LDP augmentation keeps selected metric fixed and adds soft pseudo-labels with .3x total gold mass.',
      external_data=False,deep_learning=False,feature_access='inductive'))
    print('Frozen',len(configs()),'xRFM leaf recipes.',flush=True)
def registry():
    r=json.loads((OUT/'registry.json').read_text())
    for p,h in r['sources'].items():assert e.sha(BASE/p)==h,p
    for p,h in r['raw_data'].items():assert e.sha(BASE/'data'/p)==h,p
    for p,h in r['protocols'].items():assert e.sha(BASE/'protocols'/p)==h,p
    for p,h in r['literature'].items():assert e.sha(BASE/p)==h,p
    return r
def configure():
    e.ROOT=OUT;e.registry=registry;e.baseline=study_v7.study_v5.baseline;e.models=types.SimpleNamespace(predict=models_v8.predict,threshold=models.threshold)
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    else:configure();e.main()
