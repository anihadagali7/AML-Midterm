"""Recent-method study 3: Dirichlet probability-space generative learning."""
import sys,json,time,types
from pathlib import Path
import study as e,models,models_v9,study_v5
BASE=Path(__file__).resolve().parent;OUT=BASE/'studies/v9'
def configs():
    cs=[dict(name='simplex_prototype_control',family='dirichlet',representation='prototypes',control=True)]
    for rep in ['prototypes','anchors']:
        for mass in [0.,.5]:cs.append(dict(name=f'dirichlet_{rep}_u{mass:g}',family='dirichlet',representation=rep,unlabeled_mass=mass))
    cs.append(dict(name='dirichlet_anchors_prior',family='dirichlet',representation='anchors',unlabeled_mass=.5,prior_power=1.))
    for rep in ['prototypes','anchors']:cs.append(dict(name=f'hybrid_dirichlet_{rep}',family='dirichlet',representation=rep,unlabeled_mass=.5,blend=.2))
    return cs
def freeze():
    OUT.mkdir(exist_ok=True,parents=True)
    for name in ['data','references','protocols']:
        if not (OUT/name).exists():(OUT/name).symlink_to(BASE/name,target_is_directory=True)
    assert not (OUT/'registry.json').exists()
    sources=study_v5.registry()['sources'].copy();sources.update({p:e.sha(BASE/p) for p in ['models_v9.py','study_v9.py']})
    e.save(OUT/'registry.json',dict(version='recent-method-v9',frozen_at=time.time(),recipes=configs(),sources=sources,
      raw_data={p.name:e.sha(p) for p in (BASE/'data').glob('*.csv')},protocols={p.name:e.sha(p) for p in (BASE/'protocols').glob('*.json')},
      comparator='Submitted 70/15/15 ensemble; legacy ani_delta is delta versus this reference.',
      selection='Positive mean and >=2/3 repeat wins in both pools advances; confirm positive mean and >=5/8 wins in both before supplementary checks.',
      source='Martin et al., Transductive Zero-Shot and Few-Shot CLIP, CVPR 2024. Dirichlet likelihood and MM-quadratic solver only.',
      adaptations='No CLIP/neural features. Supplied embeddings become probability coordinates via gold prototypes or unsupervised 16 anchors. Inductive query boundary; only masked fitting rows get EM assignments. Fixed gold multilabel mass, 2-observation shrinkage, 20 EM / up to 300 MM updates. Uniform or learned priors; no MDL class-discovery penalty.',
      external_data=False,deep_learning=False,feature_access='inductive'))
    print('Frozen',len(configs()),'Dirichlet recipes.',flush=True)
def registry():
    r=json.loads((OUT/'registry.json').read_text())
    for p,h in r['sources'].items():assert e.sha(BASE/p)==h,p
    for p,h in r['raw_data'].items():assert e.sha(BASE/'data'/p)==h,p
    for p,h in r['protocols'].items():assert e.sha(BASE/'protocols'/p)==h,p
    return r
def configure():
    e.ROOT=OUT;e.registry=registry;e.baseline=study_v5.baseline;e.models=types.SimpleNamespace(predict=models_v9.predict,threshold=models.threshold)
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    else:configure();e.main()
