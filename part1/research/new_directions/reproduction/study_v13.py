"""JARF-inspired kernel-preconditioned tree experiment, frozen before scoring."""
import sys,json,time,types
from pathlib import Path
import study as e,models,models_v13,study_v11,current_reference
BASE=Path(__file__).resolve().parent;OUT=BASE/'studies/v13'
def configs():
    settings=[('forest_raw','forest','raw',False),('forest_pca','forest','pca',False),
      ('forest_jacobian','forest','jacobian',False),('forest_axes','forest','eigen',False),
      ('pseudo_forest_jacobian','forest','jacobian',True),('boost_raw','boost','raw',False),
      ('boost_jacobian','boost','jacobian',False),('boost_axes','boost','eigen',False)]
    cs=[dict(name=name,family='jacobian_tree',learner=learner,geometry=geometry,pseudo=pseudo,blend=0.) for name,learner,geometry,pseudo in settings]
    cs.extend([{**c,'name':'hybrid_'+c['name'],'blend':.35} for c in cs if c['geometry'] not in ['raw','pca']])
    return cs
def freeze():
    OUT.mkdir(exist_ok=True,parents=True)
    for name in ['data','references','protocols']:
        if not (OUT/name).exists():(OUT/name).symlink_to(BASE/name,target_is_directory=True)
    assert not (OUT/'registry.json').exists()
    sources=study_v11.registry()['sources'].copy();sources.update({p:e.sha(BASE/p) for p in ['models_v13.py','study_v13.py']})
    sources.update({str(p.relative_to(BASE)):e.sha(p) for p in (BASE/'references/jarf_2025').iterdir() if p.is_file()})
    e.save(OUT/'registry.json',dict(version='jacobian-trees-v13',frozen_at=time.time(),recipes=configs(),sources=sources,
      raw_data={p.name:e.sha(p) for p in (BASE/'data').glob('*.csv')},protocols={p.name:e.sha(p) for p in (BASE/'protocols').glob('*.json')},
      comparator='Current C.1/LDA ensemble; legacy ani_delta refers to this reference.',
      selection='Initial positive mean and >=2/3 wins in BOTH pools; confirm positive mean >=5/8 wins in BOTH; supplementary only after confirmation.',
      source='https://arxiv.org/abs/2512.08306',status='December 2025 preprint; adaptation, not exact benchmark replication.',
      hypothesis='Original coordinates are randomly rotated. Use one kernel-Jacobian preconditioner to expose useful combinations to nonlinear tree classifiers; distinguish this from using a learned kernel as the classifier.',
      adaptations='Binary kernel-ridge analytic gradients replace forest finite differences; 256 fixed-seed allowed fitting-input probes, summed gradient outer products, trace normalization plus .01 ridge. Compare H-map from paper and full eigenbasis coordinates. Raw/PCA controls. RF 200 entropy trees; fixed shallow boosted trees. Optional existing GMM/LDA missing labels with total weight .25 times gold. Supervised-only standalone models are diagnostics; blends or pseudo variant retain missing-label generation.',
      external_data=False,deep_learning=False,feature_access='inductive'))
    print('Frozen',len(configs()),'Jacobian-tree recipes.',flush=True)
def registry():
    r=json.loads((OUT/'registry.json').read_text())
    for p,h in r['sources'].items():assert e.sha(BASE/p)==h,p
    for p,h in r['raw_data'].items():assert e.sha(BASE/'data'/p)==h,p
    for p,h in r['protocols'].items():assert e.sha(BASE/'protocols'/p)==h,p
    return r
def configure():
    e.ROOT=OUT;e.registry=registry;e.baseline=current_reference.baseline;e.models=types.SimpleNamespace(predict=models_v13.predict,threshold=models.threshold)
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    else:configure();e.main()
