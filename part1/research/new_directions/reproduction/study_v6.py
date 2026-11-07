"""Requested multi-algorithm clustering ensemble, after the v5 objectives."""
import sys,json,time,types
from pathlib import Path
import study as e
import models,models_v6,study_v5
BASE=Path(__file__).resolve().parent;OUT=BASE/'studies/v6'
def configs():
    cs=[]
    def add(name,members,weight=.2,**kw):
        cs.append(dict(name=name,family='cluster_ensemble',members=members,weight=weight,**kw))
    add('selftrain_no_clusters',[],0.)
    for m in ['gmm','kmeans','ward','density']:add('single_'+m,[m])
    three=['gmm','kmeans','ward'];four=three+['density']
    add('teacher_mean3',three);add('teacher_mean4',four)
    add('teacher_mean4_w50',four,.5)
    add('teacher_agreement4',four,agreement=True)
    add('coassociation',['coassoc']);add('coassociation_w50',['coassoc'],.5)
    add('prediction_mean3',three,output_average=True);add('prediction_mean4',four,output_average=True)
    add('hybrid_teacher4',four,replace_weight=.2)
    add('hybrid_coassociation',['coassoc'],replace_weight=.2)
    add('hybrid_prediction4',four,output_average=True,replace_weight=.2)
    return cs
def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['data','references','protocols']:
        if not (OUT/name).exists():(OUT/name).symlink_to(BASE/name,target_is_directory=True)
    assert not (OUT/'registry.json').exists()
    sources=study_v5.registry()['sources'].copy()
    sources.update({p:e.sha(BASE/p) for p in ['models_v6.py','study_v6.py','references/ani_adapter.py']})
    e.save(OUT/'registry.json',dict(version='new-directions-v6',frozen_at=time.time(),recipes=configs(),sources=sources,
        raw_data={p.name:e.sha(p) for p in (BASE/'data').glob('*.csv')},protocols={p.name:e.sha(p) for p in (BASE/'protocols').glob('*.json')},
        comparator='Submitted 70/15/15 ensemble. Secondary matched GMM/self-training controls. Legacy ani_delta fields refer to submitted ensemble.',
        selection='Positive paired mean and >=2/3 wins in BOTH pools against submitted ensemble advances to 8 total repeats; then positive mean and >=5/8 wins in BOTH pools required before fixed validation and chronological checks. No threshold search.',
        hypothesis='Average label-aligned cluster teacher probabilities or final classifier probabilities across GMM, spherical-view k-means, Ward, and density clusters. Also label-free co-association consensus and an agreement-only teacher. Keep self-training, classifier, acceptance, and decoder fixed.',
        research='https://www.jmlr.org/papers/v3/strehl02a.html; co-association inspiration, average-linkage adaptation, not an exact reproduction of CSPA.',
        control='single_gmm must reproduce existing Ani fold exactly; other single teachers and cluster-free self-training isolate averaging benefit.',
        external_data=False,deep_learning=False,feature_access='inductive',query_labels=False))
    print('Frozen 16 clustering recipes including matched controls.',flush=True)
def registry():
    r=json.loads((OUT/'registry.json').read_text())
    for p,h in r['sources'].items():assert e.sha(BASE/p)==h,p
    for p,h in r['raw_data'].items():assert e.sha(BASE/'data'/p)==h,p
    for p,h in r['protocols'].items():assert e.sha(BASE/'protocols'/p)==h,p
    return r
def configure():
    e.ROOT=OUT;e.registry=registry;e.baseline=study_v5.baseline;e.models=types.SimpleNamespace(predict=models_v6.predict,threshold=models.threshold)
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    else:configure();e.main()
