"""Freeze the comparison cohort before running the supplementary pooled CV."""
import json
from pathlib import Path
import benchmark_cv as b

def main():
    b.protocol();rs=b.registry();seen={b.digest(dict(config=r['config'],rule=r['rule'])) for r in rs}
    selection=b.OUT/'leader_cohort.json'
    if selection.exists():
        print('Leader cohort already frozen:',len(rs));return
    added=[]
    names={('phase10','structured_cluster'):'Structured-label SVM',('phase11','random_subspace'):'Unlabeled-subsample SVM ensemble',
        ('phase12',None):'Low-density SVM',('phase13',None):'Cluster SVM at threshold 0.60',('phase18','time_weights'):'Year-weighted SVM',
        ('phase21b','training_selected_reason_hybrid'):'Kernel-committee reason hybrid',('phase21c','classical_tri_majority'):'Classical tri-training',
        ('phase24','phase24_co_training'):'Metadata co-training hybrid',('phase25','phase25_local_ssl'):'Local SVM specialist hybrid',
        ('phase25b','phase25_control_label_tuned_ssl'):'Other SVM override hybrid',('phase25b','phase25_control_reliable_ssl'):'Uniform pseudo-weight hybrid',
        ('phase45',None):'Earlier low-density SVM',('phase6','pca'):'PCA-125 low-density SVM',
        ('phase6','auxiliary_features'):'Drug-target augmented SVM',('phase9','embedding_geometry'):'Embedding-kernel SVM'}
    for path in sorted((b.ROOT/'experiments').glob('phase*-validation-results.json')):
        obj=json.loads(path.read_text());items=obj if isinstance(obj,list) else [obj]
        for r in items:
            v=r.get('validation',r);score=v.get('sample_f1',r.get('validation_sample_f1',0))
            if not score or score<.9 or not r.get('config') or not r.get('rule'):continue
            if b.transductive(r['config']):continue
            key=b.digest(dict(config=r['config'],rule=r['rule']))
            if key in seen:continue
            phase=path.stem.split('-')[0];name=names.get((phase,r.get('name')),phase+': '+r.get('name',r['config']['model']))
            item=b.entry({**r,'validation':v},name,str(path.relative_to(b.ROOT)))
            rs.append(item);added.append(item);seen.add(key)
    b.save(selection,dict(frozen_at=b.now(),criterion='Existing five common-benchmark recipes plus every distinct inductive recipe with recorded validation sample-F1 >= 0.900. Deduplicate exact config and rule; no tuning.',
        model_count=len(rs),added_names=[r['name'] for r in added],recipes_sha256=b.digest([dict(config=r['config'],rule=r['rule']) for r in rs])))
    b.save(b.OUT/'registry.json',rs);print('Frozen comparison cohort:',len(rs),'recipes;',len(added),'added.')

if __name__=='__main__':main()
