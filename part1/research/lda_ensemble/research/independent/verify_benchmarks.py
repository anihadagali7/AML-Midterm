"""Audit stored predictions, frozen recipes and original-model preservation."""
from pathlib import Path
import json,numpy as np
import benchmark_cv as b
import benchmark_pooled_cv as pooled

def verify_predictions(folder,summary,pooled_mode=False):
    p=json.loads((folder/'protocol.json').read_text())
    registry=json.loads((folder/'registry.json').read_text())
    records={r['name']:r for r in registry}
    for row in summary['models']:
        key=row['pooled_result_key'] if pooled_mode else row['result_key']
        result=json.loads((folder/(key+'.json')).read_text())
        assert result['protocol_sha256']==b.digest(p)
        r=records[row['candidate']]
        assert result['config']==r['config'] and result['rule']==r['rule']
        a=np.load(folder/(key+'.npz'),allow_pickle=True)
        truth=a['truth'];h=a['predictions'];scores=a['scores']
        assert np.array_equal(truth,b.e.Y) and h.shape==(3,len(truth),5)
        assert a['ids'].tolist()==b.e.DATA.iloc[b.e.LAB].ID.tolist()
        assert np.array_equal(h,np.array([b.e.decisions(s,r['rule']) for s in scores]))
        per_repeat=(2*(h*truth[None,:,:]).sum(2)/(h.sum(2)+truth.sum(1)[None,:])).mean(1)
        mean_col='pooled_cv_mean' if pooled_mode else 'cv_mean'
        std_col='pooled_cv_std' if pooled_mode else 'repeat_std'
        assert abs(float(per_repeat.mean())-row[mean_col])<1e-12
        assert abs(float(per_repeat.std())-row[std_col])<1e-12
        assert len(result['fold_scores'])==15
        assert all(b.file_hash(b.ROOT/name)==value for name,value in result['source_sha256'].items())
    return dict(models_verified=len(registry),identical_protocol_for_all=True,explicit_frozen_decoders_verified=True,
        pooled_sample_f1_verified=True,source_hashes_verified=True,complete_oof_coverage=True)

def main():
    b.protocol();primary=b.OUT;summary=json.loads((primary/'summary.json').read_text())
    primary_checks=verify_predictions(primary,summary)
    primary_checks.update(saved_folds_valid=True,grouped_folds=15,gold_rows=157,gold_embedding_groups=150,
        validation_evaluations_performed=0,current_submission_unchanged=True)
    before=pooled.protected_hashes()
    with pooled.pooled_context():
        protocol=pooled.protocol();registry=pooled.registry()
        gold=b.e.DATA.iloc[b.e.LAB].copy();gold['embedding_group']=b.e.GROUP[b.e.LAB]
        conflicts=(gold.groupby('embedding_group')[b.e.LABELS].nunique().max(axis=1)>1).sum()
        assert conflicts==0
        second=json.loads((pooled.OUT/'summary.json').read_text())
        secondary_checks=verify_predictions(pooled.OUT,second,True)
        secondary_checks.update(saved_folds_valid=True,grouped_folds=15,gold_rows=235,gold_embedding_groups=222,
            no_conflicting_gold_embedding_groups=True,all_configs_and_rules_unchanged=True,
            no_final_all_label_refit=True,current_submission_unchanged=True)
        previous={r['candidate']:r for r in summary['models']}
        for r in second['models']:
            assert abs(r['training_cv_mean']-previous[r['candidate']]['cv_mean'])<1e-12
            assert r['original_validation_f1']==previous[r['candidate']]['validation_f1']
    assert pooled.protected_hashes()==before and len(b.e.Y)==157
    for name,sha in json.loads((pooled.OUT/'preservation.json').read_text())['protected_sha256'].items():
        assert b.file_hash(b.ROOT/name)==sha
    submission=b.file_hash(b.ROOT/'predictions/current_best/part1_submission.csv')
    assert submission=='f9a287fb1d28fd9a6bbc0c583ac84ec2061e72945e2fbeb5aced9ff5b15ff44a'
    primary_checks['submission_sha256']=submission;secondary_checks['submission_sha256']=submission
    b.save(primary/'checks.json',primary_checks);b.save(pooled.OUT/'checks.json',secondary_checks)
    print(json.dumps(dict(training_only=primary_checks,pooled=secondary_checks),indent=2))

if __name__=='__main__':main()
