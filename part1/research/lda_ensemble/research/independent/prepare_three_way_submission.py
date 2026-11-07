"""Export the frozen 70/15/15 ensemble after its final 235-gold-label refit.

No tuning or upload. Historical validation belongs to the train157 fit, not to
this final model. Test inputs remain prediction queries throughout fitting.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import run_ani_improvements as study

r,b,e,models=study.r,study.b,study.e,study.models
ROOT=r.ROOT
OUT=ROOT/'predictions/three_way_trainval235'
SUBMISSION=ROOT/'predictions/part1_three_way_trainval235.csv'


def main():
    study.guard();before=study.protected()
    selected_path=study.OUT/'all_four_checks_candidate.json'
    selected=json.loads(selected_path.read_text());candidate=selected['candidate']
    registered=next(c for c in study.registry() if c['name']==candidate['name'])
    assert candidate==registered
    assert candidate['config']['mode']=='blend'
    assert candidate['config']['weights']==dict(ani=.7,tri=.15,svm=.15)
    assert candidate['rule']=='t0.5'
    assert candidate['config']['source_sha256']==study.own_sources()
    OUT.mkdir(exist_ok=True)
    recipe=dict(candidate=candidate,final_gold_labels=235,original_train_gold=157,
        former_validation_gold=78,original_masked_rows=1014,
        test_inputs_used_for_fit=False,final_refit=True,
        source_sha256={**r.sources(),**study.own_sources()},
        exporter_sha256=b.file_hash(Path(__file__)),selected_recipe_sha256=b.file_hash(selected_path),
        protected_sha256=before)
    lock=OUT/'frozen_recipe.json'
    if lock.exists():assert json.loads(lock.read_text())==recipe
    else:r.save(lock,recipe)

    train,val,full=r.frames()
    test=pd.read_csv(ROOT/'data/part1_test.csv')
    sample=pd.read_csv(ROOT/'data/part1_sample_submission.csv')
    assert len(test)==312 and test.ID.is_unique
    assert test.ID.tolist()==sample.ID.tolist()
    assert not set(test.ID)&set(full.ID)
    assert not set(e.LABELS)&set(test.columns)
    fit_audit=[];ani_audit=[]
    with threadpool_limits(limits=1),warnings.catch_warnings(),r.data_context(full):
        warnings.simplefilter('error',ConvergenceWarning)
        assert len(e.Y)==235 and len(e.UNLAB)==1014 and len(e.DATA)==1249
        gold_ids=set(e.DATA.iloc[e.LAB].ID);masked_ids=set(e.DATA.iloc[e.UNLAB].ID)
        original_features=e.features;original_native=models.native_variant

        def guarded_features(indices,query,config):
            fitting=e.DATA.iloc[indices]
            assert fitting.ID.is_unique
            assert set(fitting.ID)<=set(full.ID)
            assert not set(fitting.ID)&set(test.ID)
            fit_audit.append(dict(fitting_rows=len(fitting),
                fitting_gold=len(set(fitting.ID)&gold_ids),fitting_masked=len(set(fitting.ID)&masked_ids),
                query_rows=len(query),test_inputs_excluded=True))
            return original_features(indices,query,config)

        def guarded_native(pos,query,config,excluded=None):
            p=original_native(pos,query,config,excluded)
            audit=models.LAST_AUDIT.copy()
            assert set(audit['gold_ids'])==gold_ids
            assert set(audit['masked_ids'])==masked_ids
            assert not (set(audit['gold_ids'])|set(audit['masked_ids']))&set(test.ID)
            ani_audit.append(audit)
            return p

        e.features=guarded_features;models.native_variant=guarded_native
        try:
            models.clear()
            scores=e.predict(np.arange(235),test,candidate['config'])
            models.clear()
            check=e.predict(np.arange(235),test.iloc[:3],candidate['config'])
            np.testing.assert_allclose(check,scores[:3],rtol=0,atol=1e-10)
        finally:
            e.features=original_features;models.native_variant=original_native;models.clear()
        pd.testing.assert_frame_equal(e.DATA,full)
    assert scores.shape==(312,5) and np.isfinite(scores).all()
    hard=e.decisions(scores,candidate['rule'])
    assert set(np.unique(hard))<={0,1} and (hard.sum(axis=1)>0).all()
    submission=pd.DataFrame(dict(ID=test.ID,
        Predicted=[' '.join(label for label,flag in zip(e.LABELS,row) if flag) for row in hard]))
    assert list(submission.columns)==list(sample.columns)==['ID','Predicted']
    payload=submission.to_csv(index=False).encode()
    if SUBMISSION.exists():assert SUBMISSION.read_bytes()==payload,'Refusing to replace different submission bytes.'
    else:SUBMISSION.write_bytes(payload)
    reread=pd.read_csv(SUBMISSION);pd.testing.assert_frame_equal(reread,submission)
    decoded=np.array([[int(label in value.split()) for label in e.LABELS] for value in reread.Predicted])
    np.testing.assert_array_equal(decoded,hard)
    assert reread.Predicted.map(lambda value:set(value.split())<=set(e.LABELS)).all()
    score_frame=pd.DataFrame(scores,columns=['score_'+label for label in e.LABELS])
    score_frame.insert(0,'ID',test.ID);score_frame.to_csv(OUT/'test_scores.csv',index=False)
    assert before==study.protected();study.guard()
    manifest=dict(created_at=b.now(),status='Ready for manual Kaggle upload; not submitted',
        candidate=candidate['name'],config=candidate['config'],decision_rule=candidate['rule'],
        supervised_gold_rows=235,masked_training_rows=1014,submission_rows=312,
        final_train_plus_validation_refit=True,test_inputs_used_for_fit=False,
        independent_validation_of_refit=None,historical_157_label_validation_f1=selected['results']['validation'],
        development_evidence=selected['results'],frozen_recipe_sha256=b.file_hash(lock),
        submission_file=str(SUBMISSION.relative_to(ROOT)),submission_sha256=b.file_hash(SUBMISSION),
        csv_columns_correct=True,sample_id_order_matches=True,all_ids_unique=True,
        all_labels_valid_and_nonempty=True,small_batch_refit_predictions_match=True,
        original_data_gold_labels_and_prior_submissions_unchanged=True,
        fitting_ids=full.ID.tolist(),gold_ids=sorted(gold_ids),fit_audit=fit_audit,
        test_predicted_label_counts=dict(zip(e.LABELS,hard.sum(axis=0).astype(int).tolist())))
    r.save(OUT/'ani_fit_audit.json',ani_audit)
    r.save(OUT/'manifest.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['fitting_ids','gold_ids','fit_audit','development_evidence']},indent=2))
    print('READY:',SUBMISSION)


if __name__=='__main__':main()
