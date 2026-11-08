"""Prepare one fixed exploratory partial-Other candidate; never upload or tune."""
import os
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[v]='1'
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;P1=HERE.parents[1];STUDY=HERE.parent/'partial_secondary_v1'
sys.path[:0]=[str(STUDY),str(P1)]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import study_run as study
from partial_model import PartialOtherAnchored
from anchored_solution import AnchoredReasonClassifier,select_reasons

CSV=P1/'submissions/candidates/partial_other_unknown80_cand2_inductive20_trainval235.csv'


def csv_frame(ids,pred):
    return pd.DataFrame(dict(ID=ids,Predicted=[' '.join(label for label,yes in zip(study.LABELS,row) if yes) for row in pred]))


def file_hashes():
    files=[Path(__file__),HERE/'component_provenance.json',HERE/'parent_final_components.npz',
        STUDY/'protocol.json',STUDY/'partial_model.py',STUDY/'results/summary/verification.json',
        P1/'anchored_solution.py']+[P1/'data'/f'part1_{name}.csv' for name in ['train','val','test','sample_submission']]
    return {str(p.relative_to(P1)):study.sha(p) for p in files}


def main():
    study.guard();assert not CSV.exists(),'Never overwrite an exported submission.'
    assert not (HERE/'export_protocol.json').exists(),'Use a new preparation version.'
    provenance=study.read(HERE/'component_provenance.json')
    assert provenance['source_sha256']==study.sha(HERE/'parent_final_components.npz')
    study.save(HERE/'export_protocol.json',dict(registered_at=datetime.now(timezone.utc).isoformat(),
        recipe='80% PartialOtherAnchored(binary_unknown) native scores +20% exact final235 five-seed strict-inductive Cand2 marginals; .3 cutoff and argmax.',
        source_benchmark_commit='18af826a',source_benchmark_protocol_sha256=study.sha(STUDY/'protocol.json'),
        trigger='User asked whether an exploratory Kaggle test is worthwhile with five submissions remaining. Assistant recommended one trial and prepared this local artifact; no upload authorized or performed.',
        selection='Full binary_unknown variant chosen for its .901020 chronology result and larger recovery of Other, before computing new test predictions. No hyperparameter, threshold, weight or per-test-row choice follows this selection.',
        exception='Exploratory preparation after completed negative grouped CV. This does not change the original failed advancement status or promote the model. Current best reported score remains .89935.',
        known_metrics=dict(train=.9203821656050954,pooled=.9149822695035461,validation=.8910256410256411,chronology=.9010204081632653),
        fitting='235 supplied gold +1014 originally masked rows in original train-then-validation order. Test inputs only enter prediction; IDs only align outputs. Refit altered anchored and unchanged anchored control. Reuse exact existing Cand2 final fit.',
        output=str(CSV.relative_to(P1)),files=file_hashes()))
    train=pd.read_csv(P1/'data/part1_train.csv');val=pd.read_csv(P1/'data/part1_val.csv')
    fit=pd.concat([train,val],ignore_index=True);test=pd.read_csv(P1/'data/part1_test.csv')
    sample=pd.read_csv(P1/'data/part1_sample_submission.csv')
    X,Y,Q=fit[study.EMB].to_numpy(float),fit[study.LABELS].to_numpy(int),test[study.EMB].to_numpy(float)
    assert int((Y>=0).all(1).sum())==235 and int((Y==-100).all(1).sum())==1014
    assert set(fit.ID).isdisjoint(test.ID)
    z=study.loadnp(HERE/'parent_final_components.npz')
    np.testing.assert_array_equal(z['ids'],test.ID)
    with threadpool_limits(limits=1):
        control=AnchoredReasonClassifier().fit(X,Y);a=control.predict_proba(Q)
        np.testing.assert_allclose(a,z['anchored_scores'],atol=1e-7,rtol=0)
        parent=csv_frame(test.ID,select_reasons(.8*a+.2*z['cand2_scores']))
        parent_bytes=parent.to_csv(index=False,lineterminator='\n').encode()
        assert hashlib.sha256(parent_bytes).hexdigest()==provenance['parent_submission_sha256']
        model=PartialOtherAnchored('binary_unknown').fit(X,Y)
        raw=model.predict_proba(Q)
    scores=.8*raw+.2*z['cand2_scores'];pred=select_reasons(scores)
    assert np.isfinite(scores).all() and (pred.sum(1)>0).all()
    out=csv_frame(test.ID,pred)
    assert len(out)==312 and out.ID.is_unique and out.ID.tolist()==sample.ID.tolist()
    CSV.parent.mkdir(parents=True,exist_ok=True);out.to_csv(CSV,index=False,lineterminator='\n')
    np.savez_compressed(HERE/'final_predictions.npz',ids=test.ID.to_numpy(str),partial_scores=raw,
        cand2_scores=z['cand2_scores'],scores=scores,predictions=pred,parent_predictions=z['predictions'])
    changed=(pred!=z['predictions']).any(1)
    pd.DataFrame(dict(ID=test.ID[changed],parent=parent.Predicted[changed],candidate=out.Predicted[changed])).to_csv(HERE/'changed_predictions.csv',index=False)
    record=dict(status='Prepared exploratory candidate; not uploaded; Kaggle score unknown',
        csv=str(CSV.relative_to(P1)),sha256=study.sha(CSV),rows=len(out),gold=235,masked=1014,
        test_inputs_used_for_fit=False,source_benchmark_recipe='binary_unknown',source_benchmark_commit='18af826a',
        parent_exact_scored_csv_reproduced=True,parent_max_score_error=float(np.abs(a-z['anchored_scores']).max()),
        changed_label_sets=int(changed.sum()),added_labels={label:int(((pred[:,j]==1)&(z['predictions'][:,j]==0)).sum()) for j,label in enumerate(study.LABELS)},
        removed_labels={label:int(((pred[:,j]==0)&(z['predictions'][:,j]==1)).sum()) for j,label in enumerate(study.LABELS)},
        final_rounds=model.round_audit_,prediction_sha256=study.sha(HERE/'final_predictions.npz'),
        export_protocol_sha256=study.sha(HERE/'export_protocol.json'))
    study.save(HERE/'prepared_submission.json',record)
    assert study.read(HERE/'export_protocol.json')['files']==file_hashes()
    study.guard();print(record)


if __name__=='__main__':main()
