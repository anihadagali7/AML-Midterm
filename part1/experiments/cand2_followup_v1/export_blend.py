"""Prepare the fixed 80/20 exploratory blend; never tune on test predictions."""
import io
from datetime import datetime,timezone
import hashlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from run import HERE,P1,LABELS,EMB,AnchoredReasonClassifier,select_reasons,freeze,read,save,sha


def csv_frame(ids,pred):
    return pd.DataFrame(dict(ID=ids,Predicted=[' '.join(l for l,yes in zip(LABELS,row) if yes) for row in pred]))


def main():
    freeze()
    csv=P1/'submissions/candidates/anchored80_cand2_inductive20.csv'
    assert not csv.exists(),'Do not overwrite an exported artifact.'
    original=read(HERE/'cand2_final_provenance.json')
    assert sha(HERE/'cand2_final_predictions.npz')==original['source_prediction_sha256']
    save(HERE/'blend_export_protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        recipe='80% anchored raw decision scores + 20% strict-inductive five-seed Cand2 label marginals; .3+argmax',
        purpose='Concrete next exploratory direction after reported Cand2 result and fixed blend diagnostic; not a promoted leader.',
        status='Fails full both-pool repeat-win rule; no Kaggle outcome or upload.',
        settings_unchanged_from_scored_protocol=sha(HERE/'protocol.json'),
        builder_sha256=sha(__file__),cand2_prediction_sha256=original['source_prediction_sha256'],
        parent_source_sha256=sha(P1/'anchored_solution.py'),
        access='Inductive final235; original 1014 masked training rows; test inputs prediction only. No new Cand2 fitting.',
        source_prediction_record='cand2_final_provenance.json'))
    frame=pd.concat([pd.read_csv(P1/'data/part1_train.csv'),pd.read_csv(P1/'data/part1_val.csv')],ignore_index=True)
    test=pd.read_csv(P1/'data/part1_test.csv')
    with threadpool_limits(limits=1):
        parent=AnchoredReasonClassifier().fit(frame[EMB].to_numpy(float),frame[LABELS].to_numpy(int))
        a=parent.predict_proba(test[EMB].to_numpy(float))
    anchor=csv_frame(test.ID,select_reasons(a))
    expected=pd.read_csv(P1/'submissions/archive/submitted/anchored_reason_trainval235__kaggle_0.89636.csv')
    pd.testing.assert_frame_equal(anchor,expected)
    with np.load(HERE/'cand2_final_predictions.npz') as z:
        assert test.ID.tolist()==z['ids'].tolist()
        cand2=csv_frame(z['ids'],z['predictions'])
        assert hashlib.sha256(cand2.to_csv(index=False,lineterminator='\n').encode()).hexdigest()==original['source_submission_sha256']
        c=z['Q']@z['CM']
    s=.8*a+.2*c;pred=select_reasons(s);out=csv_frame(test.ID,pred)
    sample=pd.read_csv(P1/'data/part1_sample_submission.csv')
    assert out.ID.tolist()==sample.ID.tolist() and out.ID.is_unique and len(out)==312
    assert (pred.sum(1)>0).all() and np.isfinite(s).all()
    csv.parent.mkdir(parents=True,exist_ok=True);out.to_csv(csv,index=False,lineterminator='\n')
    np.savez_compressed(HERE/'blend_final_predictions.npz',ids=test.ID.to_numpy(str),anchored_scores=a,cand2_scores=c,scores=s,predictions=pred)
    record=dict(status='prepared exploratory candidate; not uploaded; Kaggle score unknown',
        csv=str(csv.relative_to(P1)),sha256=sha(csv),rows=312,gold=235,masked=1014,test_features_used_for_fit=False,
        changed_vs_anchored=int((out.Predicted!=anchor.Predicted).sum()),changed_vs_cand2=int((out.Predicted.to_numpy()!=cand2.Predicted.to_numpy()).sum()),
        parent_scored_predictions_reproduced=True,cand2_exact_scored_csv_reconstructed=True,
        final_prediction_sha256=sha(HERE/'blend_final_predictions.npz'),export_protocol_sha256=sha(HERE/'blend_export_protocol.json'))
    save(HERE/'blend_prepared_submission.json',record);freeze();print(record)


if __name__=='__main__':main()
