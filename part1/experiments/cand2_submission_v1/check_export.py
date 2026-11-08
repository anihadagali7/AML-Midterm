"""Independent saved-prediction decoding and export verification; no fitting."""
import json
import numpy as np
import pandas as pd
from build import HERE, P1, CSV, LABELS, freeze, sha, read, save


def main():
    freeze()
    record=read(HERE/'prepared_submission.json')
    assert sha(CSV)==record['sha256']
    assert sha(HERE/'final_predictions.npz')==record['prediction_sha256']
    with np.load(HERE/'final_predictions.npz') as z:
        q,cm=z['Q'],z['CM']
        possible=np.array([[(bits>>j)&1 for j in range(5)] for bits in range(1,32)])
        utility=np.array([[2*np.sum(a*b)/(a.sum()+b.sum()) for b in cm] for a in possible])
        decoded=possible[np.argmax(q@utility.T,axis=1)]
        np.testing.assert_array_equal(decoded,z['predictions'])
        assert np.isfinite(q).all() and np.allclose(q.sum(1),1) and (q>=0).all()
        output=pd.read_csv(CSV)
        sample=pd.read_csv(P1/'data/part1_sample_submission.csv')
        assert output.ID.tolist()==sample.ID.tolist()==z['ids'].tolist()
        assert output.ID.is_unique and len(output)==312
        expected=[' '.join(label for label,yes in zip(LABELS,row) if yes) for row in decoded]
        assert output.Predicted.tolist()==expected
    anchor=pd.read_csv(P1/'submissions/archive/submitted/anchored_reason_trainval235__kaggle_0.89636.csv').set_index('ID')
    diff=sum(set(a.split())!=set(b.split()) for a,b in zip(output.Predicted,anchor.loc[output.ID,'Predicted']))
    assert diff==record['changed_vs_anchored']==49
    for name,expected_hash in read(HERE/'provenance.json')['copied_sha256'].items():
        assert sha(HERE/name)==expected_hash,name
    result=dict(status='passed',source_data_and_artifact_hashes=True,
        independent_expected_f1_decode=True,ids_order_and_nonempty_labels=True,
        rows=312,changed_label_sets_vs_anchored=diff,validation_replay_max_error=record['validation_replay_max_error'],
        kaggle_score=None,uploaded=False)
    save(HERE/'verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
