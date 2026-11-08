"""Verify exact prepared bytes and scores without refitting or uploading."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;STUDY=HERE.parent/'partial_secondary_v1'
sys.path.insert(0,str(STUDY))
import numpy as np
import pandas as pd
import study_run as study


def main():
    r=study.read(HERE/'prepared_submission.json');p=study.read(HERE/'export_protocol.json')
    z=study.loadnp(HERE/'final_predictions.npz');csv=pd.read_csv(study.P1/r['csv'])
    sample=pd.read_csv(study.P1/'data/part1_sample_submission.csv')
    assert study.sha(study.P1/r['csv'])==r['sha256']
    assert study.sha(HERE/'final_predictions.npz')==r['prediction_sha256']
    np.testing.assert_allclose(z['scores'],.8*z['partial_scores']+.2*z['cand2_scores'],atol=0,rtol=0)
    labels=(z['scores']>=.3).astype(int)
    labels[np.arange(len(labels)),z['scores'].argmax(1)]=1
    np.testing.assert_array_equal(labels,z['predictions']);assert (labels.sum(1)>0).all()
    expected=[' '.join(name for name,on in zip(study.LABELS,y) if on) for y in labels]
    assert expected==csv.Predicted.tolist()
    assert len(csv)==312 and csv.ID.is_unique
    assert csv.ID.tolist()==sample.ID.tolist()==z['ids'].tolist()
    d=pd.read_csv(STUDY/'results/summary/overview.csv').query("recipe=='binary_unknown'").set_index('pool')
    for k,key in [('train','train'),('pooled','pooled'),('validation','validation'),('chronology','chronology98')]:
        np.testing.assert_allclose(p['known_metrics'][k],d.loc[key,'mean_sample_f1'],atol=1e-12,rtol=0)
    for name,digest in p['files'].items():assert study.sha(study.P1/name)==digest
    study.guard()
    print(dict(passed=True,rows=312,csv_sha256=r['sha256'],models_fitted=0,uploaded=False))


if __name__=='__main__':main()
