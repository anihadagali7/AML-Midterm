"""Independent final fixed-blend byte, decoder and fitting-boundary checks."""
import numpy as np
import pandas as pd
from scipy.special import expit,logit
import export_fixed as export
study=export.study
HERE=export.HERE


def main():
    study.guard();r=study.read(HERE/'prepared_submission.json');protocol=study.read(HERE/'export_protocol.json')
    assert protocol['files']==export.sources()
    assert study.sha(HERE/'export_protocol.json')==r['export_protocol_sha256']
    assert study.sha(HERE/'final_predictions.npz')==r['prediction_sha256']
    assert study.sha(HERE/'fit_audit.json')==r['fit_audit_sha256']
    assert study.sha(export.CSV)==r['sha256']
    z=study.loadnp(HERE/'final_predictions.npz')
    expected=.8*z['anchored_scores']+.2*z['diverse_scores']
    np.testing.assert_array_equal(expected,z['scores'])
    np.testing.assert_allclose(expit(logit(np.clip(expected,1e-6,1-1e-6))-logit(.3)),z['decision_scores'],atol=1e-12,rtol=0)
    pred=(expected>=.3).astype(int);pred[np.arange(len(pred)),expected.argmax(1)]=1
    np.testing.assert_array_equal(pred,z['predictions'])
    assert (pred.sum(1)>0).all()
    table=pd.read_csv(export.CSV);sample=pd.read_csv(export.P1/'data/part1_sample_submission.csv')
    assert table.columns.tolist()==['ID','Predicted'] and len(table)==312 and table.ID.is_unique
    assert table.ID.tolist()==sample.ID.tolist()==z['ids'].tolist()
    assert table.Predicted.tolist()==[' '.join(label for label,on in zip(study.LABELS,row) if on) for row in pred]
    prior=study.loadnp(HERE/'parent_final_components.npz')
    parent=export.frame(z['ids'],z['parent_predictions'])
    assert export.hashlib.sha256(parent.to_csv(index=False,lineterminator='\n').encode()).hexdigest()==export.PARENT_SHA
    np.testing.assert_array_equal(prior['predictions'],z['parent_predictions'])
    changed=(pred!=z['parent_predictions']).any(1)
    assert int(changed.sum())==r['changed_label_sets']
    changes=pd.read_csv(HERE/'changed_predictions.csv')
    assert changes.ID.tolist()==z['ids'][changed].tolist()
    assert changes.parent.tolist()==parent.loc[changed,'Predicted'].tolist()
    assert changes.candidate.tolist()==table.loc[changed,'Predicted'].tolist()
    train=pd.read_csv(export.P1/'data/part1_train.csv');val=pd.read_csv(export.P1/'data/part1_val.csv')
    fit=pd.concat([train,val],ignore_index=True);audit=study.read(HERE/'fit_audit.json')
    assert study.idhash(fit.ID)==audit['fit_ids_sha256']
    assert set(fit.ID).isdisjoint(z['ids'])
    n=0
    for i,(_,_,part,held,seed) in enumerate(study.inner_inputs(fit,270301)):
        a=audit['inner'][i]
        assert study.idhash(part.ID)==a['fitting_ids_sha256'] and study.idhash(held.ID)==a['query_ids_sha256'] and seed==a['seed']
        n+=1
        for j,(_,_,sub,sh,ss) in enumerate(study.inner_inputs(part,270302)):
            b=a['error_training_subfolds'][j]
            assert study.idhash(sub.ID)==b['fitting_ids_sha256'] and study.idhash(sh.ID)==b['query_ids_sha256'] and ss==b['seed']
            n+=1
    overview=pd.read_csv(export.STUDY/'results/summary/overview.csv').query("recipe=='blend_diverse30'").set_index('pool')
    for key,value in protocol['known_metrics'].items():np.testing.assert_allclose(value,overview.loc[key,'mean_sample_f1'],atol=1e-12,rtol=0)
    receipt=dict(passed=True,rows=312,csv_sha256=r['sha256'],nested_boundaries_checked=n,
        full235_fit_checked=True,exact_parent_scored_csv_reproduced=True,models_fitted=0,uploaded=False)
    study.save(HERE/'verification.json',receipt);print(receipt)


if __name__=='__main__':main()
