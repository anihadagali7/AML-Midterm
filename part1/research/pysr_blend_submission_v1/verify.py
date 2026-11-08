"""Reconstruct the final PySR predictions from portable arrays and equations."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.preprocessing import normalize
import sympy as sp
from export import HERE,P1,ROOT,PARENT,FEATURES,LABELS,sha,guard


def verify():
    guard();manifest=json.loads((HERE/'manifest.json').read_text())
    for name,digest in manifest['artifacts'].items():assert sha(HERE/name)==digest,name
    csv=ROOT/manifest['submission_file'];assert sha(csv)==manifest['submission_sha256']
    frame=pd.read_csv(csv);sample=pd.read_csv(P1/'data/part1_sample_submission.csv');test=pd.read_csv(P1/'data/part1_test.csv')
    fit=pd.concat([pd.read_csv(P1/'data/part1_train.csv'),pd.read_csv(P1/'data/part1_val.csv')],ignore_index=True)
    with np.load(HERE/'model_state.npz') as z:state={k:z[k] for k in z.files}
    with np.load(HERE/'prediction_scores.npz') as z:preds={k:z[k] for k in z.files}
    np.testing.assert_array_equal(state['fit_ids'],fit.ID);np.testing.assert_array_equal(frame.ID,sample.ID);np.testing.assert_array_equal(frame.ID,test.ID)
    gold=np.isin(fit[LABELS].to_numpy(),[0,1]).all(1);np.testing.assert_array_equal(gold,state['gold_mask']);assert gold.sum()==235 and (~gold).sum()==1014
    np.testing.assert_array_equal(state['augmented_labels'][gold],fit.loc[gold,LABELS]);assert set(fit.ID).isdisjoint(test.ID)
    assert np.all(state['augmented_labels'].sum(1)>0) and abs(state['sample_weights'][~gold].sum()-23.5)<1e-12
    all_z=normalize((fit[FEATURES].to_numpy(float)-state['embedding_mean'])/state['embedding_scale'])
    np.testing.assert_allclose(all_z[gold],state['bank'],atol=1e-12,rtol=0)
    q=normalize((test[FEATURES].to_numpy(float)-state['embedding_mean'])/state['embedding_scale'])
    # Independently assign nearest centroids, then recompute all five inputs.
    centers=state['cluster_centers'];clusters=np.argmin(((q[:,None,:]-centers[None,:,:])**2).sum(2),axis=1)
    bank=state['bank'];y=state['bank_y'];bank_clusters=state['bank_cluster'];prior=y.mean(0)
    pos=normalize(y.T@bank);neg=normalize((1-y).T@bank);inputs=[]
    for row,k in zip(q,clusters):
        sims=bank@row;ix=np.argsort(-sims,kind='stable')[:10];w=np.maximum(sims[ix],0)**2
        vote=(w@y[ix]+1e-3*prior)/(w.sum()+1e-3)
        rates=(y[bank_clusters==k].sum(0)+prior)/(int((bank_clusters==k).sum())+1)
        margin=(pos-neg)@row;features=[]
        for j in range(5):
            other=np.arange(5)!=j
            features.append([logit(np.clip(vote[j],.01,.99)),margin[j],logit(np.clip(rates[j],.01,.99)),logit(np.clip(vote[other].max(),.01,.99)),margin[other].max()])
        inputs.append(features)
    inputs=(np.asarray(inputs)-state['feature_means'])/state['feature_scales']
    formulas=json.loads((HERE/'equations.json').read_text());symbols=sp.symbols(formulas['feature_names']);raw=[]
    for j,e in enumerate(formulas['equations']):
        expression=sp.sympify(e['expression'],locals=dict(zip(formulas['feature_names'],symbols)))
        fn=sp.lambdify(symbols,expression,modules='numpy');values=fn(*inputs[:,j].T)
        raw.append(expit(np.broadcast_to(values,(312,))))
    raw=np.column_stack(raw);np.testing.assert_allclose(raw,preds['pysr_raw'],atol=1e-10,rtol=0)
    with np.load(PARENT/'prediction_scores.npz') as z:
        np.testing.assert_array_equal(z['ids'],frame.ID);np.testing.assert_array_equal(z['scores'],preds['parent_scores'])
    raw=np.clip(raw,1e-9,1-1e-9);centered=.7*raw/(.7*raw+.3*(1-raw));scores=.8*preds['parent_scores']+.2*centered
    np.testing.assert_allclose(scores,preds['scores'],atol=1e-10,rtol=0)
    pred=(scores>=.5).astype(int)
    for i in range(312):
        if not pred[i].any():pred[i,scores[i].argmax()]=1
    np.testing.assert_array_equal(pred,preds['predictions'])
    assert frame.Predicted.tolist()==[' '.join(label for label,v in zip(LABELS,row) if v) for row in pred]
    assert len(frame)==312 and frame.ID.is_unique and frame.Predicted.str.len().gt(0).all()
    changed=int((pred!=preds['parent_predictions']).any(1).sum());assert changed==manifest['changed_labelsets_vs_89850']
    result=dict(passed=True,rows=312,verified_training_labels=235,originally_masked_rows=1014,independent_feature_equation_and_csv_reconstruction=True,
                maximum_pysr_score_error=float(abs(raw-preds['pysr_raw']).max()),changed_labelsets_vs_89850=changed,submission_sha256=sha(csv))
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':verify()
