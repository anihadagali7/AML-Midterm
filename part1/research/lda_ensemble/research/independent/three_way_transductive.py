"""Explicit target-input fitting for the frozen 70/15/15 three-way recipe.

All evaluation labels are erased before constructing a transductive fitting
pool. Native classifier settings, self-training rounds and decoding stay fixed.
The established inductive engine and its benchmarks are not modified.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import run_ani_improvements as study

base=study.models
r,e=study.r,study.e
CANDIDATE=next(c for c in study.registry() if c['name']=='Ani + tri/SVM 15% each')
LAST_AUDIT={}


def clear():base.clear()


def fitting_frame(pos,excluded=None):
    u=e.UNLAB if excluded is None else e.UNLAB[~np.isin(e.GROUP[e.UNLAB],excluded)]
    rows=np.sort(np.r_[e.LAB[np.asarray(pos)],u])
    frame=e.DATA.iloc[rows].copy().reset_index(drop=True)
    if excluded is not None:assert not set(e.GROUP[rows])&set(excluded)
    return frame


def with_targets(fit,query):
    target=query.drop(columns=e.LABELS,errors='ignore').copy()
    for label in e.LABELS:target[label]=-100
    assert not set(fit.ID)&set(target.ID)
    result=pd.concat([fit,target],ignore_index=True)
    assert result.ID.is_unique
    return result


def ani(fit,query):
    """The native Ani computation on an explicitly supplied fitting frame."""
    known=fit[e.LABELS].ge(0).all(axis=1).to_numpy()
    assert np.logical_or(known,fit[e.LABELS].eq(-100).all(axis=1)).all()
    y=fit.loc[known,e.LABELS].to_numpy(int)
    assert not set(fit.loc[known,'ID'])&set(query.ID)
    x=fit[e.EMB].to_numpy();scaler=StandardScaler().fit(x);z=scaler.transform(x)
    pca=PCA(30,random_state=0).fit(z)
    gmm=GaussianMixture(30,covariance_type='diag',n_init=3,random_state=0).fit(pca.transform(z))
    assert gmm.converged_
    responsibility=gmm.predict_proba(pca.transform(z))
    gold=np.flatnonzero(known);u=np.flatnonzero(~known)
    query_mask=np.isin(fit.iloc[u].ID,query.ID)
    rows=gold.copy();target=y.copy();history=[]
    for iteration in range(2):
        packet=base.base_fit(z[rows],target,np.ones(len(rows)))
        t=target/target.sum(axis=1,keepdims=True)
        rates=(responsibility[rows].T@t+t.mean(axis=0))/(responsibility[rows].sum(axis=0)[:,None]+1)
        lr,knn=base.base_components(packet,z[u])
        prob=.8*(.7*lr+.3*knn)+.2*(responsibility[u]@rates)
        keep=prob.max(axis=1)>=.8
        rows=np.r_[gold,u[keep]];target=np.vstack([y,r.adapters.decide(prob,.3)[keep]])
        history.append(dict(round=iteration+1,pseudo_rows=int(keep.sum()),pseudo_target_rows=int((keep&query_mask).sum())))
    packet=base.base_fit(z[rows],target,np.ones(len(rows)))
    lr,knn=base.base_components(packet,scaler.transform(query[e.EMB].to_numpy()))
    audit=dict(gold_ids=fit.loc[known,'ID'].tolist(),masked_ids=fit.loc[~known,'ID'].tolist(),
        gold_rows=len(gold),masked_rows=len(u),target_rows_in_fit=int(query_mask.sum()),
        target_labels_hidden=True,history=history)
    return .7*lr+.3*knn,audit


def partners_in_frame(fit,query):
    with r.data_context(fit):
        clear();pos=np.arange(len(e.Y))
        tri=base.LEGACY(pos,query,base.TRI)
        svm=base.LEGACY(pos,query,base.SVM)
        assert set(e.DATA.iloc[e.LAB].ID).isdisjoint(query.ID)
    clear()
    return tri,svm


def predict(pos,query,mode,excluded=None):
    global LAST_AUDIT
    query=query.drop(columns=e.LABELS,errors='ignore').copy()
    if mode=='inductive':
        score=base.predict(pos,query,CANDIDATE['config'],excluded)
        LAST_AUDIT=dict(mode=mode,target_inputs_for_fit=False)
        return score
    assert mode in ['ani_transductive','all_transductive','repacked_inductive']
    fit=fitting_frame(pos,excluded)
    trans=fit if mode=='repacked_inductive' else with_targets(fit,query)
    raw,audit=ani(trans,query)
    if mode=='ani_transductive':
        tri=base.LEGACY(pos,query,base.TRI,excluded)
        svm=base.LEGACY(pos,query,base.SVM,excluded)
    else:tri,svm=partners_in_frame(trans,query)
    score=.7*base.centered(raw)+.15*tri+.15*svm
    assert score.shape==(len(query),5) and np.isfinite(score).all()
    LAST_AUDIT=dict(mode=mode,ani=audit,partner_target_access=mode=='all_transductive',
        original_fitting_ids=fit.ID.tolist(),target_ids=query.ID.tolist(),gold_labels_from_target=False)
    return score
