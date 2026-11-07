"""Faster recency in Ani's bases and fold-local recent/context neighbors."""
import os
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
import argparse
import gc
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from ani_models import AniFeatures, BASES, PHASES, fit_base, predict_base
from improvement_models import recency_weights, meta_inputs
from extension_common import OUT, register, previous_packet, controls, summary, resolve
from experiments import load_data
from local_benchmark import predict_and_score

FAST={"fast15":1.5,"fast2":2.}
NEIGHBORS=[dict(view="criteria",k=k,tau=None,power=1) for k in [10,50,200]]+[
    dict(view="criteria",k=50,tau=3.,power=1),dict(view="criteria",k=50,tau=2.,power=1),
    dict(view="criteria",k=50,tau=3.,power=2),dict(view="context",k=50,tau=3.,power=1),
    dict(view="context",k=200,tau=3.,power=1)]
RECIPES={}
for bank,tau in FAST.items():
    for trend in [False,True]:
        name=bank+("_trend" if trend else "")
        RECIPES[name]=dict(bank=bank,tau=tau,trend=trend,C=1.)
        RECIPES[name+"_half"]=dict(blend={"ani_stack":.5,name:.5})
for ix in [1,3,4,5,6]:
    RECIPES[f"replace_knn{ix}"]=dict(bank="recent",tau=3.,trend=False,C=1.,replace=ix)
for bank,tau in [("recent",3.),("unweighted",None),("fast2",2.)]:
    name=f"{bank}_neighbors"
    RECIPES[name]=dict(bank=bank,tau=tau,trend=False,C=.3,neighbors=True)
    RECIPES[name+"_trend"]=dict(bank=bank,tau=tau,trend=True,C=.3,neighbors=True)
    if bank!="unweighted":
        RECIPES[name+"_half"]=dict(blend={"ani_stack":.5,name:.5})


def neighbor_predictions(fa,fb,train,ev):
    predictions=np.zeros((len(ev),len(NEIGHBORS)))
    y=train.label.to_numpy();years=train.start_year.fillna(train.start_year.median()).to_numpy()
    for phase in PHASES:
        a=np.flatnonzero(train.phase.eq(phase));b=np.flatnonzero(ev.phase.eq(phase))
        if not len(b):continue
        assert len(a)>0
        criterion=(fb["knn"][b]@fa["knn"][a].T).toarray()
        context=.5*criterion
        for field,scale in [("disease",.2),("drug",.2),("icd",.1)]:
            context+=scale*(fb[field][b]@fa[field][a].T).toarray()
        for j,spec in enumerate(NEIGHBORS):
            sims=criterion if spec["view"]=="criteria" else context
            k=min(spec["k"],len(a))
            indices=np.argpartition(-sims,k-1,axis=1)[:,:k]
            sim=np.take_along_axis(sims,indices,axis=1)
            if spec["power"]>1:
                sim=(sim/np.maximum(sim.max(axis=1,keepdims=True),1e-12))**spec["power"]
            w=np.ones(len(a)) if spec["tau"] is None else np.exp((years[a]-years[a].max())/spec["tau"])
            w=w/w.mean()
            prior=(w@y[a]+1)/(w.sum()+2)
            weights=sim*w[indices]
            predictions[b,j]=((weights*y[a][indices]).sum(axis=1)+2*prior)/(weights.sum(axis=1)+2)
    assert np.isfinite(predictions).all() and ((predictions>0)&(predictions<1)).all()
    return predictions


def fit_extended(train,ev,old,space=None,retain=False,progress=None):
    strict=space is None
    outer=AniFeatures().fit(train) if strict else space
    w={k:recency_weights(train,tau,outer.year_median) for k,tau in FAST.items()}
    fast={k:dict(oof=np.zeros((len(train),8)),eval=np.zeros((len(ev),8))) for k in FAST}
    neighbors=dict(oof=np.zeros((len(train),len(NEIGHBORS))),eval=np.zeros((len(ev),len(NEIGHBORS))))
    audits=[]
    for fold,(a,b) in enumerate(GroupKFold(5).split(train,groups=train.trial_id)):
        assert not set(train.trial_id.iloc[a])&set(train.trial_id.iloc[b])
        features=AniFeatures().fit(train.iloc[a]) if strict else outer
        if strict:
            assert set(features.fit_ids)==set(train.ID.iloc[a])
            assert not set(features.fit_ids)&(set(ev.ID)|set(train.ID.iloc[b]))
        fa,fb=features.transform(train.iloc[a]),features.transform(train.iloc[b])
        for j,(name,blocks,C) in enumerate(BASES):
            for key in FAST:
                if name=="criteria kNN":
                    fast[key]["oof"][b,j]=old["bank"]["recent"]["oof"][b,j];continue
                m=fit_base(name,blocks,C,fa,train.iloc[a],w[key][a])
                fast[key]["oof"][b,j]=predict_base(name,blocks,m,fb,train.iloc[b])
        neighbors["oof"][b]=neighbor_predictions(fa,fb,train.iloc[a],train.iloc[b])
        audits.append(dict(fold=fold,fit_ids=train.ID.iloc[a].tolist(),eval_ids=train.ID.iloc[b].tolist(),
                           preprocessing_fit_ids=features.fit_ids))
        del features,fa,fb,m;gc.collect()
        if progress:progress(fold+1)
    fa,fb=outer.transform(train),outer.transform(ev)
    models={k:[] for k in FAST}
    for j,(name,blocks,C) in enumerate(BASES):
        for key in FAST:
            if name=="criteria kNN":
                fast[key]["eval"][:,j]=old["bank"]["recent"]["eval"][:,j]
                if retain:models[key].append(fit_base(name,blocks,C,fa,train,w[key]))
                continue
            m=fit_base(name,blocks,C,fa,train,w[key]);fast[key]["eval"][:,j]=predict_base(name,blocks,m,fb,ev)
            if retain:models[key].append(m)
    neighbors["eval"]=neighbor_predictions(fa,fb,train,ev)
    packet=dict(fast=fast,neighbors=neighbors,audit=audits,train_ids=train.ID.tolist(),eval_ids=ev.ID.tolist())
    if retain:packet.update(features=outer,models=models,neighbor_train_features=fa,neighbor_train=train)
    return packet


def x_meta(old,extra,df,side,recipe):
    bank=dict(old["bank"]);bank.update(extra["fast"])
    if "replace" in recipe:
        bank=dict(bank);bank[recipe["bank"]]={k:v.copy() for k,v in bank[recipe["bank"]].items()}
        bank[recipe["bank"]][side][:,7]=extra["neighbors"][side][:,recipe["replace"]]
    x=meta_inputs(bank,side,df,dict(banks=[recipe["bank"]],trend=recipe["trend"]),old["year_median"],old["year_cap"])
    if recipe.get("neighbors"):
        z=np.clip(extra["neighbors"][side],1e-4,1-1e-4);x=np.column_stack([x,np.log(z/(1-z))])
    return x


def fit_recipes(old,extra,train,ev,baseline):
    predictions=dict(baseline);models={}
    for name,recipe in RECIPES.items():
        if "blend" in recipe:
            predictions[name]=sum(w*predictions[k] for k,w in recipe["blend"].items());continue
        xt,xe=[x_meta(old,extra,d,s,recipe) for d,s in [(train,"oof"),(ev,"eval")]]
        w=recency_weights(train,recipe["tau"],old["year_median"])
        model=LogisticRegression(C=recipe["C"],max_iter=3000).fit(xt,train.label,sample_weight=w)
        predictions[name]=model.predict_proba(xe)[:,1];models[name]=model
    return predictions,models


def run(track):
    universe,protocol,out=register("neighbors",dict(models=RECIPES,neighbors=NEIGHBORS,fast=FAST),["extension_neighbors.py"])
    folder=out/track;folder.mkdir(exist_ok=True)
    cache=OUT/"cache"/out.name/track;cache.mkdir(parents=True,exist_ok=True)
    space=AniFeatures().fit(pd.concat(list(load_data().values()),ignore_index=True)) if track=="published" else None
    for split in protocol["splits"]:
        if (folder/f"scores_{split['name']}.csv").exists():continue
        start=time.time();train,ev=resolve(universe,split);old=previous_packet(track,split)
        assert old["training_ids"]==train.ID.tolist()
        path=cache/f"{split['name']}.joblib"
        if path.exists():extra=joblib.load(path)
        else:
            extra=fit_extended(train,ev,old,space=space,
                progress=lambda f:print(track,split["name"],"inner",f,round(time.time()-start,1),flush=True))
            joblib.dump(extra,path,compress=3)
        predictions,_=fit_recipes(old,extra,train,ev,controls(track,split,ev))
        predict_and_score(folder,split,ev,predictions)
        arrays=folder/"bases";arrays.mkdir(exist_ok=True)
        np.savez_compressed(arrays/f"{split['name']}.npz",**{k+"_"+s:v for k,p in extra["fast"].items() for s,v in p.items()},
            **{"neighbors_"+s:v for s,v in extra["neighbors"].items()})
        (arrays/f"{split['name']}_audit.json").write_text(json.dumps(extra["audit"]))
        print(track,split["name"],"complete",round(time.time()-start,1),flush=True)
    summary(folder,universe,protocol,RECIPES)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--track",choices=["strict","published"],default="strict")
    run(parser.parse_args().track)
