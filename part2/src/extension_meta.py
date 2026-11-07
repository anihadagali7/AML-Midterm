"""Recent calibration and shallow nonlinear stacks on identical saved OOF bases."""
import os
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
import argparse
import json
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from ani_models import onehot
from improvement_models import recency_weights, meta_inputs
from extension_common import register, previous_packet, controls, summary, resolve
from local_benchmark import predict_and_score

RECIPES = {}
for tau in [1.,1.5,2.]:
    for trend in [False,True]:
        RECIPES[f"meta_tau{tau:g}"+("_trend" if trend else "")] = dict(kind="lr",tau=tau,trend=trend,C=1.)
for drop in [[1,2,4,6],[4],[1,2]]:
    RECIPES["reduced_"+"_".join(map(str,drop))] = dict(kind="lr",tau=3.,trend=False,C=1.,drop=drop)
RECIPES["nonnegative_stack"] = dict(kind="nonnegative",tau=3.,trend=False,C=1.)
RECIPES["nonnegative_trend"] = dict(kind="nonnegative",tau=3.,trend=True,C=1.)
for leaves in [3,7]:
    for year in [False,True]:
        name=f"nonlinear_{leaves}"+("_year" if year else "")
        RECIPES[name] = dict(kind="hgb",tau=3.,trend=False,leaves=leaves,year=year)
        RECIPES[name+"_half"] = dict(blend={"ani_stack":.5,name:.5})


def inputs(packet, df, side, recipe):
    x=meta_inputs(packet["bank"],side,df,dict(banks=["recent"],trend=recipe.get("trend",False)),
                  packet["year_median"],packet["year_cap"])
    if recipe.get("drop"):
        x=np.delete(x,recipe["drop"],axis=1)
    if recipe.get("year"):
        year=np.minimum(df.start_year.fillna(packet["year_median"]),packet["year_cap"]-2)
        x=np.column_stack([x,(year-2007)/3])
    return x


def fit_metas(packet, train, ev, baseline):
    predictions=dict(baseline);models={}
    for name, recipe in RECIPES.items():
        if "blend" in recipe:
            predictions[name]=sum(w*predictions[n] for n,w in recipe["blend"].items());continue
        xt,xe=[inputs(packet,df,side,recipe) for df,side in [(train,"oof"),(ev,"eval")]]
        w=recency_weights(train,recipe["tau"],packet["year_median"])
        if recipe["kind"] == "lr":
            model=LogisticRegression(C=recipe["C"],max_iter=3000).fit(xt,train.label,sample_weight=w)
            p=model.predict_proba(xe)[:,1]
        elif recipe["kind"] == "hgb":
            model=HistGradientBoostingClassifier(max_iter=150,learning_rate=.04,max_leaf_nodes=recipe["leaves"],
                min_samples_leaf=100,l2_regularization=10.,early_stopping=False,random_state=42).fit(xt,train.label,sample_weight=w)
            p=model.predict_proba(xe)[:,1]
        else:
            y=train.label.to_numpy();design=np.column_stack([xt,np.ones(len(xt))])
            def objective(beta):
                z=design@beta
                loss=(w*(np.logaddexp(0,z)-y*z)).sum()+.5*np.dot(beta[:-1],beta[:-1])/recipe["C"]
                grad=design.T@(w*(expit(z)-y));grad[:-1]+=beta[:-1]/recipe["C"]
                return loss,grad
            opt=minimize(objective,np.zeros(design.shape[1]),jac=True,method="L-BFGS-B",
                         bounds=[(0,None)]*8+[(None,None)]*(design.shape[1]-8),
                         options=dict(maxiter=3000,ftol=1e-12,gtol=1e-7))
            assert opt.success,opt.message
            model=opt.x;p=expit(np.column_stack([xe,np.ones(len(xe))])@model)
        predictions[name]=p;models[name]=model
    return predictions,models


def run(track):
    universe,protocol,out=register("meta",RECIPES,["extension_meta.py"])
    folder=out/track;folder.mkdir(exist_ok=True)
    for split in protocol["splits"]:
        if (folder/f"scores_{split['name']}.csv").exists():continue
        train,ev=resolve(universe,split);packet=previous_packet(track,split)
        assert packet["training_ids"]==train.ID.tolist()
        preds,_=fit_metas(packet,train,ev,controls(track,split,ev))
        predict_and_score(folder,split,ev,preds)
        print(track,split["name"],"complete",flush=True)
    summary(folder,universe,protocol,RECIPES)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--track",choices=["strict","published"],default="strict")
    run(parser.parse_args().track)
