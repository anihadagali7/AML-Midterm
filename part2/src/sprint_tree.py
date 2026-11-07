"""One fixed recent-weighted CatBoost endpoint on the same thirteen splits."""
import os
for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:os.environ.setdefault(key,"4")
import argparse
import hashlib
import importlib.metadata
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from benchmark_protocol import initialize,resolve,digest,DIRECTORY
from experiments import ROOT
from boosted import transform_dense
from improvement_models import recency_weights
from improvement_report import audit_track
from local_benchmark import predict_and_score
from sprint_benchmark import OUT,OLD_ALL,current as sprint_current,DECISION

TREE=dict(iterations=1000,depth=6,learning_rate=.035,l2_leaf_reg=6,loss_function="Logloss",
          random_seed=42,thread_count=4,verbose=False,allow_writing_files=False)
RECIPES={"tree10":dict(tree=.1,linear=0.),"tree20":dict(tree=.2,linear=0.),
         "recent_legacy20":dict(tree=.1,linear=.1),"recent_legacy30":dict(tree=.15,linear=.15)}
NAMES=["file10","file11","recent_tree"]+list(RECIPES)


def register():
    universe,protocol=initialize();gaps=json.loads((DECISION/"gaps/protocol.json").read_text())
    old=json.loads((OLD_ALL/"manifest.json").read_text())
    for name,sha in old["source_sha256"].items():assert digest(ROOT/name)==sha
    manifest=dict(tree=TREE,tau=3.,recipes=RECIPES,source_sha256={name:digest(ROOT/name) for name in ["sprint_tree.py","all_submission_models.py","boosted.py","experiments.py"]},
        policy_sha256=digest(ROOT/"SPRINT_TREE_PROTOCOL.md"),protocol_sha256=digest(DIRECTORY/"protocol.json"),
        gaps_sha256=digest(DECISION/"gaps/protocol.json"),sprint_run=str(sprint_current().relative_to(ROOT)),
        versions={p:importlib.metadata.version(p) for p in ["numpy","pandas","scikit-learn","catboost","joblib"]})
    key=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:16]
    out=OUT/"recent_tree"/key;out.mkdir(parents=True,exist_ok=True)
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (OUT/"recent_tree_current.json").write_text(json.dumps(dict(run=key),indent=2)+"\n")
    return universe,protocol,gaps,out


def dense(model,df):
    blocks=model.features.transform(df)
    return transform_dense(df,blocks,model.svds,model.mask,model.features.year_median).drop(columns="year")


def one(task):
    split,is_gap,out_name=task;out=Path(out_name);parent=out/"gaps" if is_gap else out
    if all((parent/t/f"scores_{split['name']}.csv").exists() for t in ["strict","published"]):return
    started=time.time();universe,_=initialize();train,ev=resolve(universe,split)
    legacy_path=ROOT/"experiments/all_submissions/cache"/OLD_ALL.name/f"{split['name']}.joblib"
    audit_parent=OLD_ALL/"gaps" if is_gap else OLD_ALL
    audit=json.loads((audit_parent/"audits"/f"{split['name']}.json").read_text())
    assert digest(legacy_path)==audit["model_sha256"]
    legacy=joblib.load(legacy_path);assert legacy.training_ids==train.ID.tolist()
    xt,xe=dense(legacy,train),dense(legacy,ev)
    weights=recency_weights(train,3.,legacy.features.year_median)
    cache=OUT/"cache"/out.name;cache.mkdir(parents=True,exist_ok=True)
    path=cache/f"{split['name']}.joblib"
    if path.exists():tree=joblib.load(path)
    else:
        tree=CatBoostClassifier(**TREE).fit(xt,train.label,sample_weight=weights,cat_features=xt.select_dtypes("str").columns.tolist())
        joblib.dump(tree,path,compress=3)
    pt=tree.predict_proba(xe)[:,1];pl=legacy.linear.predict_proba(ev)
    for track in ["strict","published"]:
        source=sprint_current()/"gaps"/track if is_gap else sprint_current()/track
        folder=source/"predictions"/split["name"]
        baseline=pd.read_csv(folder/"file10.csv");previous=pd.read_csv(folder/"mix_legacy01_0.2.csv")
        assert baseline.ID.tolist()==previous.ID.tolist()==ev.ID.tolist()
        p=dict(file10=baseline.probability.to_numpy(),file11=previous.probability.to_numpy(),recent_tree=pt)
        for name,r in RECIPES.items():p[name]=(1-r["tree"]-r["linear"])*p["file10"]+r["tree"]*pt+r["linear"]*pl
        predict_and_score(parent/track,split,ev,p)
    folder=parent/"audits";folder.mkdir(exist_ok=True)
    (folder/f"{split['name']}.json").write_text(json.dumps(dict(train_ids=train.ID.tolist(),eval_ids=ev.ID.tolist(),
        preprocessing_fit_ids=legacy.training_ids,feature_model_sha256=digest(legacy_path),tree_model_sha256=digest(path))))
    print(split["name"],"complete",round(time.time()-started,1),flush=True)


def summarize(out,universe,protocol,gaps):
    for track in ["strict","published"]:
        a,scores,_=audit_track(out/track,universe,protocol,names=NAMES)
        b,gap_scores,_=audit_track(out/"gaps"/track,universe,gaps,names=NAMES)
        table=a.join(b[["temporal","worst"]].rename(columns=dict(temporal="gap_mean",worst="gap_worst")))
        for col in ["official","temporal","worst","cv","gap_mean","gap_worst"]:table[col+"_delta"]=table[col]-table.loc["file10",col]
        table.to_csv(out/f"summary_{track}.csv");scores.to_csv(out/track/"fold_scores.csv",index=False)
        gap_scores.to_csv(out/"gaps"/track/"fold_scores.csv",index=False)
        print(track,table[["official","temporal","gap_mean","cv"]].to_string(),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--workers",type=int,default=2);args=parser.parse_args()
    universe,protocol,gaps,out=register()
    tasks=[(s,False,str(out)) for s in protocol["splits"]]+[(s,True,str(out)) for s in gaps["splits"]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(one,tasks))
    summarize(out,universe,protocol,gaps)
