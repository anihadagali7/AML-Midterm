"""Full-data fit and native replay for the fixed recent-tree sprint endpoint."""
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from benchmark_protocol import initialize,digest
from experiments import ROOT,load_data
from improvement_models import recency_weights
from sprint_tree import TREE,RECIPES,dense
from sprint_benchmark import OUT
from sprint_export import BUNDLE,LEGACY,native_packets,original_components


def current():
    return OUT/"recent_tree"/json.loads((OUT/"recent_tree_current.json").read_text())["run"]


def infer(packet,df,native):
    bundle=joblib.load(BUNDLE);legacy=joblib.load(LEGACY)["model"]
    old,extra=native_packets(bundle,df) if native else (bundle["old"],bundle["extra"])
    base=original_components(bundle,old,extra,df)["file10"]
    tree=packet["tree"].predict_proba(dense(legacy,df))[:,1]
    linear=legacy.linear.predict_proba(df)
    recipe=packet["recipe"]
    return np.clip((1-recipe["tree"]-recipe["linear"])*base+recipe["tree"]*tree+recipe["linear"]*linear,.001,.999)


def export(name,filename):
    out=current();train,_=initialize();test=load_data()["test"]
    for track in ["strict","published"]:assert (out/f"summary_{track}.csv").exists(),"Finish every fixed split before export"
    manifest=json.loads((out/"manifest.json").read_text())
    for file,sha in manifest["source_sha256"].items():assert digest(ROOT/file)==sha
    legacy_packet=joblib.load(LEGACY);legacy=legacy_packet["model"]
    assert legacy.training_ids==train.ID.tolist()
    for file,sha in legacy_packet["source_sha256"].items():assert digest(ROOT/file)==sha
    cache=OUT/"cache"/out.name;cache.mkdir(exist_ok=True)
    path=cache/"final_tree.joblib"
    if path.exists():tree=joblib.load(path)
    else:
        start=time.time();x=dense(legacy,train)
        tree=CatBoostClassifier(**TREE).fit(x,train.label,
            sample_weight=recency_weights(train,3.,legacy.features.year_median),cat_features=x.select_dtypes("str").columns.tolist())
        joblib.dump(tree,path,compress=3);print("Final recent tree fitted",round(time.time()-start,1),flush=True)
    packet=dict(tree=tree,recipe=RECIPES[name],training_ids=train.ID.tolist())
    packet_path=cache/f"{name}_final.joblib";joblib.dump(packet,packet_path,compress=3)
    p=infer(packet,test,False)
    changed=test.copy();changed["ID"]="unused";changed["trial_id"]="unused";changed["label"]=0
    q=infer(joblib.load(packet_path),changed,True);error=float(np.max(np.abs(p-q)));assert error<1e-12
    sample=pd.read_csv(ROOT/"part2_sample_submission.csv")
    frame=pd.DataFrame(dict(ID=test.ID,label=p)).set_index("ID").loc[sample.ID].reset_index()
    assert frame.ID.equals(sample.ID) and frame.ID.is_unique and len(frame)==1500
    assert np.isfinite(frame.label).all() and frame.label.between(0,1,inclusive="neither").all()
    dest=ROOT/"submissions"/filename
    if dest.exists():assert np.allclose(pd.read_csv(dest).label,frame.label,rtol=0,atol=1e-14)
    else:frame.to_csv(dest,index=False)
    receipt=dict(file=filename,model=name,recipe=RECIPES[name],tree=TREE,tau=3.,training_rows=len(train),
        sha256=digest(dest),kaggle_score=None,native_replay_max_error=error,
        baseline_sha256=digest(ROOT/"submissions/10_ani_fast2_neighbors_half.csv"),
        original_bundle_sha256=digest(BUNDLE),legacy_bundle_sha256=digest(LEGACY),
        native_model_path=str(packet_path.relative_to(ROOT)),native_model_sha256=digest(packet_path),
        source_sha256={file:digest(ROOT/file) for file in ["sprint_tree_export.py","sprint_tree.py","sprint_export.py"]},
        benchmark_manifest_sha256=digest(out/"manifest.json"),
        metrics={t:pd.read_csv(out/f"summary_{t}.csv").set_index("model").loc[name].to_dict() for t in ["strict","published"]})
    exports=out/"exports";exports.mkdir(exist_ok=True)
    (exports/(dest.stem+".json")).write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("model",choices=list(RECIPES));parser.add_argument("filename")
    args=parser.parse_args();export(args.model,args.filename)
