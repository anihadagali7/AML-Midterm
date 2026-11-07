"""Freeze the extension shortlist, then check larger time gaps and final inference."""
import os
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from ani_models import AniFeatures,BASES,predict_base
from benchmark_protocol import initialize,resolve,digest
from experiments import ROOT,load_data,metrics,PHASES
from improvement_models import fit_bank,fit_recipes as baseline_recipes,meta_inputs as baseline_inputs,RECIPES as BASE_RECIPES
from improvement_report import audit_track,paired_interval
from extension_common import OUT,PREVIOUS,CACHE,REFERENCES,current,previous_packet
from extension_meta import RECIPES as META_RECIPES,fit_metas,inputs as meta_inputs
from extension_neighbors import RECIPES as NN_RECIPES,fit_extended,fit_recipes as neighbor_recipes,neighbor_predictions,x_meta
from local_benchmark import predict_and_score

DECISION=OUT/"decision"


def select():
    DECISION.mkdir(parents=True,exist_ok=True)
    tables=[];inputs={};sources={}
    for batch in ["meta","neighbors"]:
        sources[batch]=str(current(batch).relative_to(ROOT))
        for track in ["strict","published"]:
            path=current(batch)/track/"summary.csv"
            data=pd.read_csv(path);assert data.completed_splits.eq(10).all()
            data["batch"]=batch;data["track"]=track;tables.append(data)
            inputs[batch+"/"+track]=digest(path)
    table=pd.concat(tables,ignore_index=True)
    table.to_csv(DECISION/"all_comparisons.csv",index=False)
    possible=table[table.track.eq("strict")&table.eligible].sort_values("temporal")
    shortlist=[]
    for row in possible.itertuples():
        published=table[(table.track=="published")&(table.model==row.model)&(table.batch==row.batch)].iloc[0]
        if published.temporal_delta>=0:continue
        shortlist.append(dict(model=row.model,batch=row.batch,
                              recipe=(META_RECIPES if row.batch=="meta" else NN_RECIPES)[row.model]))
        if len(shortlist)==2:break
    decision=dict(shortlist=shortlist,source_runs=sources,input_hashes=inputs,
                  policy_sha256=digest(ROOT/"EXTENSION_PROTOCOL.md"),
                  status="Frozen before candidate predictions on gapped windows and final fitting")
    path=DECISION/"selection.json"
    if path.exists():assert json.loads(path.read_text())==decision
    else:path.write_text(json.dumps(decision,indent=2))
    print(json.dumps(decision,indent=2),flush=True)


def selection():return json.loads((DECISION/"selection.json").read_text())


def merge_selected_predictions(track,split,ev):
    result={}
    for item in selection()["shortlist"]:
        path=ROOT/selection()["source_runs"][item["batch"]]/track/"predictions"/split["name"]/f"{item['model']}.csv"
        data=pd.read_csv(path);assert data.ID.tolist()==ev.ID.tolist()
        result[item["model"]]=data.probability.to_numpy()
    for name in REFERENCES:
        data=pd.read_csv(PREVIOUS/track/"predictions"/split["name"]/f"{name}.csv")
        assert data.ID.tolist()==ev.ID.tolist();result[name]=data.probability.to_numpy()
    return result


def compare():
    universe,protocol=initialize();names=REFERENCES+[x["model"] for x in selection()["shortlist"]]
    pairs=[]
    for track in ["strict","published"]:
        folder=DECISION/track;folder.mkdir(parents=True,exist_ok=True)
        for split in protocol["splits"]:
            _,ev=resolve(universe,split)
            predict_and_score(folder,split,ev,merge_selected_predictions(track,split,ev))
        table,scores,frames=audit_track(folder,universe,protocol,names=names)
        table.to_csv(folder/"summary.csv");scores.to_csv(folder/"fold_scores.csv",index=False)
        for item in selection()["shortlist"]:
            for reference in REFERENCES:
                for kind in ["temporal","group_cv"]:
                    pairs.append(dict(track=track,**paired_interval(frames,item["model"],reference,kind)))
    pd.DataFrame(pairs).to_csv(DECISION/"paired_comparisons.csv",index=False)
    labels=pd.read_csv(ROOT/"experiments/model_search/reason_labels_for_analysis.csv")
    from error_analysis import errors_by_reason
    rows=[];phase_rows=[]
    for track in ["strict","published"]:
        _,ev=resolve(universe,next(s for s in protocol["splits"] if s["name"]=="official"))
        for name in names:
            pred=pd.read_csv(DECISION/track/"predictions/official"/f"{name}.csv").probability.to_numpy()
            rows.append(errors_by_reason(ev,pred,labels).assign(track=track,model=name))
            m=metrics(ev,pred)
            for phase in PHASES:
                phase_rows.append(dict(track=track,model=name,phase=phase,
                    **{metric:m[phase+"_"+metric] for metric in ["logloss","f1","roc_auc","pr_auc"]}))
    pd.concat(rows,ignore_index=True).to_csv(DECISION/"reason_errors.csv",index=False)
    pd.DataFrame(phase_rows).to_csv(DECISION/"phase_metrics.csv",index=False)
    (DECISION/"reason_provenance.json").write_text(json.dumps(dict(
        source="experiments/model_search/reason_labels_for_analysis.csv",
        sha256=digest(ROOT/"experiments/model_search/reason_labels_for_analysis.csv"),
        use="Joined after predicting; no reason or embedding input was fitted"),indent=2))
    print("Saved paired comparisons and per-phase/reason diagnostics",flush=True)


def gaps(track):
    universe,_=initialize();decision=selection()
    previous=json.loads((PREVIOUS/"robustness/protocol.json").read_text())
    protocol=dict(splits=previous["splits"],selection_sha256=digest(DECISION/"selection.json"),
                  original_protocol_sha256=digest(PREVIOUS/"robustness/protocol.json"))
    root=DECISION/"gaps";root.mkdir(exist_ok=True)
    path=root/"protocol.json"
    if path.exists():assert json.loads(path.read_text())==protocol
    else:path.write_text(json.dumps(protocol,indent=2))
    folder=root/track;folder.mkdir(exist_ok=True)
    cache=OUT/"cache/decision"/track;cache.mkdir(parents=True,exist_ok=True)
    space=AniFeatures().fit(pd.concat(list(load_data().values()),ignore_index=True)) if track=="published" else None
    for split in protocol["splits"]:
        if (folder/f"scores_{split['name']}.csv").exists():continue
        start=time.time();train,ev=resolve(universe,split)
        old_path=CACHE/track/f"{split['name']}.joblib"
        old=joblib.load(old_path) if old_path.exists() else fit_bank(train,ev,features=space)
        assert old["training_ids"]==train.ID.tolist() and old["evaluation_ids"]==ev.ID.tolist()
        base,_=baseline_recipes(old,train,ev);predictions={k:base[k] for k in REFERENCES}
        extra=None
        if any(x["batch"]=="neighbors" for x in decision["shortlist"]):
            path=cache/f"{split['name']}.joblib"
            if path.exists():extra=joblib.load(path)
            else:
                extra=fit_extended(train,ev,old,space=space,
                    progress=lambda f:print(track,split["name"],"inner",f,round(time.time()-start,1),flush=True))
                joblib.dump(extra,path,compress=3)
            ps,_=neighbor_recipes(old,extra,train,ev,base)
            for item in decision["shortlist"]:
                if item["batch"]=="neighbors":predictions[item["model"]]=ps[item["model"]]
        if any(x["batch"]=="meta" for x in decision["shortlist"]):
            ps,_=fit_metas(old,train,ev,base)
            for item in decision["shortlist"]:
                if item["batch"]=="meta":predictions[item["model"]]=ps[item["model"]]
        predict_and_score(folder,split,ev,predictions)
        audits=folder/"audits";audits.mkdir(exist_ok=True)
        (audits/f"{split['name']}.json").write_text(json.dumps(dict(train_ids=train.ID.tolist(),
            eval_ids=ev.ID.tolist(),extra=extra["audit"] if extra is not None else None)))
    names=REFERENCES+[x["model"] for x in decision["shortlist"]]
    table,scores,frames=audit_track(folder,universe,protocol,names=names)
    table.to_csv(folder/"summary.csv");scores.to_csv(folder/"fold_scores.csv",index=False)
    pairs=[paired_interval(frames,item["model"],ref,"temporal") for item in decision["shortlist"] for ref in REFERENCES]
    pd.DataFrame(pairs).to_csv(folder/"paired_comparisons.csv",index=False)
    print(table.to_string(),flush=True)


def infer(bundle,df):
    old=bundle["old"]
    f=old["features"].transform(df)
    banks={k:{"eval":np.column_stack([predict_base(name,blocks,m,f,df)
        for (name,blocks,_),m in zip(BASES,models)])} for k,models in old["models"].items()}
    packet=dict(bank=banks,year_median=old["year_median"],year_cap=old["year_cap"])
    base={name:bundle["base_metas"][name].predict_proba(baseline_inputs(
        banks,"eval",df,BASE_RECIPES[name],old["year_median"],old["year_cap"]))[:,1]
        for name in REFERENCES}
    extra=bundle.get("extra");ps=dict(base)
    if extra is not None:
        fe=extra["features"].transform(df)
        updated=dict(fast={k:{"eval":np.column_stack([predict_base(name,blocks,m,fe,df)
            for (name,blocks,_),m in zip(BASES,models)])} for k,models in extra["models"].items()},
            neighbors={"eval":neighbor_predictions(extra["neighbor_train_features"],fe,extra["neighbor_train"],df)})
        for name,recipe in NN_RECIPES.items():
            if "blend" in recipe:ps[name]=sum(w*ps[k] for k,w in recipe["blend"].items())
            else:ps[name]=bundle["neighbor_metas"][name].predict_proba(x_meta(packet,updated,df,"eval",recipe))[:,1]
    if bundle.get("meta_metas"):
        for item in bundle["decision"]["shortlist"]:
            if item["batch"]=="meta":
                recipe=META_RECIPES[item["model"]]
                assert recipe["kind"]=="lr","Add explicit nonlinear replay before exporting this family"
                ps[item["model"]]=bundle["meta_metas"][item["model"]].predict_proba(meta_inputs(packet,df,"eval",recipe))[:,1]
    return {k:ps[k] for k in REFERENCES+[x["model"] for x in bundle["decision"]["shortlist"]]}


def final():
    decision=selection();universe,_=initialize();test=load_data()["test"]
    for track in ["strict","published"]:assert (DECISION/"gaps"/track/"summary.csv").exists()
    cache=OUT/"cache/decision/final";cache.mkdir(parents=True,exist_ok=True)
    path=CACHE/"published/final.joblib"
    if path.exists():old=joblib.load(path)
    else:
        features=AniFeatures().fit(pd.concat(list(load_data().values()),ignore_index=True))
        old=fit_bank(universe,test,features=features,retain_models=True)
    base,base_metas=baseline_recipes(old,universe,test)
    bundle=dict(old=old,base_metas=base_metas,decision=decision,extra=None,neighbor_metas=None,meta_metas=None)
    direct={k:base[k] for k in REFERENCES}
    if any(x["batch"]=="neighbors" for x in decision["shortlist"]):
        path=cache/"extra.joblib"
        if path.exists():extra=joblib.load(path)
        else:
            extra=fit_extended(universe,test,old,space=old["features"],retain=True,
                               progress=lambda f:print("final neighbor fold",f,flush=True))
            joblib.dump(extra,path,compress=3)
        ps,metas=neighbor_recipes(old,extra,universe,test,base)
        bundle.update(extra=extra,neighbor_metas=metas)
        for item in decision["shortlist"]:
            if item["batch"]=="neighbors":direct[item["model"]]=ps[item["model"]]
    if any(x["batch"]=="meta" for x in decision["shortlist"]):
        ps,metas=fit_metas(old,universe,test,base);bundle["meta_metas"]=metas
        for item in decision["shortlist"]:
            if item["batch"]=="meta":direct[item["model"]]=ps[item["model"]]
    bundle_path=cache/"submission_bundle.joblib";joblib.dump(bundle,bundle_path,compress=3)
    changed=test.copy();changed["ID"]="unused";changed["trial_id"]="unused";changed["label"]=0
    replay=infer(joblib.load(bundle_path),changed)
    errors={k:float(np.max(np.abs(v-replay[k]))) for k,v in direct.items()}
    assert max(errors.values())<1e-12,errors
    sample=pd.read_csv(ROOT/"part2_sample_submission.csv")
    records=[]
    for i,item in enumerate(decision["shortlist"],start=9):
        name=item["model"]
        frame=pd.DataFrame(dict(ID=test.ID,label=np.clip(direct[name],.001,.999))).set_index("ID").loc[sample.ID].reset_index()
        assert frame.ID.equals(sample.ID) and frame.ID.is_unique and len(frame)==1500
        assert np.isfinite(frame.label).all() and frame.label.between(0,1,inclusive="neither").all()
        dest=ROOT/"submissions"/f"{i:02d}_ani_{name}.csv"
        if dest.exists():assert np.allclose(pd.read_csv(dest).label,frame.label,rtol=0,atol=1e-14)
        else:frame.to_csv(dest,index=False)
        gap_deltas={t:float((lambda d:d.loc[name,"temporal"]-d.loc["ani_stack","temporal"])(
            pd.read_csv(DECISION/"gaps"/t/"summary.csv").set_index("model"))) for t in ["strict","published"]}
        records.append(dict(file=dest.name,model=name,sha256=digest(dest),kaggle_score=None,
            training_rows=9050,recipe=item,gapped_deltas=gap_deltas,
            status="Corroborated locally; Kaggle untested" if max(gap_deltas.values())<0 else "Mixed gapped evidence; exploratory only"))
    manifest=dict(submissions=records,selection_sha256=digest(DECISION/"selection.json"),
                  bundle_path=str(bundle_path.relative_to(ROOT)),bundle_sha256=digest(bundle_path),
                  native_replay_errors=errors,preprocessing="Published covariate-only transduction; supervised train+validation refit")
    (ROOT/"submissions/ani_extensions_manifest.json").write_text(json.dumps(manifest,indent=2))
    (DECISION/"final_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=["select","compare","gaps","final"])
    parser.add_argument("--track",choices=["strict","published"],default="strict")
    args=parser.parse_args()
    if args.action=="gaps":gaps(args.track)
    else:{"select":select,"compare":compare,"final":final}[args.action]()
