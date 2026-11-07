"""Export an evaluated sprint recipe and independently replay native inference."""
import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(key, "2")
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from ani_models import BASES, predict_base
from benchmark_protocol import initialize, digest
from experiments import ROOT, load_data
from extension_neighbors import RECIPES as NEIGHBOR_RECIPES, neighbor_predictions, x_meta, fit_recipes as fit_neighbors
from improvement_models import RECIPES as ORIGINAL_RECIPES, fit_recipes as fit_original, meta_inputs, recency_weights
from sprint_models import RECIPES, design
from sprint_benchmark import current, OUT

BUNDLE = ROOT / "experiments/ani_extensions/cache/decision/final/submission_bundle.joblib"
LEGACY = OUT / "cache/legacy_full.joblib"


def original_components(bundle, old, extra, df):
    p = {name: bundle["base_metas"][name].predict_proba(meta_inputs(old["bank"], "eval", df,
            ORIGINAL_RECIPES[name], old["year_median"], old["year_cap"]))[:, 1]
         for name in ["ani_stack", "ani_first", "ani_phase3_trend"]}
    for name, recipe in NEIGHBOR_RECIPES.items():
        if "blend" in recipe: p[name] = sum(w*p[k] for k,w in recipe["blend"].items())
        else: p[name] = bundle["neighbor_metas"][name].predict_proba(x_meta(old, extra, df, "eval", recipe))[:, 1]
    return dict(file10=p["fast2_neighbors_half"], ani_stack=p["ani_stack"], ani_first=p["ani_first"],
        sharp=p["replace_knn5"], trend=p["ani_phase3_trend"], fast2_neighbors=p["fast2_neighbors"],
        recent_half=p["recent_neighbors_half"], unweighted_nn=p["unweighted_neighbors"])


def native_packets(bundle, df):
    source=bundle["old"]; features=source["features"].transform(df)
    banks={key:{"eval":np.column_stack([predict_base(name, blocks, model, features, df)
        for (name,blocks,_),model in zip(BASES,models)])} for key,models in source["models"].items()}
    old=dict(bank=banks,year_median=source["year_median"],year_cap=source["year_cap"])
    extra=bundle["extra"]; fe=extra["features"].transform(df)
    new=dict(fast={key:{"eval":np.column_stack([predict_base(name,blocks,model,fe,df)
        for (name,blocks,_),model in zip(BASES,models)])} for key,models in extra["models"].items()},
        neighbors={"eval":neighbor_predictions(extra["neighbor_train_features"],fe,extra["neighbor_train"],df)})
    return old,new


def infer(packet, df, native=True):
    bundle=joblib.load(BUNDLE)
    old,extra=native_packets(bundle,df) if native else (bundle["old"],bundle["extra"])
    p=original_components(bundle,old,extra,df)
    recipe=packet["recipe"]
    if any(k.startswith("legacy") for k in recipe.get("blend",{})):
        saved=joblib.load(LEGACY)
        for name,sha in saved["source_sha256"].items(): assert digest(ROOT/name)==sha
        legacy=saved["model"].predict(df)
        p.update(legacy01=legacy["01_balanced_ensemble"],legacy03=legacy["03_phase_specialized"])
    if "blend" in recipe: result=sum(w*p[key] for key,w in recipe["blend"].items())
    else: result=.5*p["ani_stack"]+.5*packet["meta"].predict_proba(design(old,extra,df,"eval",recipe))[:,1]
    return np.clip(result,.001,.999)


def export(name, filename):
    out=current(); train,_=initialize(); test=load_data()["test"]
    recipe=RECIPES[name]
    tables={t:pd.read_csv(out/f"summary_{t}.csv").set_index("model") for t in ["strict","published"]}
    assert name in tables["strict"].index
    bundle=joblib.load(BUNDLE)
    assert bundle["old"]["training_ids"]==bundle["extra"]["train_ids"]==train.ID.tolist()
    packet=dict(name=name,recipe=recipe,training_ids=train.ID.tolist(),meta=None)
    if "blend" not in recipe:
        x=design(bundle["old"],bundle["extra"],train,"oof",recipe)
        packet["meta"]=LogisticRegression(C=recipe["C"],max_iter=3000).fit(x,train.label,
            sample_weight=recency_weights(train,recipe["tau"],bundle["old"]["year_median"]))
    if any(k.startswith("legacy") for k in recipe.get("blend",{})):
        legacy=joblib.load(LEGACY)
        assert legacy["model"].training_ids==train.ID.tolist()
    cache=OUT/"cache"/out.name;cache.mkdir(parents=True,exist_ok=True)
    model_path=cache/f"{name}.joblib";joblib.dump(packet,model_path,compress=3)
    direct=infer(packet,test,native=False)
    changed=test.copy();changed["label"]=0;changed["ID"]="unused";changed["trial_id"]="unused"
    replay=infer(joblib.load(model_path),changed,native=True)
    error=float(np.max(np.abs(direct-replay)));assert error<1e-12,error
    # Confirm the winner remains the same input artifact.
    control=original_components(bundle,bundle["old"],bundle["extra"],test)["file10"]
    saved=pd.read_csv(ROOT/"submissions/10_ani_fast2_neighbors_half.csv")
    assert saved.ID.tolist()==test.ID.tolist()
    assert np.max(np.abs(saved.label-np.clip(control,.001,.999)))<1e-14
    sample=pd.read_csv(ROOT/"part2_sample_submission.csv")
    frame=pd.DataFrame(dict(ID=test.ID,label=direct)).set_index("ID").loc[sample.ID].reset_index()
    assert frame.ID.equals(sample.ID) and frame.ID.is_unique and len(frame)==1500
    assert np.isfinite(frame.label).all() and frame.label.between(0,1,inclusive="neither").all()
    dest=ROOT/"submissions"/filename
    if dest.exists(): assert np.max(np.abs(pd.read_csv(dest).label-frame.label))<1e-14
    else: frame.to_csv(dest,index=False)
    record=dict(file=dest.name,model=name,recipe=recipe,sha256=digest(dest),kaggle_score=None,
        training_rows=len(train),native_replay_max_error=error,baseline_sha256=digest(ROOT/"submissions/10_ani_fast2_neighbors_half.csv"),
        original_bundle_sha256=digest(BUNDLE),model_bundle_path=str(model_path.relative_to(ROOT)),model_bundle_sha256=digest(model_path),
        source_sha256={f:digest(ROOT/f) for f in ["sprint_export.py","sprint_models.py"]},
        benchmark_manifest_sha256=digest(out/"manifest.json"),metrics={t:tables[t].loc[name].to_dict() for t in tables})
    if any(k.startswith("legacy") for k in recipe.get("blend",{})): record["legacy_bundle_sha256"]=digest(LEGACY)
    folder=out/"exports";folder.mkdir(exist_ok=True)
    (folder/(dest.stem+".json")).write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("model",choices=list(RECIPES));parser.add_argument("filename")
    args=parser.parse_args();export(args.model,args.filename)
