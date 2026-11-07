"""Frozen mixtures around the 0.58474 winner, with complete paired evaluation."""
import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(key, "2")
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from benchmark_protocol import initialize, resolve, digest, DIRECTORY
from experiments import ROOT, load_data
from improvement_report import audit_track, paired_interval
from local_benchmark import predict_and_score
from sprint_benchmark import OUT, DECISION, current as initial_current
from sprint_tree_export import current as tree_current
from sprint_export import BUNDLE, LEGACY, native_packets, original_components
from sprint_tree import dense

CONTROLS = {
    "file10": {}, "file11": {"legacy": .2},
    "file13": {"legacy": .3}, "file14": {"tree": .2},
}
RECIPES = {}
for l in [.25, .35, .4, .5]:
    RECIPES[f"legacy{l:g}"] = dict(legacy=l)
for l in [.2, .3, .4]:
    for t in [.05, .1, .15]:
        RECIPES[f"legacy{l:g}_tree{t:g}"] = dict(legacy=l, tree=t)
for l in [.2, .3]:
    for t in [.05, .1]:
        RECIPES[f"legacy{l:g}_tree{t:g}_trend0.1"] = dict(legacy=l, tree=t, trend=.1)
for s in [.1, .2]:
    RECIPES[f"legacy0.3_sharp{s:g}"] = dict(legacy=.3, sharp=s)
NAMES = list(CONTROLS) + list(RECIPES)
DEPENDENCIES = ["sprint_refine.py", "sprint_export.py", "sprint_models.py", "sprint_tree.py",
                "all_submission_models.py", "benchmark_models.py", "experiments.py", "boosted.py",
                "ani_models.py", "improvement_models.py", "extension_neighbors.py"]


def blend(p, recipe):
    assert all(w >= 0 for w in recipe.values()) and sum(recipe.values()) <= 1
    return (1-sum(recipe.values()))*p["file10"] + sum(w*p[k] for k,w in recipe.items())


def register():
    universe, protocol = initialize()
    gaps = json.loads((DECISION/"gaps/protocol.json").read_text())
    parents = dict(initial=initial_current(), tree=tree_current())
    for path in parents.values():
        m = json.loads((path/"manifest.json").read_text())
        for file, sha in m["source_sha256"].items():
            assert digest(ROOT/file) == sha, file
    manifest = dict(recipes=RECIPES, controls=CONTROLS,
        baseline=dict(file="11_file10_legacy20.csv", kaggle_score=.58474,
                      sha256=digest(ROOT/"submissions/11_file10_legacy20.csv")),
        source_sha256={f:digest(ROOT/f) for f in DEPENDENCIES},
        policy_sha256=digest(ROOT/"REFINE_PROTOCOL.md"),
        protocol_sha256=digest(DIRECTORY/"protocol.json"),
        gaps_sha256=digest(DECISION/"gaps/protocol.json"),
        parents={k:dict(path=str(v.relative_to(ROOT)), manifest_sha256=digest(v/"manifest.json")) for k,v in parents.items()})
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    out = OUT/"refine"/key; out.mkdir(parents=True, exist_ok=True)
    dest = out/"manifest.json"
    if dest.exists(): assert json.loads(dest.read_text()) == manifest
    else: dest.write_text(json.dumps(manifest, indent=2)+"\n")
    (OUT/"refine_current.json").write_text(json.dumps(dict(run=key), indent=2)+"\n")
    return universe, protocol, gaps, out


def current():
    return OUT/"refine"/json.loads((OUT/"refine_current.json").read_text())["run"]


def load_components(track, split, is_gap, ev):
    parents = [initial_current(), tree_current()]
    parents = [p/"gaps"/track if is_gap else p/track for p in parents]
    mapping = {"file10":(0,"file10"), "legacy":(0,"legacy01"), "sharp":(0,"sharp"),
               "trend":(0,"trend"), "tree":(1,"recent_tree")}
    probabilities = {}; hashes = {}
    for key, (index,name) in mapping.items():
        path = parents[index]/"predictions"/split["name"]/f"{name}.csv"
        frame = pd.read_csv(path)
        for col in ["ID", "trial_id", "phase", "label"]:
            assert frame[col].tolist() == ev[col].tolist(), (key, col)
        probabilities[key] = frame.probability.to_numpy()
        hashes[str(path.relative_to(ROOT))] = digest(path)
    return probabilities, hashes


def benchmark():
    universe, protocol, gaps, out = register()
    for track in ["strict", "published"]:
        for is_gap, spec in [(False,protocol), (True,gaps)]:
            folder = out/"gaps"/track if is_gap else out/track
            for split in spec["splits"]:
                train, ev = resolve(universe, split)
                p, hashes = load_components(track, split, is_gap, ev)
                predictions = {name:blend(p, r) for name,r in (CONTROLS|RECIPES).items()}
                # Independently check the scored winner and previously exported controls.
                original = initial_current()/"gaps"/track if is_gap else initial_current()/track
                for name, alias in [("file11","mix_legacy01_0.2"), ("file13","mix_legacy01_0.3")]:
                    frame = pd.read_csv(original/"predictions"/split["name"]/f"{alias}.csv")
                    assert frame.ID.tolist() == ev.ID.tolist()
                    assert np.max(np.abs(frame.probability-predictions[name])) < 4e-15
                predict_and_score(folder, split, ev, predictions)
                audits = folder/"audits"; audits.mkdir(exist_ok=True)
                (audits/f"{split['name']}.json").write_text(json.dumps(dict(
                    train_ids=train.ID.tolist(), eval_ids=ev.ID.tolist(), component_sha256=hashes))+"\n")
        summarize(out, universe, protocol, gaps, track)
    print("Complete", out, flush=True)


def summarize(out, universe, protocol, gaps, track):
    a,scores,_ = audit_track(out/track, universe, protocol, names=NAMES)
    b,gap_scores,_ = audit_track(out/"gaps"/track, universe, gaps, names=NAMES)
    table = a.join(b[["temporal","worst"]].rename(columns=dict(temporal="gap_mean",worst="gap_worst")))
    for key in ["official","temporal","worst","cv","gap_mean","gap_worst"]:
        table[key+"_delta"] = table[key]-table.loc["file11", key]
    latest = scores[scores.split.eq("time_2011_to_2012_2013")].set_index("model").phase_mean_logloss
    table["latest_delta"] = latest-latest["file11"]
    table["mean_time_delta"] = .5*(table.temporal_delta+table.gap_mean_delta)
    table["eligible"] = (table.temporal_delta.lt(0) & table.gap_mean_delta.lt(0) &
        (table.temporal_delta.le(-.0002)|table.gap_mean_delta.le(-.0002)) &
        table.cv_delta.le(.001) & table.gap_worst_delta.le(.002))
    table.to_csv(out/f"summary_{track}.csv")
    scores.to_csv(out/track/"fold_scores.csv", index=False)
    gap_scores.to_csv(out/"gaps"/track/"fold_scores.csv", index=False)
    print(track,table.sort_values("mean_time_delta")[["temporal","gap_mean","cv","gap_worst_delta","latest_delta","eligible"]].to_string(),flush=True)


def native_components(df, cached):
    bundle = joblib.load(BUNDLE)
    old,extra = (bundle["old"],bundle["extra"]) if cached else native_packets(bundle,df)
    p = original_components(bundle, old, extra, df)
    saved = joblib.load(LEGACY)
    for file,sha in saved["source_sha256"].items(): assert digest(ROOT/file) == sha
    legacy = saved["model"]
    p["legacy"] = legacy.predict(df)["01_balanced_ensemble"]
    tree = joblib.load(OUT/"cache"/tree_current().name/"final_tree.joblib")
    p["tree"] = tree.predict_proba(dense(legacy,df))[:,1]
    return p


def export(name, filename):
    out = current(); universe,_ = initialize(); test = load_data()["test"]
    manifest = json.loads((out/"manifest.json").read_text())
    for file,sha in manifest["source_sha256"].items(): assert digest(ROOT/file) == sha
    assert name in RECIPES
    tables={t:pd.read_csv(out/f"summary_{t}.csv").set_index("model") for t in ["strict","published"]}
    for t,table in tables.items(): assert name in table.index and table.loc[name,"completed_splits"] == 10
    assert joblib.load(LEGACY)["model"].training_ids == universe.ID.tolist()
    original = joblib.load(BUNDLE)
    assert original["old"]["training_ids"] == original["extra"]["train_ids"] == universe.ID.tolist()
    p = native_components(test, cached=True)
    for control,file in [("file10","10_ani_fast2_neighbors_half.csv"),("file11","11_file10_legacy20.csv"),("file13","13_file10_legacy30.csv"),("file14","14_file10_recent_tree20.csv")]:
        saved=pd.read_csv(ROOT/"submissions"/file)
        assert saved.ID.tolist() == test.ID.tolist()
        assert np.max(np.abs(saved.label-np.clip(blend(p,CONTROLS[control]),.001,.999))) < 1e-14
    result=np.clip(blend(p,RECIPES[name]),.001,.999)
    changed=test.copy();changed["ID"]="unused";changed["trial_id"]="unused";changed["label"]=0
    replay=np.clip(blend(native_components(changed,cached=False),RECIPES[name]),.001,.999)
    error=float(np.max(np.abs(result-replay)));assert error<1e-12
    sample=pd.read_csv(ROOT/"part2_sample_submission.csv")
    frame=pd.DataFrame(dict(ID=test.ID,label=result)).set_index("ID").loc[sample.ID].reset_index()
    assert frame.ID.equals(sample.ID) and frame.ID.is_unique and len(frame)==1500
    assert np.isfinite(frame.label).all() and frame.label.between(0,1,inclusive="neither").all()
    dest=ROOT/"submissions"/filename
    if dest.exists(): assert np.max(np.abs(pd.read_csv(dest).label-frame.label))<1e-14
    else: frame.to_csv(dest,index=False)
    dependencies=[BUNDLE,LEGACY,OUT/"cache"/tree_current().name/"final_tree.joblib"]
    receipt=dict(file=filename,model=name,recipe=RECIPES[name],remaining_weight="file10",
        sha256=digest(dest),kaggle_score=None,training_rows=len(universe),native_replay_max_error=error,
        dependencies_sha256={str(p.relative_to(ROOT)):digest(p) for p in dependencies},
        source_sha256=manifest["source_sha256"],benchmark_manifest_sha256=digest(out/"manifest.json"),
        baseline=manifest["baseline"],metrics={t:table.loc[name].to_dict() for t,table in tables.items()})
    folder=out/"exports";folder.mkdir(exist_ok=True)
    (folder/(dest.stem+".json")).write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=["benchmark","export"])
    parser.add_argument("--model",choices=list(RECIPES));parser.add_argument("--filename")
    args=parser.parse_args()
    if args.action=="benchmark": benchmark()
    else:
        assert args.model and args.filename
        export(args.model,args.filename)
