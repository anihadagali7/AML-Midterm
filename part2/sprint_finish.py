"""Nine bounded final-slot variations, frozen before file 15's Kaggle result."""
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from benchmark_protocol import initialize,resolve,digest,DIRECTORY
from experiments import ROOT,load_data
from improvement_report import audit_track,paired_interval
from local_benchmark import predict_and_score
from sprint_refine import load_components,native_components,blend,DEPENDENCIES,current as refine_current
from sprint_benchmark import OUT,DECISION
from sprint_export import BUNDLE,LEGACY
from sprint_tree_export import current as tree_current

CONTROLS={"file11":dict(legacy=.2),"file13":dict(legacy=.3),"file15":dict(legacy=.2,tree=.1,trend=.1)}
RECIPES={f"path{a:g}":dict(legacy=.3-.1*a,tree=.1*a,trend=.1*a) for a in [.25,.5,.75,1.25,1.5]}
for t in [.05,.15]:RECIPES[f"trend{t:g}"]=dict(legacy=.2,tree=.1,trend=t)
for t in [.075,.125]:RECIPES[f"tree{t:g}"]=dict(legacy=.2,tree=t,trend=.1)
NAMES=list(CONTROLS)+list(RECIPES)


def register():
    universe,protocol=initialize();gaps=json.loads((DECISION/"gaps/protocol.json").read_text())
    previous=refine_current();old=json.loads((previous/"manifest.json").read_text())
    for file,sha in old["source_sha256"].items():assert digest(ROOT/file)==sha
    manifest=dict(recipes=RECIPES,controls=CONTROLS,source_sha256={f:digest(ROOT/f) for f in DEPENDENCIES+["sprint_finish.py"]},
        policy_sha256=digest(ROOT/"FINISH_PROTOCOL.md"),protocol_sha256=digest(DIRECTORY/"protocol.json"),
        gaps_sha256=digest(DECISION/"gaps/protocol.json"),parent=str(previous.relative_to(ROOT)),
        parent_manifest_sha256=digest(previous/"manifest.json"),file15_score_at_declaration=None,
        file15_score_before_evaluation=.58492,
        controls_sha256={name:digest(ROOT/"submissions"/file) for name,file in [
            ("file11","11_file10_legacy20.csv"),("file13","13_file10_legacy30.csv"),("file15","15_file10_legacy20_tree10_trend10.csv")]})
    key=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:16]
    out=OUT/"finish"/key;out.mkdir(parents=True,exist_ok=True)
    path=out/"manifest.json"
    if path.exists():assert json.loads(path.read_text())==manifest
    else:path.write_text(json.dumps(manifest,indent=2)+"\n")
    (OUT/"finish_current.json").write_text(json.dumps(dict(run=key),indent=2)+"\n")
    return universe,protocol,gaps,out


def current():return OUT/"finish"/json.loads((OUT/"finish_current.json").read_text())["run"]


def benchmark():
    universe,protocol,gaps,out=register()
    for track in ["strict","published"]:
        all_scores=[];tables=[]
        for is_gap,spec in [(False,protocol),(True,gaps)]:
            folder=out/"gaps"/track if is_gap else out/track
            for split in spec["splits"]:
                train,ev=resolve(universe,split);p,hashes=load_components(track,split,is_gap,ev)
                predictions={name:blend(p,r) for name,r in (CONTROLS|RECIPES).items()}
                parent=refine_current()/"gaps"/track if is_gap else refine_current()/track
                for name,alias in [("file13","file13"),("file15","legacy0.2_tree0.1_trend0.1")]:
                    frame=pd.read_csv(parent/"predictions"/split["name"]/f"{alias}.csv")
                    assert frame.ID.tolist()==ev.ID.tolist()
                    assert np.max(np.abs(frame.probability-predictions[name]))<4e-15
                predict_and_score(folder,split,ev,predictions)
                audits=folder/"audits";audits.mkdir(exist_ok=True)
                (audits/f"{split['name']}.json").write_text(json.dumps(dict(train_ids=train.ID.tolist(),eval_ids=ev.ID.tolist(),component_sha256=hashes))+"\n")
            summary,scores,_=audit_track(folder,universe,spec,names=NAMES)
            scores.to_csv(folder/"fold_scores.csv",index=False);all_scores.append(scores);tables.append(summary)
        table=tables[0].join(tables[1][["temporal","worst"]].rename(columns=dict(temporal="gap_mean",worst="gap_worst")))
        latest=all_scores[0][all_scores[0].split.eq("time_2011_to_2012_2013")].set_index("model").phase_mean_logloss
        table["latest"]=latest
        for ref in ["file13","file15"]:
            for key in ["official","temporal","gap_mean","gap_worst","cv","latest"]:
                table[key+"_vs_"+ref]=table[key]-table.loc[ref,key]
        table.to_csv(out/f"summary_{track}.csv")
        print(track,table[["temporal","gap_mean","cv","temporal_vs_file15","gap_mean_vs_file15","gap_worst_vs_file15"]].to_string(),flush=True)
    print(out,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=["benchmark"])
    args=parser.parse_args();benchmark()
