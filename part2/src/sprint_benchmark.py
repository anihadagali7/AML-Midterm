"""Score the frozen sprint batch using verified existing base-model packets."""
import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(key, "2")
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from benchmark_protocol import initialize, resolve, digest, DIRECTORY
from experiments import ROOT
from extension_common import previous_packet, current as extension_current
from extension_followthrough import DECISION
from extension_neighbors import fit_recipes as fit_neighbors
from improvement_models import fit_recipes as fit_original
from improvement_report import audit_track
from local_benchmark import predict_and_score
from sprint_models import RECIPES, BASELINE, predict_recipes

OUT = ROOT / "experiments/leaderboard_sprint"
OLD_ALL = ROOT / "experiments/all_submissions/runs/8f68385e25e07615"
CONTROLS = [BASELINE, "ani_stack", "ani_first", "sharp", "trend", "legacy01", "legacy03"]


def register():
    universe, protocol = initialize()
    gaps = json.loads((DECISION / "gaps/protocol.json").read_text())
    # Pin original implementations and old packets, without rewriting their runs.
    old = json.loads((ROOT / "experiments/stack_improvements/runs/097ac8161372126d/manifest.json").read_text())
    for name, expected in old["source_sha256"].items(): assert digest(ROOT/name) == expected, name
    for name, expected in json.loads((extension_current("neighbors") / "manifest.json").read_text())["source_sha256"].items():
        assert digest(ROOT/name) == expected, name
    manifest = dict(recipes=RECIPES, baseline="10_ani_fast2_neighbors_half.csv", baseline_score=.58609,
        baseline_sha256=digest(ROOT / "submissions/10_ani_fast2_neighbors_half.csv"),
        policy_sha256=digest(ROOT/"SPRINT_PROTOCOL.md"), protocol_sha256=digest(DIRECTORY/"protocol.json"),
        gaps_sha256=digest(DECISION/"gaps/protocol.json"),
        source_sha256={name:digest(ROOT/name) for name in ["sprint_models.py", "sprint_benchmark.py"]},
        versions={p:importlib.metadata.version(p) for p in ["numpy","pandas","scipy","scikit-learn","joblib"]})
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    out = OUT / "runs" / key; out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (OUT / "current_run.json").write_text(json.dumps(dict(run=key), indent=2)+"\n")
    return universe, protocol, gaps, out


def current():
    return OUT / "runs" / json.loads((OUT / "current_run.json").read_text())["run"]


def packets(track, split, is_gap):
    old = previous_packet(track, split)
    root = ROOT / "experiments/ani_extensions/cache"
    extra_path = root / "decision" / track / f"{split['name']}.joblib" if is_gap else root / extension_current("neighbors").name / track / f"{split['name']}.joblib"
    return old, joblib.load(extra_path), extra_path


def components(old, extra, train, ev, track, split, is_gap):
    base, _ = fit_original(old, train, ev)
    neighbors, _ = fit_neighbors(old, extra, train, ev, base)
    p = dict(file10=neighbors["fast2_neighbors_half"], ani_stack=base["ani_stack"], ani_first=base["ani_first"],
             sharp=neighbors["replace_knn5"], trend=base["ani_phase3_trend"],
             fast2_neighbors=neighbors["fast2_neighbors"], recent_half=neighbors["recent_neighbors_half"],
             unweighted_nn=neighbors["unweighted_neighbors"])
    folder = OLD_ALL / "gaps" / track if is_gap else OLD_ALL / track
    for name, alias in [("legacy01", "01_balanced_ensemble"), ("legacy03", "03_phase_specialized")]:
        d = pd.read_csv(folder / "predictions" / split["name"] / f"{alias}.csv")
        assert d.ID.tolist() == ev.ID.tolist()
        p[name] = d.probability.to_numpy()
    for name, alias in [("file10", "10_ani_fast2_neighbors_half"), ("ani_stack", "ani_stack"), ("sharp", "09_ani_replace_knn5")]:
        d = pd.read_csv(folder / "predictions" / split["name"] / f"{alias}.csv")
        assert d.ID.tolist() == ev.ID.tolist()
        assert np.max(np.abs(d.probability-p[name])) < 1e-10, (track, split["name"], name)
    return p


def summarize(out, universe, protocol, gaps):
    for track in ["strict", "published"]:
        main, scores, _ = audit_track(out/track, universe, protocol, names=CONTROLS+list(RECIPES))
        gap, gap_scores, _ = audit_track(out/"gaps"/track, universe, gaps, names=CONTROLS+list(RECIPES))
        table = main.join(gap[["temporal","worst"]].rename(columns=dict(temporal="gap_mean",worst="gap_worst")))
        baseline = table.loc[BASELINE]
        for key in ["official","temporal","worst","cv","gap_mean","gap_worst"]:
            table[key+"_delta"] = table[key]-baseline[key]
        table["time_average_delta"] = .5*(table.temporal_delta+table.gap_mean_delta)
        table["eligible"] = (table.temporal_delta.lt(0) & table.gap_mean_delta.lt(0) &
            (table.temporal_delta.le(-.0002) | table.gap_mean_delta.le(-.0002)) & table.cv_delta.le(.001))
        table.to_csv(out / f"summary_{track}.csv")
        scores.to_csv(out/track/"fold_scores.csv",index=False)
        gap_scores.to_csv(out/"gaps"/track/"fold_scores.csv",index=False)
        print(track,table.sort_values("time_average_delta")[["temporal","gap_mean","cv","time_average_delta","eligible"]].head(18).to_string(),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--track",choices=["strict","published"],required=True)
    parser.add_argument("--summarize",action="store_true");args=parser.parse_args()
    universe, protocol, gaps, out = register()
    if args.summarize:
        summarize(out,universe,protocol,gaps);return
    for is_gap, spec in [(False,protocol),(True,gaps)]:
        folder=out/"gaps"/args.track if is_gap else out/args.track
        for split in spec["splits"]:
            if (folder/f"scores_{split['name']}.csv").exists():continue
            train,ev=resolve(universe,split);old,extra,extra_path=packets(args.track,split,is_gap)
            assert old["training_ids"]==extra["train_ids"]==train.ID.tolist()
            assert old["evaluation_ids"]==extra["eval_ids"]==ev.ID.tolist()
            p=components(old,extra,train,ev,args.track,split,is_gap)
            candidate,_=predict_recipes(old,extra,train,ev,p)
            predict_and_score(folder,split,ev,{**{k:p[k] for k in CONTROLS},**candidate})
            audits=folder/"audits";audits.mkdir(exist_ok=True)
            (audits/f"{split['name']}.json").write_text(json.dumps(dict(train_ids=train.ID.tolist(),eval_ids=ev.ID.tolist(),
                extra_packet_sha256=digest(extra_path),old_fold_audit=old["audit"],extra_fold_audit=extra["audit"])))
            print(args.track,split["name"],"complete",flush=True)
    print("Finished",args.track,out,flush=True)


if __name__=="__main__":main()
