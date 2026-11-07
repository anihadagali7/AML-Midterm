"""Source-checked access to the fixed benchmark and previous base predictions."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from benchmark_protocol import initialize, resolve, digest, DIRECTORY
from experiments import ROOT
from improvement_report import audit_track

PREVIOUS = ROOT / "experiments/stack_improvements/runs/097ac8161372126d"
CACHE = ROOT / "experiments/stack_improvements/cache/097ac8161372126d"
OUT = ROOT / "experiments/ani_extensions"
REFERENCES = ["ani_stack", "ani_first", "ani_phase3_trend"]


def register(batch, recipes, files):
    universe, protocol = initialize()
    old = json.loads((PREVIOUS / "manifest.json").read_text())
    for f, expected in old["source_sha256"].items():
        assert digest(ROOT / f) == expected, f
    manifest = dict(batch=batch, recipes=recipes, protocol_sha256=digest(DIRECTORY / "protocol.json"),
        design_sha256=digest(ROOT / "EXTENSION_PROTOCOL.md"),
        source_sha256={f: digest(ROOT / f) for f in files+["extension_common.py"]},
        previous_run_sha256=digest(PREVIOUS / "manifest.json"),
        versions={p:importlib.metadata.version(p) for p in ["numpy","pandas","scipy","scikit-learn","joblib"]})
    key = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:16]
    path = OUT / batch / key
    path.mkdir(parents=True,exist_ok=True)
    (path / "manifest.json").write_text(json.dumps(manifest,indent=2))
    (OUT / f"{batch}_current.json").write_text(json.dumps(dict(directory=str(path.relative_to(ROOT))),indent=2))
    return universe, protocol, path


def current(batch):
    return ROOT / json.loads((OUT / f"{batch}_current.json").read_text())["directory"]


def previous_packet(track, split):
    name = split["name"] if isinstance(split,dict) else split
    if (CACHE / track / f"{name}.joblib").exists():
        return joblib.load(CACHE / track / f"{name}.joblib")
    path = PREVIOUS / track / "bases" / f"{name}.npz"
    arrays = np.load(path)
    audit = json.loads(path.with_name(name+"_audit.json").read_text())
    audit["bank"] = {k:{s:arrays[k+"_"+s] for s in ["oof","eval"]}
                     for k in ["unweighted","recent","moderate"]}
    audit["fold_ids"] = arrays["fold_ids"]
    return audit


def controls(track, split, ev):
    probabilities = {}
    for name in REFERENCES:
        path = PREVIOUS / track / "predictions" / split["name"] / f"{name}.csv"
        data = pd.read_csv(path)
        assert data.ID.tolist() == ev.ID.tolist()
        probabilities[name] = data.probability.to_numpy()
    return probabilities


def summary(folder, universe, protocol, names):
    table, scores, _ = audit_track(folder, universe, protocol, names=REFERENCES+list(names))
    baseline = table.loc["ani_stack"]
    table["temporal_delta"] = table.temporal-baseline.temporal
    table["cv_delta"] = table.cv-baseline.cv
    times = scores[scores.kind.eq("temporal")].pivot(index="split",columns="model",values="phase_mean_logloss")
    table["temporal_wins"] = (times.lt(times.ani_stack,axis=0)).sum(axis=0)
    table["eligible"] = ((table.temporal_delta <= -.001) & (table.temporal_wins >= 2)
        & (table.worst <= baseline.worst+.005) & (table.cv <= baseline.cv+.003))
    table.loc[REFERENCES,"eligible"] = False
    table.sort_values("temporal").to_csv(folder / "summary.csv")
    scores.to_csv(folder / "fold_scores.csv",index=False)
    print(table.sort_values("temporal").to_string(),flush=True)
    return table
