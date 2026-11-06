"""Both Ani winners and a frozen, small proposal batch on the existing protocol."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from ani_models import AniFeatures
from benchmark_protocol import initialize, resolve, digest, DIRECTORY
from experiments import ROOT, load_data
from improvement_models import RECIPES, TAUS, fit_bank, fit_recipes, replay
from local_benchmark import predict_and_score, summarize

OUT = ROOT / "experiments/stack_improvements"


def setup():
    universe, protocol = initialize()
    manifest = dict(protocol_sha256=digest(DIRECTORY / "protocol.json"), recipes=RECIPES,
                    banks=TAUS, versions={p: importlib.metadata.version(p) for p in
                    ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]},
                    source_sha256={f: digest(ROOT / f) for f in ["improvement_models.py",
                    "improvement_benchmark.py", "ani_models.py", "benchmark_protocol.py",
                    "local_benchmark.py", "experiments.py"]},
                    references={"ani_first": {"kaggle": .58810, "commit": "628088a3be8050b0a20d8719991ec8f517ad8649"},
                                "ani_stack": {"kaggle": .58688, "commit": "1b8bd11c47428fb5453e53cdc275542b92a3b347"}},
                    selection="Freeze all recipes, evaluate all ten saved splits. Prioritize chronological mean and worst window; require corroboration in pooled grouped CV and compare both winners. Bootstrap intervals are conditional exploratory evidence.")
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    out = OUT / "runs" / key
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "current_run.json").write_text(json.dumps(dict(run=key), indent=2))
    return universe, protocol, out


def packet_path(out, track, split):
    cache = OUT / "cache" / out.name / track
    cache.mkdir(parents=True, exist_ok=True)
    return cache / f"{split}.joblib"


def run(track, names=None):
    universe, protocol, run_out = setup()
    out = run_out / track
    out.mkdir(exist_ok=True)
    features = None
    if track == "published":
        data = load_data()
        features = AniFeatures().fit(pd.concat(list(data.values()), ignore_index=True))
    for split in protocol["splits"]:
        if names and split["name"] not in names:
            continue
        target = out / f"scores_{split['name']}.csv"
        if target.exists() and set(pd.read_csv(target).model) == set(RECIPES):
            continue
        start = time.time()
        train, ev = resolve(universe, split)
        cache = packet_path(run_out, track, split["name"])
        if cache.exists():
            packet = joblib.load(cache)
        else:
            print(track, split["name"], "fitting", flush=True)
            packet = fit_bank(train, ev, features=features,
                progress=lambda s: print(track, split["name"], s, round(time.time()-start, 1), "s", flush=True))
            joblib.dump(packet, cache, compress=3)
        assert packet["training_ids"] == train.ID.tolist()
        assert packet["evaluation_ids"] == ev.ID.tolist()
        probabilities, _ = fit_recipes(packet, train, ev)
        predict_and_score(out, split, ev, probabilities)
        bank_dir = out / "bases"
        bank_dir.mkdir(exist_ok=True)
        np.savez_compressed(bank_dir / f"{split['name']}.npz", fold_ids=packet["fold_ids"],
                            **{k+"_"+side: v for k, values in packet["bank"].items() for side, v in values.items()})
        audit = {k: v for k, v in packet.items() if k not in ["bank", "fold_ids"]}
        (bank_dir / f"{split['name']}_audit.json").write_text(json.dumps(audit, indent=2))
        print(track, split["name"], "complete", round(time.time()-start, 1), "s", flush=True)
        print(pd.read_csv(target)[["model", "phase_mean_logloss"]].to_string(index=False), flush=True)
    summarize(out, protocol, bootstrap=names is None)


def final_fit():
    universe, _, run_out = setup()
    data = load_data()
    folder = run_out / "final"
    folder.mkdir(exist_ok=True)
    cache = packet_path(run_out, "published", "final")
    if cache.exists():
        packet = joblib.load(cache)
    else:
        space = AniFeatures().fit(pd.concat(list(data.values()), ignore_index=True))
        packet = fit_bank(universe, data["test"], features=space, retain_models=True,
                          progress=lambda s: print("final", s, flush=True))
        joblib.dump(packet, cache, compress=3)
    predictions, metas = fit_recipes(packet, universe, data["test"])
    meta_path = cache.with_name("final_metas.joblib")
    joblib.dump(metas, meta_path, compress=3)
    changed = data["test"].copy()
    changed["ID"] = "unused"
    changed["trial_id"] = "unused"
    changed["label"] = 0
    other = replay(joblib.load(cache), joblib.load(meta_path), changed)
    parity = {}
    for name, p in predictions.items():
        assert np.allclose(other[name], p, rtol=0, atol=1e-12)
        pd.DataFrame({"ID": data["test"].ID, "label": np.clip(p, .001, .999)}).to_csv(folder / f"{name}.csv", index=False)
        if name in ["ani_first", "ani_stack"]:
            file = "part2_first_submission.csv" if name == "ani_first" else "part2_best_submission.csv"
            reference = pd.read_csv(ROOT / "experiments/ani_comparison/source" / file)
            assert reference.ID.equals(data["test"].ID)
            parity[name] = dict(source_sha256=digest(ROOT / "experiments/ani_comparison/source" / file),
                                max_probability_error=float(np.max(np.abs(reference.label - np.clip(p, .001, .999)))))
    (folder / "parity.json").write_text(json.dumps(parity, indent=2))
    print(json.dumps(parity, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["strict", "published"], default="strict")
    parser.add_argument("--splits", nargs="+")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    final_fit() if args.final else run(args.track, args.splits)
