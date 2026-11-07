"""Predeclared gapped forward checks of frozen shortlist and both Ani references."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from ani_models import AniFeatures
from benchmark_protocol import initialize, resolve, digest
from experiments import load_data, ROOT
from improvement_benchmark import packet_path
from improvement_models import fit_bank, fit_recipes
from improvement_trends import fit_trends
from improvement_text import TEXT_RECIPES, fit_extra_bank, fit_text
from improvement_report import current, audit_track, paired_interval, REFERENCES
from local_benchmark import predict_and_score


def run(track):
    out = current()
    universe, _ = initialize()
    selection = json.loads((out / "selection.json").read_text())
    names = REFERENCES + selection["shortlist"]
    root = out / "robustness"
    root.mkdir(exist_ok=True)
    splits = []
    for cut, start, end in [(2005, 2008, 2009), (2007, 2010, 2011), (2009, 2012, 2013)]:
        year = universe.start_year.fillna(2007)
        train, ev = universe[year <= cut], universe[year.between(start, end)]
        splits.append(dict(name=f"gap_{cut}_to_{start}_{end}", kind="temporal", repeat=None,
                           fit_ids=train.ID.tolist(), eval_ids=ev.ID.tolist(), fit_rows=len(train), eval_rows=len(ev)))
    protocol = dict(splits=splits, selection_sha256=digest(out / "selection.json"),
                    note="Previously consulted labels; larger forecast gaps, not an independent holdout.")
    path = root / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        path.write_text(json.dumps(protocol, indent=2))
    folder = root / track
    folder.mkdir(exist_ok=True)
    space = None
    if track == "published":
        space = AniFeatures().fit(pd.concat(list(load_data().values()), ignore_index=True))
    for split in splits:
        if (folder / f"scores_{split['name']}.csv").exists():
            continue
        train, ev = resolve(universe, split)
        cache = packet_path(out, track, split["name"])
        start = time.time()
        if cache.exists():
            packet = joblib.load(cache)
        else:
            packet = fit_bank(train, ev, features=space,
                progress=lambda s: print(track, split["name"], s, round(time.time()-start, 1), flush=True))
            joblib.dump(packet, cache, compress=3)
        predictions, _ = fit_recipes(packet, train, ev)
        trend_predictions, _ = fit_trends(packet, train, ev)
        predictions.update(trend_predictions)
        if any(name in TEXT_RECIPES for name in names):
            extra_cache = packet_path(out, track, "text_"+split["name"])
            if extra_cache.exists():
                extra = joblib.load(extra_cache)
            else:
                extra = fit_extra_bank(train, ev, space=space)
                joblib.dump(extra, extra_cache, compress=3)
            text_predictions, _ = fit_text(packet, extra, train, ev)
            predictions.update(text_predictions)
        predict_and_score(folder, split, ev, {k: predictions[k] for k in names})
        audit_dir = folder / "audits"
        audit_dir.mkdir(exist_ok=True)
        (audit_dir / f"{split['name']}.json").write_text(json.dumps({k: v for k, v in packet.items()
            if k not in ["bank", "fold_ids"]}, indent=2))
    summary, scores, frames = audit_track(folder, universe, protocol, names=names)
    summary.to_csv(folder / "summary.csv")
    scores.to_csv(folder / "fold_scores.csv", index=False)
    pairs = [paired_interval(frames, name, ref, "temporal") for name in names for ref in REFERENCES if name != ref]
    pd.DataFrame(pairs).to_csv(folder / "paired_comparisons.csv", index=False)
    print(summary.to_string(), flush=True)
    if all((root / t / "summary.csv").exists() for t in ["strict", "published"]):
        (root / "verification.json").write_text(json.dumps(dict(tracks=2, splits_per_track=3, models=names,
            protocol_sha256=digest(path), selection_sha256=digest(out / "selection.json"),
            source_sha256=digest(__file__), complete=True), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["strict", "published"], default="strict")
    run(parser.parse_args().track)
