"""Declared follow-up: explicit capped year effects across recency scales.

This registry was fixed after the first batch's chronological results and first
group-CV folds were inspected. It is an adaptive second batch, not a blind test.
All eight additions reuse the same base predictions and all ten saved splits.
"""
import argparse
import json
import shutil
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from ani_models import onehot
from benchmark_protocol import initialize, resolve, digest
from experiments import ROOT, load_data
from improvement_models import RECIPES, meta_inputs, recency_weights
from improvement_report import current, audit_track, eligible, REFERENCES
from local_benchmark import predict_and_score

TRENDS = {
    "unweighted_phase3_year": dict(bank="unweighted", tau=None, phases="III", scale=1.),
    "unweighted_all_years": dict(bank="unweighted", tau=None, phases="all", scale=1.),
    "unweighted_shrunk_years": dict(bank="unweighted", tau=None, phases="all", scale=.25),
    "tau6_phase3_year": dict(bank="moderate", tau=6., phases="III", scale=1.),
    "tau6_all_years": dict(bank="moderate", tau=6., phases="all", scale=1.),
    "tau6_shrunk_years": dict(bank="moderate", tau=6., phases="all", scale=.25),
    "recent_all_years": dict(bank="recent", tau=3., phases="all", scale=1.),
    "recent_shrunk_years": dict(bank="recent", tau=3., phases="all", scale=.25),
}


def inputs(bank, side, df, recipe, median, cap):
    x = meta_inputs(bank, side, df, dict(banks=[recipe["bank"]]), median, cap)
    # Exactly the original Ani cap and time units, with all phases supported.
    year = (np.minimum(df.start_year.fillna(median).to_numpy(), cap)-2007)/3
    trend = onehot(df) * year[:, None] * recipe["scale"]
    if recipe["phases"] == "III":
        trend = trend[:, 2, None]
    return np.hstack([x, trend])


def fit_trends(packet, train, ev):
    predictions, models = {}, {}
    for name, recipe in TRENDS.items():
        xt = inputs(packet["bank"], "oof", train, recipe, packet["year_median"], packet["year_cap"])
        xe = inputs(packet["bank"], "eval", ev, recipe, packet["year_median"], packet["year_cap"])
        w = recency_weights(train, recipe["tau"], packet["year_median"])
        model = LogisticRegression(C=1., max_iter=3000).fit(xt, train.label, sample_weight=w)
        predictions[name], models[name] = model.predict_proba(xe)[:, 1], model
    return predictions, models


def declare():
    folder = current() / "trend_followup"
    folder.mkdir(exist_ok=True)
    manifest = dict(recipes=TRENDS, source_sha256=digest(__file__),
                    context="Adaptive batch fixed after first-batch chronological results and early CV folds; same selection rule, no new boundaries.")
    path = folder / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest, "Follow-up is frozen; version changes separately"
    else:
        path.write_text(json.dumps(manifest, indent=2))
    return folder


def run():
    folder = declare()
    universe, protocol = initialize()
    for track in ["strict", "published"]:
        dest = folder / track
        dest.mkdir(exist_ok=True)
        for split in protocol["splits"]:
            source = current() / track
            arrays = np.load(source / "bases" / f"{split['name']}.npz")
            audit = json.loads((source / "bases" / f"{split['name']}_audit.json").read_text())
            packet = dict(year_median=audit["year_median"], year_cap=audit["year_cap"],
                          bank={k: {s: arrays[k+"_"+s] for s in ["oof", "eval"]}
                                for k in ["unweighted", "moderate", "recent"]})
            train, ev = resolve(universe, split)
            probabilities, _ = fit_trends(packet, train, ev)
            for name in RECIPES:
                d = pd.read_csv(source / "predictions" / split["name"] / f"{name}.csv")
                assert d.ID.tolist() == ev.ID.tolist()
                probabilities[name] = d.probability.to_numpy()
            predict_and_score(dest, split, ev, probabilities)
        summary, scores, _ = audit_track(dest, universe, protocol, names=list(RECIPES)+list(TRENDS))
        summary.to_csv(dest / "independent_summary.csv")
        scores.to_csv(dest / "independent_scores.csv", index=False)
        print(track, summary.sort_values("temporal").to_string(), flush=True)


def select():
    out = current()
    folder = declare()
    summaries = {t: pd.read_csv(folder / t / "independent_summary.csv").set_index("model")
                 for t in ["strict", "published"]}
    eligible_names = eligible(summaries["strict"])
    shortlist = summaries["strict"].loc[eligible_names].sort_values("temporal").head(2).index.tolist()
    decision = dict(shortlist=shortlist, selection_track="strict", strict_eligible=eligible_names,
                    published_eligible=eligible(summaries["published"]),
                    protocol_document_sha256=digest(ROOT / "IMPROVEMENT_PROTOCOL.md"),
                    selection_code_sha256=digest(__file__),
                    input_sha256={t: digest(folder / t / "independent_summary.csv") for t in summaries},
                    followup_manifest_sha256=digest(folder / "manifest.json"),
                    status="Frozen across both batches before gapped-window checks and final fitting",
                    known_kaggle_best=dict(model="ani_stack", score=.58688))
    path = out / "selection.json"
    if path.exists():
        assert json.loads(path.read_text()) == decision
    else:
        path.write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))


def final():
    import joblib
    from improvement_benchmark import packet_path
    out = current()
    universe, _ = initialize()
    test = load_data()["test"]
    packet = joblib.load(packet_path(out, "published", "final"))
    predictions, models = fit_trends(packet, universe, test)
    changed = test.copy()
    changed["label"] = 1
    changed["ID"] = "unused"
    changed["trial_id"] = "unused"
    other, _ = fit_trends(packet, universe, changed)
    for name, p in predictions.items():
        assert np.array_equal(other[name], p)
        pd.DataFrame(dict(ID=test.ID, label=np.clip(p, .001, .999))).to_csv(out / "final" / f"{name}.csv", index=False)
    joblib.dump(models, packet_path(out, "published", "trend_metas"), compress=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["declare", "run", "select", "final"])
    args = parser.parse_args()
    {"declare": declare, "run": run, "select": select, "final": final}[args.action]()
