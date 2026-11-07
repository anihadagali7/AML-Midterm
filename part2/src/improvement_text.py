"""Final adaptive batch: add two genuinely different text estimators to Ani OOF.

Declared after the two calibration batches. Uses all original splits, the same
feature boundaries, and the unchanged selection rule. No gapped results seen.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
import argparse
import gc
import json
import time
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from ani_models import AniFeatures, matrix
from benchmark_protocol import initialize, resolve, digest
from experiments import ROOT, load_data
from improvement_models import RECIPES, recency_weights, meta_inputs
from improvement_report import current, audit_track, eligible
from improvement_trends import TRENDS
from improvement_benchmark import packet_path
from local_benchmark import predict_and_score

TEXT_RECIPES = {
    "recent_joint_stack": dict(bank="recent", tau=3., extra=[0], trend=False),
    "recent_ratio_stack": dict(bank="recent", tau=3., extra=[1], trend=False),
    "recent_text_stack": dict(bank="recent", tau=3., extra=[0, 1], trend=False),
    "recent_text_trend": dict(bank="recent", tau=3., extra=[0, 1], trend=True),
    "unweighted_text_stack": dict(bank="unweighted", tau=None, extra=[0, 1], trend=False),
    "unweighted_text_trend": dict(bank="unweighted", tau=None, extra=[0, 1], trend=True),
}
EXTRA_TAUS = {"unweighted": None, "recent": 3.}


def fit_extra(f, df, w, kind):
    if kind == 0:
        x = matrix(f, ["criteria", "disease", "icd", "drug", "targets", "pathways", "actions", "meta"])
        model = LogisticRegression(C=1., max_iter=3000).fit(x, df.label, sample_weight=w)
        return (model, None)
    x = f["crit_counts"].astype(float).copy()
    x.data[:] = 1
    y = df.label.to_numpy()
    positive = 1 + np.asarray(x.T @ (w*y)).ravel()
    negative = 1 + np.asarray(x.T @ (w*(1-y))).ravel()
    ratio = np.log(positive / positive.sum()) - np.log(negative / negative.sum())
    xx = sparse.hstack([x.multiply(ratio), f["phase"]], format="csr")
    model = LogisticRegression(C=.1, solver="liblinear", max_iter=3000, random_state=42).fit(xx, y, sample_weight=w)
    return (model, ratio)


def predict_extra(f, model, kind):
    lr, ratio = model
    if kind == 0:
        x = matrix(f, ["criteria", "disease", "icd", "drug", "targets", "pathways", "actions", "meta"])
    else:
        counts = f["crit_counts"].astype(float).copy()
        counts.data[:] = 1
        x = sparse.hstack([counts.multiply(ratio), f["phase"]], format="csr")
    return lr.predict_proba(x)[:, 1]


def fit_extra_bank(train, ev, space=None, retain=False, progress=None):
    strict = space is None
    outer = AniFeatures().fit(train) if strict else space
    weights = {k: recency_weights(train, tau, outer.year_median) for k, tau in EXTRA_TAUS.items()}
    bank = {k: dict(oof=np.zeros((len(train), 2)), eval=np.zeros((len(ev), 2))) for k in EXTRA_TAUS}
    audit = []
    for fold, (a, b) in enumerate(GroupKFold(5).split(train, groups=train.trial_id)):
        assert not set(train.trial_id.iloc[a]) & set(train.trial_id.iloc[b])
        features = AniFeatures().fit(train.iloc[a]) if strict else outer
        if strict:
            assert set(features.fit_ids) == set(train.ID.iloc[a])
            assert not set(features.fit_ids) & (set(train.ID.iloc[b]) | set(ev.ID))
        fa, fb = features.transform(train.iloc[a]), features.transform(train.iloc[b])
        for key in EXTRA_TAUS:
            for kind in range(2):
                model = fit_extra(fa, train.iloc[a], weights[key][a], kind)
                bank[key]["oof"][b, kind] = predict_extra(fb, model, kind)
        audit.append(dict(fold=fold, fit_ids=train.ID.iloc[a].tolist(), eval_ids=train.ID.iloc[b].tolist(),
                          feature_fit_ids=features.fit_ids, outcome_ratio_fit_ids=train.ID.iloc[a].tolist()))
        del features, fa, fb, model
        gc.collect()
        if progress:
            progress(fold + 1)
    fa, fb = outer.transform(train), outer.transform(ev)
    models = {k: [] for k in EXTRA_TAUS}
    for key in EXTRA_TAUS:
        for kind in range(2):
            model = fit_extra(fa, train, weights[key], kind)
            bank[key]["eval"][:, kind] = predict_extra(fb, model, kind)
            if retain:
                models[key].append(model)
    packet = dict(bank=bank, audit=audit, train_ids=train.ID.tolist(), eval_ids=ev.ID.tolist())
    if retain:
        packet.update(features=outer, models=models)
    return packet


def meta_x(packet, extra, side, df, recipe):
    base = meta_inputs(packet["bank"], side, df, dict(banks=[recipe["bank"]], trend=recipe["trend"]),
                       packet["year_median"], packet["year_cap"])
    z = np.clip(extra["bank"][recipe["bank"]][side][:, recipe["extra"]], 1e-4, 1-1e-4)
    return np.hstack([base, np.log(z / (1-z))])


def fit_text(packet, extra, train, ev):
    predictions, models = {}, {}
    for name, recipe in TEXT_RECIPES.items():
        xt, xe = [meta_x(packet, extra, side, frame, recipe) for side, frame in [("oof", train), ("eval", ev)]]
        w = recency_weights(train, recipe["tau"], packet["year_median"])
        model = LogisticRegression(C=1., max_iter=3000).fit(xt, train.label, sample_weight=w)
        predictions[name], models[name] = model.predict_proba(xe)[:, 1], model
    return predictions, models


def declare():
    out = current() / "text_followup"
    out.mkdir(exist_ok=True)
    manifest = dict(recipes=TEXT_RECIPES, source_sha256=digest(__file__),
        joint_model="L2 logistic C1 on all eight feature blocks plus phase",
        ratio_model="Binary criteria counts times smoothed, class-normalized log-count ratio; logistic C0.1; ratios fit inside each inner fold",
        context="Final adaptive batch after two failed calibration batches, before any gapped-window scores; no further sweep")
    path = out / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        path.write_text(json.dumps(manifest, indent=2))
    return out


def run(track):
    out = declare() / track
    out.mkdir(exist_ok=True)
    universe, protocol = initialize()
    space = None
    if track == "published":
        space = AniFeatures().fit(pd.concat(list(load_data().values()), ignore_index=True))
    for split in protocol["splits"]:
        if (out / f"scores_{split['name']}.csv").exists():
            continue
        start = time.time()
        train, ev = resolve(universe, split)
        packet = joblib.load(packet_path(current(), track, split["name"]))
        cache = packet_path(current(), track, "text_"+split["name"])
        if cache.exists():
            extra = joblib.load(cache)
        else:
            extra = fit_extra_bank(train, ev, space=space,
                progress=lambda f: print(track, split["name"], f, round(time.time()-start, 1), flush=True))
            joblib.dump(extra, cache, compress=3)
        assert extra["train_ids"] == packet["training_ids"]
        assert extra["eval_ids"] == ev.ID.tolist()
        predictions, _ = fit_text(packet, extra, train, ev)
        for name in list(RECIPES)+list(TRENDS):
            d = pd.read_csv(current() / "trend_followup" / track / "predictions" / split["name"] / f"{name}.csv")
            assert d.ID.tolist() == ev.ID.tolist()
            predictions[name] = d.probability.to_numpy()
        predict_and_score(out, split, ev, predictions)
        folder = out / "bases"
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder / f"{split['name']}.npz", **{k+"_"+s: v for k, values in extra["bank"].items() for s,v in values.items()})
        (folder / f"{split['name']}_audit.json").write_text(json.dumps(extra["audit"], indent=2))
        print(track, split["name"], "complete", round(time.time()-start, 1), flush=True)
    names = list(RECIPES)+list(TRENDS)+list(TEXT_RECIPES)
    summary, scores, _ = audit_track(out, universe, protocol, names=names)
    summary.to_csv(out / "independent_summary.csv")
    scores.to_csv(out / "independent_scores.csv", index=False)
    print(summary.sort_values("temporal").to_string(), flush=True)


def select():
    out = current()
    folder = declare()
    summaries = {t: pd.read_csv(folder / t / "independent_summary.csv").set_index("model") for t in ["strict", "published"]}
    possible = eligible(summaries["strict"])
    shortlist = summaries["strict"].loc[possible].sort_values("temporal").head(2).index.tolist()
    decision = dict(shortlist=shortlist, strict_eligible=possible,
                    published_eligible=eligible(summaries["published"]), selection_track="strict",
                    protocol_document_sha256=digest(ROOT / "IMPROVEMENT_PROTOCOL.md"),
                    selection_code_sha256=digest(__file__),
                    input_sha256={t:digest(folder / t / "independent_summary.csv") for t in summaries},
                    status="Frozen across three declared batches before gapped tests and final fitting",
                    known_kaggle_best=dict(model="ani_stack", score=.58688))
    path = out / "selection.json"
    if path.exists():
        assert json.loads(path.read_text()) == decision
    else:
        path.write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))


def final():
    out = current()
    universe, _ = initialize()
    test = load_data()["test"]
    packet = joblib.load(packet_path(out, "published", "final"))
    cache = packet_path(out, "published", "text_final")
    if cache.exists():
        extra = joblib.load(cache)
    else:
        extra = fit_extra_bank(universe, test, space=packet["features"], retain=True,
                               progress=lambda f: print("final text fold", f, flush=True))
        joblib.dump(extra, cache, compress=3)
    predictions, models = fit_text(packet, extra, universe, test)
    meta_path = packet_path(out, "published", "text_metas")
    joblib.dump(models, meta_path, compress=3)
    native = joblib.load(cache)
    models = joblib.load(meta_path)
    changed = test.copy()
    changed["ID"], changed["trial_id"], changed["label"] = "unused", "unused", 0
    features = native["features"].transform(changed)
    regenerated = dict(bank={k: dict(eval=np.column_stack([predict_extra(features, m, i)
        for i, m in enumerate(native["models"][k])])) for k in EXTRA_TAUS})
    for name, p in predictions.items():
        x = meta_x(packet, regenerated, "eval", changed, TEXT_RECIPES[name])
        assert np.allclose(models[name].predict_proba(x)[:,1], p, rtol=0, atol=1e-12)
        pd.DataFrame(dict(ID=test.ID, label=np.clip(p, .001, .999))).to_csv(out / "final" / f"{name}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["declare", "run", "select", "final"])
    parser.add_argument("--track", choices=["strict", "published"], default="strict")
    args = parser.parse_args()
    if args.action == "run":
        run(args.track)
    else:
        {"declare":declare, "select":select, "final":final}[args.action]()
