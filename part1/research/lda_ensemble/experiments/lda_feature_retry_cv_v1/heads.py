"""Retest previously promising metadata with the current source and pseudo-labels."""
from pathlib import Path
import sys
import hashlib
import json
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.preprocessing import normalize

BIO = Path(__file__).resolve().parent.parent/"biochemical_features_cv_v1"
sys.path.insert(0, str(BIO))
from representations import BioBlock

SETTINGS = {mode+"_"+view:(mode, view) for mode in ["augmented", "goldhalf"]
            for view in ["none", "drug_targets", "morgan"]}


def representation(packet, fit, query, view, features):
    cache = packet.setdefault("lda_feature_representations", {})
    if view not in cache:
        if view == "drug_targets":
            transform = features.Block(view).fit(fit)
            a = transform.transform(fit)
            selected, names = None, transform.feature_names
        else:
            transform = BioBlock("morgan", {}).fit(fit)
            x, missing, _ = transform.transform(fit)
            selected = set()
            for j in range(5):
                if len(np.unique(packet["y"][:, j]))<2:
                    continue
                values = np.nan_to_num(chi2(x[packet["gold"]], packet["y"][:, j])[0], nan=0., posinf=0., neginf=0.)
                selected.update(int(k) for k in np.argsort(-values, kind="stable")[:60] if values[k]>0)
            selected = np.array(sorted(selected), dtype=int)
            a = np.column_stack([normalize(x[:, selected]).toarray(), missing])
            names = [transform.names[k] for k in selected]+["__missing__"]
        multiplier = .3*np.sqrt(np.mean(np.sum(packet["z"]**2, axis=1)))/max(np.sqrt(np.mean(np.sum(a*a, axis=1))), 1e-12)
        cache[view] = dict(transform=transform, a=a*multiplier, selected=selected, multiplier=multiplier, names=names)
    saved = cache[view]
    if view == "drug_targets":
        q = saved["transform"].transform(query)
    else:
        x, missing, _ = saved["transform"].transform(query)
        q = np.column_stack([normalize(x[:, saved["selected"]]).toarray(), missing])
    return saved["a"], q*saved["multiplier"], saved


def fit_predict(packet, fit, query, name, source, features):
    mode, view = SETTINGS[name]
    base = source.fitted_variant(packet, source.SETTINGS["revision_full"])
    zq = packet["scaler"].transform(query[features.EMB].to_numpy())
    x, q = packet["z"], zq
    detail = {}
    if view != "none":
        a, b, saved = representation(packet, fit, query, view, features)
        x, q = np.column_stack([x, a]), np.column_stack([q, b])
        detail = dict(feature_names_sha256=hashlib.sha256(json.dumps(saved["names"]).encode()).hexdigest(),
            dimensions=a.shape[1], multiplier=float(saved["multiplier"]), vocabulary_fit_ids=fit.ID.tolist(),
            feature_selection_gold_ids=fit.iloc[packet["gold"]].ID.tolist() if view=="morgan" else [])
    cache = packet.setdefault("lda_feature_heads", {})
    rows, targets = (base["rows"], base["targets"]) if mode=="augmented" else (packet["gold"], packet["y"])
    if name not in cache:
        cache[name] = source.fit_base(x[rows], targets, C=.1, lda_weight=0.)["lr"]
    model = cache[name]
    probability = np.zeros((len(query), 5))
    probability[:, model.classes_] = model.predict_proba(q)
    old = np.zeros_like(probability)
    old[:, base["base"]["lr"].classes_] = base["base"]["lr"].predict_proba(zq)
    raw = source.predict_base(base["base"], zq)
    blend = 1. if mode=="augmented" else .5
    changed = features.centered(np.clip(raw+.5*blend*(probability-old), 0, 1))
    return changed, features.centered(probability), dict(block=name, fitting_ids=fit.iloc[rows].ID.tolist(),
        pseudo_labels_latest_revision=True, final_logistic_C=.1, replacement_fraction=blend,
        source_logistic_weight=.5, original_LDA_and_neighbors_preserved=True, **detail)
