"""Pipeline composition from config dicts.

A *single* pipeline config:
    {
      "features":   {"scale": True, "pca": 64, "l2": False, "blocks": {...}} | None,
      "synth":      [ {"type": "gmm", "k": 16}, {"type": "knn_vote", "k": 10}, ... ],
      "labeler":    {"type": "cluster"|"spread"|"cs"|"sld"|"selftrain", "lam": 0.1, ...} | None,
      "clf":        {"type": "lr"|"ovr"|"lda"|"knn"|"hgb"|"nb"|"direct", ...},
      "temperature": True,
    }
An *ensemble*:  {"ensemble": [single, single, ...], "weights": None | [w1, w2, ...]}
A *stack* is a single pipeline whose synth list holds {"type": "oof_model", "model": single}
entries (nested OOF meta-features) and whose features are None (or a small base block).

A full experiment config adds the head and the transductive toggle:
    {"model": <single|ensemble>, "head": {"prior": "none"|"sld", "sld_a": 100,
     "decode": "map"|"expf1"}, "transductive": True}

run_model(model_cfg, ctx, rows) -> {"P": calibrated combo probs for rows, "prior": pi_s, ...}
Only labels of ctx.train are ever read.
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .classifiers import make_classifier
from .features import featurize
from .head import apply_temperature, fit_temperature


def _mask_absent(P, prior):
    """Combos with zero training mass keep probability exactly 0 (temperature/log clipping
    would otherwise resurrect them and SLD would then amplify them)."""
    present = np.asarray(prior) > 1e-6
    if present.all():
        return P
    P = np.where(present[None, :], P, 0.0)
    return P / np.maximum(P.sum(1, keepdims=True), 1e-300)
from .labelers import expand, pseudo_labels, soft_labels, soft_labels_train_oof
from .synth import build_synth_block

warnings.filterwarnings("ignore", message="The least populated class")


def inner_folds(y, n_splits=5, seed=0):
    n_splits = int(min(n_splits, max(2, np.bincount(y).max())))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def _design(cfg, ctx):
    """Base features + synthetic features for every global row."""
    parts = []
    if cfg.get("features") is not None:
        parts.append(featurize(ctx, cfg["features"]))
    S = build_synth_block(cfg.get("synth"), ctx)
    if S is not None:
        parts.append(S)
    if not parts:
        raise ValueError("pipeline has neither base features nor synthetic features")
    return np.hstack(parts) if len(parts) > 1 else parts[0]


def run_single(cfg, ctx, rows):
    rows = np.asarray(rows)
    D = ctx.D
    K, CM = D.K, D.CM
    y = ctx.y_train
    clf_cfg = cfg["clf"]
    info = {}

    # ---- direct labeler-as-predictor (cluster / spread / cs / sld) ----
    if clf_cfg["type"] == "direct":
        lcfg = clf_cfg["labeler"]
        P = soft_labels(lcfg, ctx, rows)
        prior = np.bincount(y, minlength=K) / len(y)
        T = 1.0
        if cfg.get("temperature", True):
            P_oof = soft_labels_train_oof(lcfg, ctx)
            T = fit_temperature(P_oof, y)
            P = apply_temperature(P, T)
        return {"P": _mask_absent(P, prior), "prior": prior, "T": T, "info": info}

    X = _design(cfg, ctx)
    Ytr = np.eye(K)[y]
    # ---- labeler -> weighted pseudo rows from the unlabeled pool ----
    pr, Q, w = np.array([], int), np.zeros((0, K)), np.zeros(0)
    if cfg.get("labeler"):
        pr, Q, w, linfo = pseudo_labels(cfg["labeler"], ctx, clf_cfg, X)
        info["labeler"] = linfo
    Xt, yt, wt = expand(X[ctx.train], Ytr, np.ones(len(y)), X[pr], Q, w)
    clf = make_classifier(clf_cfg, K, CM).fit(Xt, yt, wt)
    P = clf.predict_proba(X[rows])
    prior = clf.prior_
    T = 1.0
    if cfg.get("temperature", True):
        # classifier-stage OOF: pseudo rows fixed, inner folds over the labeled rows
        P_oof = np.zeros((len(y), K))
        for tr, te in inner_folds(y, 5, ctx.seed):
            Xi, yi, wi = expand(X[ctx.train[tr]], Ytr[tr], np.ones(len(tr)), X[pr], Q, w)
            m = make_classifier(clf_cfg, K, CM).fit(Xi, yi, wi)
            P_oof[te] = m.predict_proba(X[ctx.train[te]])
        T = fit_temperature(P_oof, y)
        P = apply_temperature(P, T)
    info["n_pseudo_rows"] = int(len(pr))
    return {"P": _mask_absent(P, prior), "prior": prior, "T": T, "info": info}


def run_model(model_cfg, ctx, rows):
    if "ensemble" in model_cfg:
        members = model_cfg["ensemble"]
        wts = np.asarray(model_cfg.get("weights") or [1.0] * len(members), float)
        wts = wts / wts.sum()
        outs = [run_single(m, ctx, rows) for m in members]
        P = sum(wi * o["P"] for wi, o in zip(wts, outs))
        prior = sum(wi * o["prior"] for wi, o in zip(wts, outs))
        return {"P": P, "prior": prior, "T": [o["T"] for o in outs], "info": {"members": [o["info"] for o in outs]},
                "member_P": [o["P"] for o in outs], "member_prior": [o["prior"] for o in outs]}
    return run_single(model_cfg, ctx, rows)


# ----------------------------------------------------------------------------- validation
def final_stage_singles(model_cfg):
    return model_cfg["ensemble"] if "ensemble" in model_cfg else [model_cfg]


def validate(cfg):
    """Enforce the 'only ONE prior correction' rule and basic sanity."""
    model = cfg["model"]
    head = cfg.get("head", {})
    corrections = set()
    for s in final_stage_singles(model):
        clf = s["clf"]
        if clf.get("class_weight") == "balanced":
            corrections.add("class_weight=balanced")
        lab = s.get("labeler") or {}
        if lab.get("type") == "selftrain" and lab.get("align", "none") != "none" and clf.get("class_weight") == "balanced":
            raise ValueError("self-training alignment + balanced class weights = double prior correction")
    if head.get("prior", "none") == "sld":
        corrections.add("sld")
    if len(corrections) > 1:
        raise ValueError(f"more than one prior correction: {sorted(corrections)}")
    return sorted(corrections)


def model_part(cfg):
    """Config without the post-hoc head (SLD / decoding) -> cache key for fold outputs."""
    return {"model": cfg["model"], "transductive": cfg.get("transductive", True)}


def deep(cfg):
    return copy.deepcopy(cfg)
