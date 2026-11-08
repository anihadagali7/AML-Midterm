"""Synthetic features: unsupervised / semi-supervised model outputs turned into new columns
that downstream classifiers (or a stacker) consume.

    synth_features(scfg, ctx) -> (n_all, m) array for every global row

Label-free generators (fit on ctx.unsup features only, cached):
  gmm        : GMM posterior responsibilities (or log-responsibilities)
  kmeans     : distances to K-means centroids, or soft assignments softmax(-d^2/T)
Label-dependent generators (training rows get OUT-OF-FOLD values, so no row sees its own label):
  knn_vote   : cosine kNN label-vote fractions over the K combos (exact leave-one-out)
  graph      : label-spreading or C&S scores (inner K-fold for training rows)
  cluster_map: cluster -> label-combo mapped probabilities (exact leave-one-out)
  oof_model  : out-of-fold probabilities of any single pipeline config (inner K-fold = nested OOF)
Label-dependent outputs are returned as log-probabilities (K columns).
"""
from __future__ import annotations

import numpy as np

from .cache import memo, memo_ctx
from .classifiers import KNNClf
from .features import featurize
from .head import safe_log
from .labelers import DEFAULT_CLUSTER_SPACE, _cluster_fit, soft_labels, soft_labels_train_oof

LABEL_FREE = {"gmm", "kmeans"}


def _label_free(scfg, ctx):
    algo = scfg["type"]
    k = int(scfg.get("k", 16))
    space = scfg.get("space", DEFAULT_CLUSTER_SPACE)
    Z = featurize(ctx, space)
    key = {"algo": algo, "k": k, "space": space, **{kk: scfg[kk] for kk in ("cov", "n_init") if kk in scfg}}
    fit = memo("cluster", key, ctx.unsup, lambda: _cluster_fit(ctx.D, algo, k, Z, ctx.unsup, scfg))
    if algo == "gmm":
        R = fit["resp"]
        return np.log(np.clip(R, 1e-6, None)) if scfg.get("out", "resp") == "logresp" else R
    d = fit["dist"]
    if scfg.get("out", "dist") == "soft":
        d2 = d ** 2
        T = float(scfg.get("temp", 1.0)) * np.median(d2.min(1))
        L = -(d2 - d2.min(1, keepdims=True)) / max(T, 1e-12)
        R = np.exp(L)
        return R / R.sum(1, keepdims=True)
    return d


def _knn_vote(scfg, ctx):
    K = ctx.D.K
    Z = featurize(ctx, scfg.get("space", {"scale": True, "l2": True}))
    y = ctx.y_train
    m = KNNClf({"type": "knn", "k": int(scfg.get("k", 10)), "weights": scfg.get("weights", "uniform"),
                "beta": float(scfg.get("beta", 1.0))}, K, ctx.D.CM).fit(Z[ctx.train], y)
    out = m.predict_proba(Z)
    out[ctx.train] = m.predict_proba(Z[ctx.train], exclude_self=True)
    return safe_log(out)


def _labeler_feature(lcfg, ctx):
    rows = np.setdiff1d(np.arange(ctx.D.n), ctx.train)
    out = np.zeros((ctx.D.n, ctx.D.K))
    out[rows] = soft_labels(lcfg, ctx, rows)
    out[ctx.train] = soft_labels_train_oof(lcfg, ctx)
    return safe_log(out)


def _oof_model(scfg, ctx):
    from .pipeline import inner_folds, run_single
    sub = scfg["model"]
    y = ctx.y_train
    out = np.zeros((ctx.D.n, ctx.D.K))
    rows = np.setdiff1d(np.arange(ctx.D.n), ctx.train)
    out[rows] = run_single(sub, ctx, rows)["P"]
    for tr, te in inner_folds(y, int(scfg.get("n_inner", 5)), ctx.seed + 17):
        inner = ctx.sub(ctx.train[tr], ctx.train[te])
        out[ctx.train[te]] = run_single(sub, inner, ctx.train[te])["P"]
    return safe_log(out)


def synth_features(scfg, ctx) -> np.ndarray:
    if scfg["type"] in LABEL_FREE:
        return _synth(scfg, ctx)
    return memo_ctx("synth", scfg, ctx, lambda: _synth(scfg, ctx))


def _synth(scfg, ctx) -> np.ndarray:
    t = scfg["type"]
    if t in LABEL_FREE:
        return _label_free(scfg, ctx)
    if t == "knn_vote":
        return _knn_vote(scfg, ctx)
    if t == "graph":
        mode = scfg.get("mode", "spread")
        lcfg = {"type": "spread" if mode == "spread" else "cs", "graph": scfg.get("graph"),
                "alpha": scfg.get("alpha", 0.5), "a1": scfg.get("a1", 0.5), "a2": scfg.get("alpha", 0.5)}
        return _labeler_feature(lcfg, ctx)
    if t == "cluster_map":
        lcfg = {"type": "cluster", **{k: v for k, v in scfg.items() if k != "type"}}
        return _labeler_feature(lcfg, ctx)
    if t == "oof_model":
        return _oof_model(scfg, ctx)
    raise ValueError(t)


def build_synth_block(synth_list, ctx):
    """Concatenate all synthetic features and standardize them on train+pool rows."""
    if not synth_list:
        return None
    parts = [synth_features(s, ctx) for s in synth_list]
    S = np.hstack(parts)
    fit_rows = np.union1d(ctx.train, ctx.pool)
    mu = S[fit_rows].mean(0)
    sd = S[fit_rows].std(0)
    sd[sd < 1e-9] = 1.0
    return (S - mu) / sd
