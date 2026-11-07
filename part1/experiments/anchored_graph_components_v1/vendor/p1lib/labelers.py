"""Labelers: fill the missing labels of the unlabeled pool with soft combo distributions.

Every labeler type exposes
    soft_labels(lcfg, ctx, rows) -> Q (len(rows), K)      for rows NOT in ctx.train
and, where cheap and exact, a leave-one-out version for training rows (used when the labeler's
output is a synthetic feature or a direct predictor that needs OOF scores):
    soft_labels_train_oof(lcfg, ctx) -> Q (len(ctx.train), K)

`pseudo_labels(lcfg, ctx, clf_cfg)` turns soft labels into weighted pseudo-rows
(rows, Q, w) for the pipeline's classifier (soft-label replication, weight lam * q_ik).

Types
  cluster   : cluster-then-label (kmeans | gmm | seeded_kmeans | hdbscan), m-smoothed map   [T4]
  spread    : label spreading on the kNN graph (Zhou et al. 2004)                            [T2]
  cs        : Correct & Smooth on the kNN graph (Huang et al. 2021)                          [T2]
  sld       : calibrated base classifier + SLD-EM on the pool (Saerens et al. 2002)          [T1]
  selftrain : guarded self-training (agreement, alignment, soft weights, ramp)               [T3]
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans, kmeans_plusplus
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

from .cache import memo, memo_ctx, rows_key
from .classifiers import make_classifier
from .features import featurize
from .graph import correct_and_smooth, label_spread
from .head import apply_temperature, fit_temperature, sld_em, sld_em_factorized

DEFAULT_CLUSTER_SPACE = {"scale": True, "pca": 32, "l2": True}


# ============================================================================ clustering
def _cluster_fit(D, algo, k, Z, rows, cfg, seed=0):
    info = {}
    if algo == "kmeans":
        km = KMeans(n_clusters=k, n_init=int(cfg.get("n_init", 10)), max_iter=300, tol=1e-4,
                    random_state=seed).fit(Z[rows])
        dist = km.transform(Z)
        info.update(inertia=float(km.inertia_), n_iter=int(km.n_iter_))
        return {"dist": dist, "labels": km.predict(Z), "info": info}
    if algo == "gmm":
        g = GaussianMixture(n_components=k, covariance_type=cfg.get("cov", "diag"), reg_covar=1e-3,
                            n_init=int(cfg.get("n_init", 3)), max_iter=500, tol=1e-4,
                            init_params="k-means++", random_state=seed).fit(Z[rows])
        info.update(converged=bool(g.converged_), n_iter=int(g.n_iter_), bic=float(g.bic(Z[rows])),
                    lower_bound=float(g.lower_bound_))
        return {"resp": g.predict_proba(Z), "labels": g.predict(Z), "info": info}
    if algo == "hdbscan":
        h = HDBSCAN(min_cluster_size=int(cfg.get("min_cluster_size", 10)),
                    min_samples=cfg.get("min_samples")).fit(Z[rows])
        lab = np.full(D.n, -1)
        lab[rows] = h.labels_
        others = np.setdiff1d(np.arange(D.n), rows)
        if len(others):                     # inductive: attach to the nearest fitted point
            Zr = Z[rows]
            nn = np.argmax(Z[others] @ Zr.T / (np.linalg.norm(Z[others], axis=1)[:, None] *
                                                np.linalg.norm(Zr, axis=1)[None] + 1e-12), axis=1)
            lab[others] = h.labels_[nn]
        info.update(n_clusters=int(h.labels_.max() + 1), noise_frac=float((h.labels_ < 0).mean()))
        return {"labels": lab, "info": info}
    raise ValueError(algo)


def cluster_resp(lcfg, ctx):
    """(n_all, C) cluster membership (hard one-hot, kmeans-soft, or GMM responsibilities)."""
    algo = lcfg.get("algo", "kmeans")
    k = int(lcfg.get("k", 24))
    space = lcfg.get("space", DEFAULT_CLUSTER_SPACE)
    Z = featurize(ctx, space)
    if algo == "seeded_kmeans":                 # label-dependent init -> not shared across folds
        y = ctx.y_train
        cls = np.unique(y)
        seeds = np.array([Z[ctx.train][y == c].mean(0) for c in cls])
        extra = max(k - len(cls), 0)
        if extra:
            more, _ = kmeans_plusplus(Z[ctx.unsup], n_clusters=extra, random_state=ctx.seed)
            seeds = np.vstack([seeds, more])
        km = KMeans(n_clusters=len(seeds), init=seeds, n_init=1, max_iter=300).fit(Z[ctx.unsup])
        lab = km.predict(Z)
        return np.eye(len(seeds))[lab]
    cfg_key = {"algo": algo, "k": k, "space": space, **{kk: lcfg[kk] for kk in ("cov", "n_init", "min_cluster_size", "min_samples") if kk in lcfg}}
    fit = memo("cluster", cfg_key, ctx.unsup, lambda: _cluster_fit(ctx.D, algo, k, Z, ctx.unsup, lcfg))
    if algo == "gmm":
        return fit["resp"]
    if algo == "kmeans" and lcfg.get("soft_temp"):
        d2 = fit["dist"] ** 2
        T = float(lcfg["soft_temp"]) * np.median(d2.min(1))
        L = -(d2 - d2.min(1, keepdims=True)) / max(T, 1e-12)
        R = np.exp(L)
        return R / R.sum(1, keepdims=True)
    lab = fit["labels"]
    C = int(max(lab.max() + 1, 1))
    R = np.zeros((ctx.D.n, C))
    R[lab >= 0, lab[lab >= 0]] = 1.0          # hdbscan noise rows keep all-zero membership
    return R


def _cluster_map(R_train, y, K, m, prior):
    N = R_train.T @ np.eye(K)[y]                    # (C, K) label mass per cluster
    return N, (N + m * prior) / (N.sum(1, keepdims=True) + m)


def cluster_soft(lcfg, ctx, rows):
    K = ctx.D.K
    R = cluster_resp(lcfg, ctx)
    y = ctx.y_train
    prior = np.bincount(y, minlength=K) / len(y)
    _, pi = _cluster_map(R[ctx.train], y, K, float(lcfg.get("m", 4)), prior)
    Q = R[rows] @ pi
    mass = R[rows].sum(1, keepdims=True)
    Q = Q + (1 - np.clip(mass, 0, 1)) * prior        # noise / empty -> global prior
    return Q / Q.sum(1, keepdims=True)


def cluster_soft_train_oof(lcfg, ctx):
    """Exact leave-one-out cluster map for training rows."""
    K = ctx.D.K
    R = cluster_resp(lcfg, ctx)[ctx.train]
    y = ctx.y_train
    m = float(lcfg.get("m", 4))
    prior = np.bincount(y, minlength=K) / len(y)
    N, _ = _cluster_map(R, y, K, m, prior)
    nc = N.sum(1)                                    # (C,)
    out = np.zeros((len(y), K))
    for i in range(len(y)):
        Ni = N - np.outer(R[i], np.eye(K)[y[i]])
        pi = (Ni + m * prior) / (nc - R[i] + m)[:, None]
        q = R[i] @ pi
        q = q + (1 - min(R[i].sum(), 1)) * prior
        out[i] = q / q.sum()
    return out


# ============================================================================ graph labelers
def spread_soft(lcfg, ctx, rows):
    K = ctx.D.K
    F = label_spread(ctx, lcfg.get("graph"), float(lcfg.get("alpha", 0.5)), ctx.train, ctx.y_train, K, rows)
    prior = np.bincount(ctx.y_train, minlength=K) / len(ctx.train)
    F = np.clip(F, 0, None)
    s = F.sum(1, keepdims=True)
    F = np.where(s > 1e-12, F / np.maximum(s, 1e-12), prior)   # unreachable rows -> prior
    beta = float(lcfg.get("beta", 0.01))
    F = (F + beta * prior) / (1 + beta)
    return F


def _base_oof_and_all(base_cfg, ctx, X, n_inner=5):
    """Base classifier probabilities: OOF for training rows, full-fit for every other row."""
    from .pipeline import inner_folds
    K, CM = ctx.D.K, ctx.D.CM
    y = ctx.y_train
    P_oof = np.zeros((len(y), K))
    for tr, te in inner_folds(y, n_inner, ctx.seed):
        m = make_classifier(base_cfg, K, CM).fit(X[ctx.train[tr]], y[tr])
        P_oof[te] = m.predict_proba(X[ctx.train[te]])
    m = make_classifier(base_cfg, K, CM).fit(X[ctx.train], y)
    P_all = m.predict_proba(X)
    return P_oof, P_all


def cs_soft(lcfg, ctx, rows):
    K = ctx.D.K
    base_cfg = lcfg.get("base", {"type": "lr", "C": 0.1})
    X = featurize(ctx, lcfg.get("base_features", {"scale": True}))
    P_oof, P_all = _base_oof_and_all(base_cfg, ctx, X)
    return correct_and_smooth(ctx, lcfg.get("graph"), float(lcfg.get("a1", 0.5)), float(lcfg.get("a2", 0.5)),
                              ctx.train, ctx.y_train, P_oof, P_all, K, rows)


# ============================================================================ SLD labeler
def calibrated_base(base_cfg, ctx, X, rows, temperature=True):
    """Fit base classifier on ctx.train, temperature-scale with inner OOF, predict rows."""
    from .pipeline import inner_folds
    K, CM = ctx.D.K, ctx.D.CM
    y = ctx.y_train
    m = make_classifier(base_cfg, K, CM).fit(X[ctx.train], y)
    P = m.predict_proba(X[rows])
    T = 1.0
    if temperature:
        P_oof = np.zeros((len(y), K))
        for tr, te in inner_folds(y, 5, ctx.seed):
            mm = make_classifier(base_cfg, K, CM).fit(X[ctx.train[tr]], y[tr])
            P_oof[te] = mm.predict_proba(X[ctx.train[te]])
        T = fit_temperature(P_oof, y)
        P = apply_temperature(P, T)
    return P, m.prior_, T


def sld_soft(lcfg, ctx, rows):
    X = featurize(ctx, lcfg.get("base_features", {"scale": True}))
    base_cfg = lcfg.get("base", {"type": "lr", "C": 0.1})
    P, prior, _ = calibrated_base(base_cfg, ctx, X, rows, lcfg.get("temperature", False))
    if lcfg.get("mode", "combo") == "factorized":
        Q, pi = sld_em_factorized(P, prior, ctx.D.CM, a=float(lcfg.get("a", 0.0)), floor=float(lcfg.get("floor", 0.25)))
    else:
        Q, pi = sld_em(P, prior, a=float(lcfg.get("a", 0.0)), floor=float(lcfg.get("floor", 0.25)))
    return Q


# ============================================================================ self-training
def _sinkhorn(P, target, iters=200):
    Q = P.copy()
    n = len(Q)
    for _ in range(iters):
        col = Q.sum(0)
        Q = Q * (target * n / np.maximum(col, 1e-12))
        Q = Q / Q.sum(1, keepdims=True)
    return Q


def _softmatch_weight(conf):
    mu, sd = conf.mean(), conf.std() + 1e-6
    return np.where(conf >= mu, 1.0, np.exp(-((conf - mu) ** 2) / (2 * sd ** 2)))


def selftrain_pseudo(lcfg, ctx, clf_cfg, X, pool_rows):
    """Guarded self-training.  Returns (rows, Q, w, info) for the final pseudo-labeled set."""
    K, CM = ctx.D.K, ctx.D.CM
    y = ctx.y_train
    Ytr = np.eye(K)[y]
    R = int(lcfg.get("rounds", 2))
    lam_max = float(lcfg.get("lam", 0.3))
    members = lcfg.get("members", ["lr"])
    align = lcfg.get("align", "none")
    tau = lcfg.get("tau", 0.75)
    top_p = lcfg.get("top_p")
    hard = bool(lcfg.get("hard", False))
    weighting = lcfg.get("weighting", "hard")
    # members that do not use pseudo-labels are computed once
    static = []
    if "lda" in members:
        static.append(make_classifier({"type": "lda"}, K, CM).fit(X[ctx.train], y).predict_proba(X[pool_rows]))
    if "cs" in members:
        static.append(cs_soft({"a1": 0.5, "a2": 0.5}, ctx, pool_rows))
    cl_major = None
    if lcfg.get("cluster_filter"):
        cl_major = cluster_soft({"algo": "kmeans", "k": int(lcfg.get("cluster_k", 24)), "m": 1}, ctx, pool_rows).argmax(1)
    sel_rows, Q, w = pool_rows[:0], np.zeros((0, K)), np.zeros(0)
    prev_lab = None
    info = {}
    for r in range(1, R + 1):
        lam_r = lam_max * (r / R if lcfg.get("ramp", False) else 1.0)
        Xt, yt, wt = expand(X[ctx.train], Ytr, np.ones(len(y)), X[sel_rows], Q, w)
        main = make_classifier(clf_cfg, K, CM).fit(Xt, yt, wt)
        Pm_list = [main.predict_proba(X[pool_rows])] + static
        Pm = np.mean(Pm_list, axis=0)
        prior_src = np.bincount(y, minlength=K) / len(y)
        if align == "sld":
            Pm, pi = sld_em_factorized(Pm, prior_src, CM, a=float(lcfg.get("align_a", 100.0)))
        elif align == "sinkhorn":
            _, pi = sld_em_factorized(Pm, prior_src, CM, a=float(lcfg.get("align_a", 100.0)))
            Pm = _sinkhorn(Pm, pi)
        conf = Pm.max(1)
        lab = Pm.argmax(1)
        if top_p:
            keep = np.zeros(len(lab), bool)
            for c in range(K):
                mc = np.where(lab == c)[0]
                if len(mc):
                    thr = np.quantile(conf[mc], 1 - float(top_p))
                    keep[mc[conf[mc] >= thr]] = True
        else:
            keep = conf >= float(tau)
        if len(Pm_list) > 1:
            agree = np.sum([P.argmax(1) == lab for P in Pm_list], axis=0)
            keep &= agree >= 2
        if cl_major is not None:
            keep &= cl_major == lab
        g = _softmatch_weight(conf) if weighting == "softmatch" else np.ones(len(conf))
        sel_rows = pool_rows[keep]
        Q = np.eye(K)[lab[keep]] if hard else Pm[keep]
        w = lam_r * g[keep]
        info = {"round": r, "n_selected": int(keep.sum()), "lam": lam_r}
        if prev_lab is not None and np.mean(prev_lab != lab) < 0.01:
            break
        prev_lab = lab
    return sel_rows, Q, w, info


# ============================================================================ dispatch
SOFT = {"cluster": cluster_soft, "spread": spread_soft, "cs": cs_soft, "sld": sld_soft}


_POST = ("lam", "tau", "hard", "pool")      # applied after soft labels -> not part of the memo key


def soft_labels(lcfg, ctx, rows):
    rows = np.asarray(rows)
    core = {k: v for k, v in lcfg.items() if k not in _POST}
    return memo_ctx("soft", core, ctx, lambda: SOFT[lcfg["type"]](lcfg, ctx, rows), extra=rows.tolist())


def soft_labels_train_oof(lcfg, ctx, n_inner=5):
    """OOF soft labels for the training rows (exact LOO for cluster; inner K-fold otherwise)."""
    if lcfg["type"] == "cluster":
        core = {k: v for k, v in lcfg.items() if k not in _POST}
        return memo_ctx("soft_oof", core, ctx, lambda: cluster_soft_train_oof(lcfg, ctx))
    from .pipeline import inner_folds
    y = ctx.y_train
    out = np.zeros((len(y), ctx.D.K))
    for tr, te in inner_folds(y, n_inner, ctx.seed):
        sub = ctx.sub(ctx.train[tr], ctx.train[te])
        out[te] = soft_labels(lcfg, sub, ctx.train[te])
    return out


def pool_rows_for(lcfg, ctx):
    if lcfg.get("pool", "unl") == "all" and ctx.transductive:
        rows = np.union1d(np.union1d(ctx.pool, ctx.eval), ctx.D.idx["test"])
        return np.setdiff1d(rows, ctx.train)
    return np.setdiff1d(ctx.pool, ctx.train)


def pseudo_labels(lcfg, ctx, clf_cfg, X):
    """Weighted pseudo-rows (rows, Q, w, info) for the pipeline classifier."""
    rows = pool_rows_for(lcfg, ctx)
    if lcfg["type"] == "selftrain":
        return selftrain_pseudo(lcfg, ctx, clf_cfg, X, rows)
    Q = soft_labels(lcfg, ctx, rows)
    conf = Q.max(1)
    keep = conf >= float(lcfg.get("tau", 0.0))
    if lcfg.get("hard", False):
        Q = np.eye(ctx.D.K)[Q.argmax(1)]
    lam = float(lcfg.get("lam", 0.1))
    return rows[keep], Q[keep], np.full(keep.sum(), lam), {"n_selected": int(keep.sum())}


def expand(X_l, Y_l, w_l, X_u, Q_u, w_u, min_q=0.05):
    """Soft-label replication: one copy of each row per combo with weight w * q (exact
    expected cross-entropy for LR).  Y_l may be one-hot or soft."""
    Xs, ys, ws = [], [], []
    for Xb, Yb, wb in ((X_l, Y_l, w_l), (X_u, Q_u, w_u)):
        if len(Xb) == 0:
            continue
        i, k = np.where(Yb >= min_q)
        Xs.append(Xb[i])
        ys.append(k)
        ws.append(wb[i] * Yb[i, k])
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)
