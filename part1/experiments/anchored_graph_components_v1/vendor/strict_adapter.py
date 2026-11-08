"""Inductive, group-safe adapters for the four fixed Dylan v2 candidates.

Original p1lib is copied unchanged under source/. This file changes fitting
boundaries and nested calibration, not classifier settings or selection weights.
"""
from dataclasses import replace
import hashlib

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from p1lib import cache, features, graph, labelers
from p1lib.classifiers import make_classifier, KNNClf
from p1lib.context import Ctx
from p1lib.head import apply_temperature, fit_temperature, safe_log
from p1lib.pipeline import _mask_absent

AUDIT = []


def groups(D):
    if not hasattr(D, '_embedding_groups'):
        D._embedding_groups = pd.util.hash_pandas_object(
            D.df[[f'emb_{i}' for i in range(150)]].round(10), index=False).to_numpy()
    return D._embedding_groups


def context(D, train, query, pool, seed=0, name='', depth=0):
    train, query, pool = [np.asarray(v, dtype=int) for v in [train, query, pool]]
    unsup = np.union1d(train, pool)
    assert not np.intersect1d(unsup, query).size
    assert not np.intersect1d(unsup, D.idx['test']).size
    # The only accessible labels belong to the fitting gold rows.
    y = np.full_like(D.y, -1); y[train] = D.y[train]
    Y = np.full_like(D.Y, -1); Y[train] = D.Y[train]
    df = D.df.copy(); df.loc[:, ['Safety', 'Funding', 'Efficacy', 'Recruitment', 'Other']] = Y
    hidden = replace(D, df=df, y=y, Y=Y)
    hidden._embedding_groups = groups(D)
    return Ctx(hidden, train, query, pool, unsup, False, seed, name, depth)


def subcontext(ctx, train, query):
    g = groups(ctx.D)
    banned = g[query]
    train = np.asarray(train)[~np.isin(g[train], banned)]
    pool = ctx.pool[~np.isin(g[ctx.pool], banned)]
    sub = context(ctx.D, train, query, pool, ctx.seed, ctx.name + '/inner', ctx.depth + 1)
    assert not np.intersect1d(g[sub.unsup], g[query]).size
    assert np.isin(sub.unsup, ctx.unsup).all()
    AUDIT.append(dict(depth=sub.depth, gold=len(train), masked=len(pool), query=len(query),
                      fitting_ids_sha256=hashlib.sha256(sub.D.df.iloc[sub.unsup].ID.to_numpy(str).tobytes()).hexdigest(),
                      gold_group_overlap=0, masked_group_overlap=0))
    return sub


def inner_folds(ctx, n=5):
    _, inv = np.unique(groups(ctx.D)[ctx.train], return_inverse=True)
    target = np.zeros((inv.max() + 1, 5), dtype=int)
    np.maximum.at(target, inv, ctx.D.CM[ctx.y_train])
    splitter = MultilabelStratifiedKFold(n_splits=n, shuffle=True, random_state=ctx.seed)
    for tr, te in splitter.split(np.zeros((len(target), 1)), target):
        yield np.flatnonzero(np.isin(inv, tr)), np.flatnonzero(np.isin(inv, te))


def base_oof(ctx, base_cfg, feature_cfg):
    def compute():
        K, CM = ctx.D.K, ctx.D.CM
        X = features.featurize(ctx, feature_cfg)
        P_oof = np.zeros((len(ctx.train), K))
        for tr, te in inner_folds(ctx):
            sub = subcontext(ctx, ctx.train[tr], ctx.train[te])
            Xi = features.featurize(sub, feature_cfg)
            m = make_classifier(base_cfg, K, CM).fit(Xi[sub.train], sub.y_train)
            P_oof[te] = m.predict_proba(Xi[sub.eval])
        m = make_classifier(base_cfg, K, CM).fit(X[ctx.train], ctx.y_train)
        return P_oof, m.predict_proba(X)
    return cache.memo_ctx('strict_base_oof', [base_cfg, feature_cfg], ctx, compute)


def cs_probabilities(lcfg, ctx, rows):
    assert lcfg['type'] == 'cs'
    core = {k: v for k, v in lcfg.items() if k not in ('lam', 'tau', 'hard', 'pool')}
    def compute():
        oof, all_p = base_oof(ctx, lcfg.get('base', {'type': 'lr', 'C': .1}),
                            lcfg.get('base_features', {'scale': True}))
        return graph.correct_and_smooth(ctx, lcfg.get('graph'), float(lcfg.get('a1', .5)),
                float(lcfg.get('a2', .5)), ctx.train, ctx.y_train, oof, all_p, ctx.D.K, rows)
    return cache.memo_ctx('strict_cs', core, ctx, compute, extra=np.asarray(rows).tolist())


def knn_features(cfg, ctx):
    def compute():
        space = cfg.get('space', {'scale': True, 'l2': True})
        knn = dict(type='knn', k=int(cfg.get('k', 10)), weights=cfg.get('weights', 'uniform'),
                   beta=float(cfg.get('beta', 1)))
        Z = features.featurize(ctx, space)
        m = KNNClf(knn, ctx.D.K, ctx.D.CM).fit(Z[ctx.train], ctx.y_train)
        out = m.predict_proba(Z)
        g = groups(ctx.D)
        # Preserve the source's leave-one-out idea, but leave out the entire
        # embedding group and refit scaling with that group's masked copies purged.
        for value in np.unique(g[ctx.train]):
            held = ctx.train[g[ctx.train] == value]
            train = ctx.train[g[ctx.train] != value]
            sub = subcontext(ctx, train, held)
            # Avoid caching one large transformed matrix per leave-group-out fit.
            Zi = features._embed(sub.D, features.norm_features_cfg(space), sub.unsup)
            m = KNNClf(knn, ctx.D.K, ctx.D.CM).fit(Zi[sub.train], sub.y_train)
            out[held] = m.predict_proba(Zi[held])
        return safe_log(out)
    return cache.memo_ctx('strict_knn_feature', cfg, ctx, compute)


def kmeans_features(cfg, ctx):
    assert cfg['type'] == 'kmeans' and cfg.get('out') == 'soft'
    space = cfg.get('space', labelers.DEFAULT_CLUSTER_SPACE)
    Z = features.featurize(ctx, space)
    key = dict(algo='kmeans', k=int(cfg.get('k', 16)), space=space,
               **{k: cfg[k] for k in ('cov', 'n_init') if k in cfg})
    fit = cache.memo('cluster', key, ctx.unsup,
                     lambda: labelers._cluster_fit(ctx.D, 'kmeans', key['k'], Z, ctx.unsup, cfg))
    d2 = fit['dist'] ** 2
    # Source uses all global rows for this median, including eval/test. Restrict it.
    T = float(cfg.get('temp', 1)) * np.median(d2[ctx.unsup].min(1))
    logp = -(d2 - d2.min(1, keepdims=True)) / max(T, 1e-12)
    result = np.exp(logp)
    return result / result.sum(1, keepdims=True)


def design(cfg, ctx):
    parts = []
    if cfg.get('features') is not None:
        assert not cfg['features'].get('blocks')
        parts.append(features.featurize(ctx, cfg['features']))
    synth = []
    for scfg in cfg.get('synth', []):
        if scfg['type'] == 'kmeans':
            synth.append(kmeans_features(scfg, ctx))
        elif scfg['type'] == 'knn_vote':
            synth.append(knn_features(scfg, ctx))
        else:
            raise ValueError(scfg)
    if synth:
        S = np.hstack(synth)
        mu, sd = S[ctx.unsup].mean(0), S[ctx.unsup].std(0)
        sd[sd < 1e-9] = 1
        parts.append((S - mu) / sd)
    return np.hstack(parts) if len(parts) > 1 else parts[0]


def uncalibrated(cfg, ctx, rows):
    K, CM, y = ctx.D.K, ctx.D.CM, ctx.y_train
    if cfg['clf']['type'] == 'direct':
        return dict(P=cs_probabilities(cfg['clf']['labeler'], ctx, rows),
                    prior=np.bincount(y, minlength=K) / len(y), info={})
    X = design(cfg, ctx)
    pr, Q, w = np.array([], int), np.zeros((0, K)), np.zeros(0)
    lcfg = cfg.get('labeler')
    if lcfg:
        assert lcfg.get('pool', 'unl') == 'unl'
        pr = ctx.pool
        Q = cs_probabilities(lcfg, ctx, pr)
        keep = Q.max(1) >= float(lcfg.get('tau', 0))
        if lcfg.get('hard', False):
            Q = np.eye(K)[Q.argmax(1)]
        pr, Q = pr[keep], Q[keep]
        w = np.full(len(pr), float(lcfg.get('lam', .1)))
    Xt, yt, wt = labelers.expand(X[ctx.train], np.eye(K)[y], np.ones(len(y)), X[pr], Q, w)
    m = make_classifier(cfg['clf'], K, CM).fit(Xt, yt, wt)
    return dict(P=m.predict_proba(X[rows]), prior=m.prior_, info={'n_pseudo_rows': len(pr)})


def run_single(cfg, ctx, rows):
    out = uncalibrated(cfg, ctx, rows)
    T = 1.
    if cfg.get('temperature', True):
        oof = np.zeros((len(ctx.train), ctx.D.K))
        for tr, te in inner_folds(ctx):
            sub = subcontext(ctx, ctx.train[tr], ctx.train[te])
            # Relearn features, pseudo labels and classifier inside each inner fold.
            oof[te] = uncalibrated(cfg, sub, sub.eval)['P']
        T = fit_temperature(oof, ctx.y_train)
        out['P'] = apply_temperature(out['P'], T)
    out['P'] = _mask_absent(out['P'], out['prior'])
    out['T'] = T
    return out


def run_model(cfg, ctx, rows):
    if 'ensemble' not in cfg:
        return run_single(cfg, ctx, rows)
    members = cfg['ensemble']
    weights = np.array(cfg.get('weights') or [1.] * len(members)); weights /= weights.sum()
    outs = [run_single(m, ctx, rows) for m in members]
    return dict(P=sum(w * o['P'] for w, o in zip(weights, outs)),
                prior=sum(w * o['prior'] for w, o in zip(weights, outs)),
                T=[o['T'] for o in outs], info={'members': [o['info'] for o in outs]})
