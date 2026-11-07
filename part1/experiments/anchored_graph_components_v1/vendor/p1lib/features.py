"""Preprocessors: standardize -> [PCA] -> [L2-normalize] on the embeddings, plus optional
non-embedding blocks.  All fitting uses only `ctx.unsup` rows (label-free), so the same
object is refit per fold when the feature-row set changes (inductive) and reused otherwise.

features config (all keys optional):
    {"scale": True, "pca": None|int, "whiten": False, "l2": False,
     "blocks": {"criteria": {"d": 32, "weight": 1.0}, "pathways": {...}, ...}}
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from .blocks import build_block
from .cache import memo

DEFAULT_FEATURES = {"scale": True, "pca": None, "l2": False, "blocks": {}}


def norm_features_cfg(cfg):
    c = dict(DEFAULT_FEATURES)
    c.update(cfg or {})
    c["blocks"] = dict(c.get("blocks") or {})
    return c


def _embed(D, cfg, rows):
    X = D.emb.copy()
    if cfg["scale"]:
        mu = X[rows].mean(0)
        sd = X[rows].std(0)
        X = (X - mu) / sd
    if cfg["pca"]:
        p = PCA(n_components=int(cfg["pca"]), whiten=bool(cfg.get("whiten", False)), random_state=0).fit(X[rows])
        X = p.transform(X)
    if cfg["l2"]:
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X


def featurize(ctx, cfg) -> np.ndarray:
    """Return the (n_all, d) representation for every global row."""
    cfg = norm_features_cfg(cfg)
    emb_cfg = {k: cfg[k] for k in ("scale", "pca", "l2", "whiten") if k in cfg}
    X = memo("embed", emb_cfg, ctx.unsup, lambda: _embed(ctx.D, cfg, ctx.unsup))
    if not cfg["blocks"]:
        return X
    parts = [X] if cfg.get("use_emb", True) else []
    for name in sorted(cfg["blocks"]):
        parts.append(build_block(ctx.D, name, cfg["blocks"][name], ctx.unsup))
    return np.hstack(parts)
