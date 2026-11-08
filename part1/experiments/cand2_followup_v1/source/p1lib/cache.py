"""Tiny in-process memo for label-free fits (scalers, PCA, clusterers, graphs).

A label-free object depends only on (its config, the set of rows whose features it was fit on).
In transductive mode that row set is the same (all 1,561 rows) for every fold, so one fit is
reused across folds and trials; in inductive mode the row set changes per fold and the cache
simply misses.  Nothing that touches labels is ever cached here.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

_CACHE: dict = {}
_MAX = 4000


def cfg_key(obj) -> str:
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def rows_key(rows: np.ndarray) -> str:
    return hashlib.md5(np.sort(np.asarray(rows)).astype(np.int64).tobytes()).hexdigest()[:16]


def memo(kind: str, cfg, rows, fn):
    key = (kind, cfg_key(cfg), rows_key(rows))
    if key in _CACHE:
        return _CACHE[key]
    val = fn()
    if len(_CACHE) > _MAX:
        _CACHE.clear()
    _CACHE[key] = val
    return val


def clear():
    _CACHE.clear()


def memo_ctx(kind: str, cfg, ctx, fn, extra=None):
    """Memo for LABEL-DEPENDENT results: keyed by the exact training rows (so a result is only
    reused for the identical fold), the feature rows, the pool and the requested rows."""
    key = (kind, cfg_key(cfg), rows_key(ctx.train), rows_key(ctx.unsup), rows_key(ctx.pool),
           ctx.seed, None if extra is None else cfg_key(extra))
    if key in _CACHE:
        return _CACHE[key]
    val = fn()
    if len(_CACHE) > _MAX:
        _CACHE.clear()
    _CACHE[key] = val
    return val
