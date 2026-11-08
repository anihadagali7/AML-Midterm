"""Fold context: which rows are labeled-for-training, which are predicted, which are unlabeled.

Every component receives a `Ctx` and may only read labels of `ctx.train`.  Features of
`ctx.unsup` rows may be used by label-free steps (scaler, PCA, clustering, graph nodes).

transductive=True  -> unsup = all 1,561 rows (train + val + test features, no labels)
transductive=False -> unsup = all rows except the eval rows and the test rows
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass
class Ctx:
    D: object
    train: np.ndarray            # global idx of labeled training rows (labels visible)
    eval: np.ndarray             # global idx of rows to predict (labels hidden)
    pool: np.ndarray             # global idx of the pseudo-label pool (default: 1,014 unlabeled)
    unsup: np.ndarray            # global idx of rows whose features label-free steps may use
    transductive: bool = True
    seed: int = 0
    name: str = ""
    depth: int = 0               # nesting depth (0 = outer fold)

    @property
    def y_train(self):
        return self.D.y[self.train]

    @property
    def sld_target(self):
        """Rows SLD estimates the target prior on: the eval batch when transductive, else the
        unlabeled pool (inductive: we may not look at the eval/test features)."""
        return self.eval if self.transductive else self.pool

    def sub(self, train, eval_, name=None):
        """Inner context for nested OOF: same unlabeled pool / feature rows, fewer labels."""
        return replace(self, train=np.asarray(train), eval=np.asarray(eval_),
                       name=name or self.name + "/inner", depth=self.depth + 1)


def make_ctx(D, train, eval_, transductive=True, pool=None, seed=0, name=""):
    train = np.asarray(train)
    eval_ = np.asarray(eval_)
    pool = D.idx["unl"] if pool is None else np.asarray(pool)
    if transductive:
        unsup = np.arange(D.n)
    else:
        banned = np.zeros(D.n, bool)
        banned[eval_] = True
        banned[D.idx["test"]] = True
        unsup = np.where(~banned)[0]
    assert len(np.intersect1d(train, eval_)) == 0, "train/eval overlap"
    assert (D.y[train] >= 0).all(), "training rows must be labeled"
    return Ctx(D=D, train=train, eval=eval_, pool=pool, unsup=unsup,
               transductive=transductive, seed=seed, name=name)
