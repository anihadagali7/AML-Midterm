"""Frozen, inductive C&S component experiments around the submitted anchored model."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / 'vendor'), str(HERE.parents[1])]

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

from anchored_solution import AnchoredReasonClassifier, select_reasons
from pseudo_anchored import PseudoTeacherAnchored
from p1lib import cache
from p1lib.data import Data, LABELS, EMB, BITS
from p1lib.graph import Graph
import strict_adapter as strict

GRAPH = dict(space=dict(scale=True, pca=64, l2=True), k=10, s=1., sym='max')
CS = dict(type='cs', graph=GRAPH, a1=.8, a2=.6382007398005838,
          base=dict(type='lr', C=.1), base_features=dict(scale=True))
VARIANTS = ['cs_pseudo', 'graph_residual_raw', 'graph_residual_autoscale']


def fitting_context(frame):
    """The caller supplies eligible fitting rows only; no outer query is stored."""
    frame = frame.reset_index(drop=True).copy()
    Y = frame[LABELS].to_numpy(dtype=int, copy=True)
    known = (Y >= 0).all(1)
    assert (known | (Y == -100).all(1)).all()
    Y[~known] = -1
    bitmask = Y[known] @ BITS
    combos = np.unique(bitmask)
    CM = ((combos[:, None] & BITS[None, :]) > 0).astype(int)
    y = np.full(len(frame), -1)
    y[known] = np.searchsorted(combos, bitmask)
    train, pool = np.flatnonzero(known), np.flatnonzero(~known)
    D = Data(frame, frame[EMB].to_numpy(float), Y, y, combos, CM,
             idx=dict(lab=train, unl=pool, train=np.arange(len(frame)),
                      labval=train, val=np.array([], int), test=np.array([], int)))
    return strict.context(D, train, np.array([], int), pool, seed=0)


def fit_pseudo(frame):
    cache.clear()
    strict.AUDIT.clear()
    ctx = fitting_context(frame)
    Q = strict.cs_probabilities(CS, ctx, ctx.pool)
    assert np.isfinite(Q).all() and np.allclose(Q.sum(1), 1.)
    teacher = Q @ ctx.D.CM
    m = PseudoTeacherAnchored().fit(ctx.D.emb, frame[LABELS].to_numpy(int), teacher)
    m.component_audit_ = dict(teacher='C&S powerset LR, grouped inner OOF',
        teacher_generated=int(m.generated_mask_.sum()),
        generated_positive_counts=m.generated_labels_.sum(0).tolist(),
        inner=list(strict.AUDIT), rounds=m.round_audit_)
    return m


class ResidualCorrector:
    """Signed five-label residual field; shares one fit between the two scales.

    Inner anchored fits produce OOF errors. The graph sees fitting inputs only.
    Prediction extends that field to new rows and clips each label independently.
    There is deliberately no cross-label normalization or final smoothing stage.
    """
    def fit(self, frame, parent):
        strict.AUDIT.clear()
        ctx = fitting_context(frame)
        X = ctx.D.emb
        original_Y = frame[LABELS].to_numpy(int)
        oof = np.zeros((len(ctx.train), 5))
        coverage = np.zeros(len(ctx.train), int)
        for tr, te in strict.inner_folds(ctx):
            sub = strict.subcontext(ctx, ctx.train[tr], ctx.train[te])
            assert (original_Y[sub.train].sum(0) > 0).all()
            inner = AnchoredReasonClassifier().fit(X[sub.unsup], original_Y[sub.unsup])
            oof[te] = inner.predict_proba(X[sub.eval])
            coverage[te] += 1
        assert (coverage == 1).all()
        self.scaler = StandardScaler().fit(X)
        self.pca = PCA(64, random_state=0).fit(self.scaler.transform(X))
        self.graph = Graph(self.transform(X), np.arange(len(X)), len(X), 10, 1., 'max')
        E0 = np.zeros((len(X), 5))
        E0[ctx.train] = original_Y[ctx.train] - oof
        raw = self.graph.propagate(E0, .8)
        sigma = np.abs(E0[ctx.train]).sum(1).mean()
        scale = sigma / np.maximum(np.abs(raw).sum(1, keepdims=True), 1e-12)
        self.fields = dict(raw=raw, autoscale=scale * raw)
        self.parent = parent
        self.audit = dict(inner=list(strict.AUDIT), inner_coverage_exactly_once=True,
            inner_oof_sample_f1=float((2 * (select_reasons(oof) * original_Y[ctx.train]).sum(1)
                / (select_reasons(oof).sum(1) + original_Y[ctx.train].sum(1))).mean()),
            residual_mean_l1=float(sigma),
            scale_quantiles=np.quantile(scale, [0, .25, .5, .75, 1]).tolist(),
            raw_field_abs_quantiles=np.quantile(np.abs(raw), [0, .5, .9, 1]).tolist())
        return self

    def transform(self, X):
        return normalize(self.pca.transform(self.scaler.transform(X)))

    def predict_proba(self, X, mode):
        return np.clip(self.parent.predict_proba(X) + self.correction(X, mode), 0, 1)

    def correction(self, X, mode):
        return self.graph.extend(self.transform(X), self.fields[mode])
