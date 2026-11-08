"""Classifiers over the K label combinations.  Common interface:

    clf = make_classifier(cfg, K, CM)
    clf.fit(X, y, w)            # y: int combo ids, w: sample weights (soft labels are
                                #    passed as replicated rows, one per combo, weight lam*q)
    P = clf.predict_proba(X)    # (n, K), columns aligned to all K combos
    clf.prior_                  # effective source prior the model was trained under
                                # (needed by SLD; uniform when class_weight='balanced')

cfg["type"] in {"lr", "ovr", "lda", "knn", "hgb", "nb"}.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.covariance import ledoit_wolf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def _weighted_prior(y, w, K):
    p = np.bincount(y, weights=w, minlength=K).astype(float)
    return p / p.sum()


def _balance(y, w, K):
    """class_weight='balanced' computed from the weighted label mass."""
    mass = np.bincount(y, weights=w, minlength=K)
    present = mass > 0
    cw = np.zeros(K)
    cw[present] = w.sum() / (present.sum() * mass[present])
    return w * cw[y], present


class _Base:
    def __init__(self, cfg, K, CM):
        self.cfg = cfg
        self.K = K
        self.CM = CM

    def _prep(self, y, w):
        w = np.ones(len(y)) if w is None else np.asarray(w, float)
        if self.cfg.get("class_weight") == "balanced":
            w, present = _balance(y, w, self.K)
            self.prior_ = present / present.sum()
        else:
            self.prior_ = _weighted_prior(y, w, self.K)
        return w

    def _full(self, Pc, classes):
        P = np.zeros((Pc.shape[0], self.K))
        P[:, classes] = Pc
        P = P + 1e-9
        return P / P.sum(1, keepdims=True)


class LRPowerset(_Base):
    """Multinomial logistic regression on the label powerset (T1)."""

    def fit(self, X, y, w=None):
        w = self._prep(y, w)
        self.m = LogisticRegression(C=float(self.cfg.get("C", 0.1)), l1_ratio=0, max_iter=5000,
                                    tol=1e-5).fit(X, y, sample_weight=w)
        return self

    def predict_proba(self, X):
        return self._full(self.m.predict_proba(X), self.m.classes_)


class OvRLR(_Base):
    """One-vs-rest LR on the 5 labels; combo probabilities = product of marginals renormalized
    over the K allowed combinations (so the output is always a valid label set)."""

    def fit(self, X, y, w=None):
        w0 = np.ones(len(y)) if w is None else np.asarray(w, float)
        self.prior_ = _weighted_prior(y, w0, self.K)
        B = self.CM[y]
        self.ms = []
        for j in range(5):
            yj = B[:, j]
            wj = w0.copy()
            if self.cfg.get("class_weight") == "balanced":
                pos = wj[yj == 1].sum()
                neg = wj[yj == 0].sum()
                if pos > 0 and neg > 0:
                    wj = np.where(yj == 1, wj * (pos + neg) / (2 * pos), wj * (pos + neg) / (2 * neg))
            if yj.min() == yj.max():
                self.ms.append(float(yj[0]))
                continue
            self.ms.append(LogisticRegression(C=float(self.cfg.get("C", 0.1)), l1_ratio=0, max_iter=5000,
                                              tol=1e-5).fit(X, yj, sample_weight=wj))
        if self.cfg.get("class_weight") == "balanced":
            self.prior_ = np.ones(self.K) / self.K
        return self

    def marginals(self, X):
        M = np.zeros((len(X), 5))
        for j, m in enumerate(self.ms):
            M[:, j] = m if isinstance(m, float) else m.predict_proba(X)[:, 1]
        return np.clip(M, 1e-6, 1 - 1e-6)

    def predict_proba(self, X):
        M = self.marginals(X)
        L = np.log(M) @ self.CM.T + np.log(1 - M) @ (1 - self.CM).T
        L -= L.max(1, keepdims=True)
        P = np.exp(L)
        return P / P.sum(1, keepdims=True)


class ShrinkLDA(_Base):
    """Shrinkage LDA (tied Gaussian classes) with sample-weight support.  Follows sklearn's
    solver='lsqr', shrinkage='auto': per-class Ledoit-Wolf covariances (on standardized
    features) averaged with the class priors."""

    def fit(self, X, y, w=None):
        w = self._prep(y, w)
        K, d = self.K, X.shape[1]
        self.classes_ = np.unique(y)
        means = np.zeros((K, d))
        S = np.zeros((d, d))
        prior = self.prior_
        for k in self.classes_:
            m = y == k
            wk = w[m]
            Xk = X[m]
            mu = (wk[:, None] * Xk).sum(0) / wk.sum()
            means[k] = mu
            Xc = (Xk - mu) * np.sqrt(wk / wk.mean())[:, None]
            if len(Xk) < 2:
                continue
            sd = Xc.std(0)
            sd[sd < 1e-12] = 1.0
            C = ledoit_wolf(Xc / sd, assume_centered=True)[0]
            S += prior[k] * (sd[:, None] * C * sd[None, :])
        S += 1e-8 * np.eye(d)
        self.means_ = means
        self.coef_ = np.linalg.solve(S, means[self.classes_].T).T
        self.intercept_ = -0.5 * (self.coef_ * means[self.classes_]).sum(1) + np.log(prior[self.classes_])
        return self

    def predict_proba(self, X):
        Lg = X @ self.coef_.T + self.intercept_
        Lg -= Lg.max(1, keepdims=True)
        P = np.exp(Lg)
        P /= P.sum(1, keepdims=True)
        return self._full(P, self.classes_)


class KNNClf(_Base):
    """Cosine kNN over combos with distance weighting and soft targets; smoothed toward the
    training prior with beta pseudo-counts so probabilities are never exactly 0."""

    def fit(self, X, y, w=None):
        w = self._prep(y, w)
        self.Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        self.Yw = np.eye(self.K)[y] * w[:, None]
        self.w = w
        return self

    def predict_proba(self, X, exclude_self=False):
        k = int(self.cfg.get("k", 10))
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        S = Xn @ self.Xn.T
        if exclude_self:
            np.fill_diagonal(S, -np.inf)
        k = min(k, S.shape[1] - (1 if exclude_self else 0))
        nn = np.argpartition(-S, k - 1, axis=1)[:, :k]
        sims = np.take_along_axis(S, nn, 1)
        if self.cfg.get("weights", "distance") == "distance":
            wt = np.exp(sims / float(self.cfg.get("tau", 0.1)))
        else:
            wt = np.ones_like(sims)
        votes = np.einsum("nk,nkc->nc", wt, self.Yw[nn])
        mass = np.einsum("nk,nk->n", wt, self.w[nn])
        beta = float(self.cfg.get("beta", 1.0))
        P = (votes + beta * self.prior_) / (mass + beta)[:, None]
        return P / P.sum(1, keepdims=True)


class HGB(_Base):
    """HistGradientBoosting over combos (mainly as a stacker on meta-features)."""

    def fit(self, X, y, w=None):
        w = self._prep(y, w)
        c = self.cfg
        self.m = HistGradientBoostingClassifier(
            learning_rate=float(c.get("learning_rate", 0.05)), max_iter=int(c.get("max_iter", 100)),
            max_leaf_nodes=int(c.get("max_leaf_nodes", 6)), max_depth=c.get("max_depth", 3),
            min_samples_leaf=int(c.get("min_samples_leaf", 20)),
            l2_regularization=float(c.get("l2_regularization", 1.0)),
            max_features=float(c.get("max_features", 1.0)), early_stopping=False,
            random_state=int(c.get("seed", 0))).fit(X, y, sample_weight=w)
        return self

    def predict_proba(self, X):
        return self._full(self.m.predict_proba(X), self.m.classes_)


class GaussNB(_Base):
    """Gaussian naive Bayes with variance smoothing (cheap, high-bias baseline)."""

    def fit(self, X, y, w=None):
        from sklearn.naive_bayes import GaussianNB
        w = self._prep(y, w)
        self.m = GaussianNB(var_smoothing=float(self.cfg.get("var_smoothing", 1e-2))).fit(X, y, sample_weight=w)
        return self

    def predict_proba(self, X):
        return self._full(self.m.predict_proba(X), self.m.classes_)


class PriorClf(_Base):
    """Predicts the training prior for every row (MAP -> always the majority combo)."""

    def fit(self, X, y, w=None):
        self._prep(y, w)
        return self

    def predict_proba(self, X):
        return np.tile(self.prior_, (len(X), 1))


REGISTRY = {"prior": PriorClf, "lr": LRPowerset, "ovr": OvRLR, "lda": ShrinkLDA, "knn": KNNClf, "hgb": HGB, "nb": GaussNB}


def make_classifier(cfg, K, CM):
    return REGISTRY[cfg["type"]](cfg, K, CM)
