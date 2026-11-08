"""Shared head: temperature calibration, ONE prior correction (SLD-EM, MAP-stabilized),
and threshold-free decoding to an allowed label set with >= 1 label.

All probabilities are over the K allowed label combinations (label powerset).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax

EPS = 1e-6


def safe_log(P):
    return np.log(np.clip(P, EPS, None))


def renorm(P):
    P = np.clip(P, 0, None) + 1e-12
    return P / P.sum(1, keepdims=True)


# ----------------------------------------------------------------------------- temperature
def fit_temperature(P_oof: np.ndarray, y: np.ndarray, w: np.ndarray | None = None,
                    bounds=(0.05, 20.0)) -> float:
    """One-parameter temperature T minimizing NLL of softmax(log P / T) on OOF probabilities."""
    L = safe_log(P_oof)
    w = np.ones(len(y)) if w is None else w

    def nll(logT):
        T = np.exp(logT)
        lp = log_softmax(L / T, axis=1)
        return -(w * lp[np.arange(len(y)), y]).sum() / w.sum()

    r = minimize_scalar(nll, bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded",
                        options={"xatol": 1e-4})
    return float(np.exp(r.x))


def apply_temperature(P: np.ndarray, T: float) -> np.ndarray:
    if T is None or T == 1.0:
        return P
    return np.exp(log_softmax(safe_log(P) / T, axis=1))


# ----------------------------------------------------------------------------- SLD-EM
def sld_em(P: np.ndarray, prior_src: np.ndarray, a: float | None = 0.0, floor: float = 0.25,
           tol: float = 1e-8, max_iter: int = 1000, cap: float | None = None):
    """Saerens-Latinne-Decaestecker EM for the target label prior, with a Dirichlet(a*pi_s)
    pseudo-count toward the source prior (MAP) and a floor pi >= floor*pi_s.

    cap: optional ceiling pi_k <= cap * pi_s,k (guards the degenerate MLE in which a rare
    'bridge' combo such as Safety+Efficacy absorbs the mass of its parents).
    Returns (Q, pi_hat): prior-corrected posteriors and the estimated target prior.
    a=None or a=inf means 'off' (returns P unchanged)."""
    if a is None or not np.isfinite(a):
        return P, prior_src.copy()
    present = prior_src > 1e-6            # combos absent from training stay at exactly 0
    ps = np.where(present, prior_src, 0.0)
    ps = ps / ps.sum()
    P = np.where(present[None, :], P, 0.0)
    P = P / np.maximum(P.sum(1, keepdims=True), 1e-300)
    ratio_den = np.where(present, ps, 1.0)
    pi = ps.copy()
    n = len(P)
    for _ in range(max_iter):
        Q = P * (pi / ratio_den)
        Q /= Q.sum(1, keepdims=True)
        new = (Q.sum(0) + a * ps) / (n + a)
        new = np.maximum(new, floor * ps)
        if cap:
            new = np.minimum(new, cap * ps)
        new /= new.sum()
        done = np.abs(new - pi).sum() < tol
        pi = new
        if done:
            break
    Q = P * (pi / ratio_den)
    Q /= Q.sum(1, keepdims=True)
    return Q, pi


def sld_em_factorized(P: np.ndarray, prior_src: np.ndarray, CM: np.ndarray, a: float | None = 0.0,
                      floor: float = 0.25, tol: float = 1e-8, max_iter: int = 1000):
    """Factorized SLD: the target/source combo-prior ratio is constrained to a product of
    per-label ratios, w_k = prod_j (pi_j/ps_j)^{CM_kj} ((1-pi_j)/(1-ps_j))^{1-CM_kj}.
    Only 5 free parameters (one per reason) instead of K-1, so a rare 'bridge' combo
    (Safety+Efficacy) cannot absorb the mass of its parents.  EM-style fixed point: E-step
    reweights combos by w, M-step sets each label prior to the mean marginal of Q (with the same
    Dirichlet pseudo-count a toward the source marginal and floor)."""
    if a is None or not np.isfinite(a):
        return P, prior_src.copy()
    present = prior_src > 1e-6
    ps_k = np.where(present, prior_src, 0.0)
    ps_k = ps_k / ps_k.sum()
    P = np.where(present[None, :], P, 0.0)
    P = P / np.maximum(P.sum(1, keepdims=True), 1e-300)
    ps = np.clip(ps_k @ CM, 1e-6, 1 - 1e-6)            # source label marginals
    pi = ps.copy()
    n = len(P)

    def weights(pi):
        lw = np.log(pi / ps) @ CM.T + np.log((1 - pi) / (1 - ps)) @ (1 - CM).T
        return np.exp(lw - lw.max())

    for _ in range(max_iter):
        Q = P * weights(pi)
        Q /= Q.sum(1, keepdims=True)
        m = Q @ CM                                          # (n, 5) label marginals
        new = (m.sum(0) + a * ps) / (n + a)
        new = np.clip(np.maximum(new, floor * ps), 1e-6, 1 - 1e-6)
        done = np.abs(new - pi).sum() < tol
        pi = new
        if done:
            break
    Q = P * weights(pi)
    Q /= Q.sum(1, keepdims=True)
    pi_k = ps_k * weights(pi)
    return Q, pi_k / pi_k.sum()


# ----------------------------------------------------------------------------- decoding
_SUBSETS = np.array([[(s >> j) & 1 for j in range(5)] for s in range(1, 32)])


def _f1_matrix(CM):
    inter = _SUBSETS @ CM.T                                   # (31, K)
    return 2 * inter / (_SUBSETS.sum(1)[:, None] + CM.sum(1)[None, :])


def decode(P: np.ndarray, CM: np.ndarray, method: str = "map") -> np.ndarray:
    """Map combo posteriors to a binary label matrix (n, 5); never empty.
    map   : the most probable allowed combination
    expf1 : Bayes-optimal subset for expected sample-F1 over the 31 non-empty subsets"""
    if method == "map":
        return CM[P.argmax(1)]
    if method == "expf1":
        F = _f1_matrix(CM)
        return _SUBSETS[(P @ F.T).argmax(1)]
    raise ValueError(method)


def apply_head(P_eval, prior_src, CM, head: dict, P_target=None):
    """SLD (optional) + decoding.  P_target: posteriors of the SLD target rows (defaults to the
    eval rows, i.e. transductive).  Returns (Ypred, Q_eval, pi_hat)."""
    head = head or {}
    pi_hat = None
    Q = P_eval
    if head.get("prior", "none") == "sld":
        a = head.get("sld_a", 0.0)
        floor = head.get("sld_floor", 0.25)
        cap = head.get("sld_cap")
        if head.get("sld_mode", "combo") == "factorized":
            em = lambda P_: sld_em_factorized(P_, prior_src, CM, a=a, floor=floor)
        else:
            em = lambda P_: sld_em(P_, prior_src, a=a, floor=floor, cap=cap)
        if P_target is None or P_target is P_eval:
            Q, pi_hat = em(P_eval)
        else:
            _, pi_hat = em(P_target)
            present = prior_src > 1e-6
            Q = renorm(np.where(present, P_eval * (pi_hat / np.where(present, prior_src, 1.0)), 0.0))
    return decode(Q, CM, head.get("decode", "map")), Q, pi_hat
