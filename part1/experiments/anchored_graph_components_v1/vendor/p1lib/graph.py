"""Symmetric RBF-weighted kNN graph over the feature rows a fold may use, and
normalized-Laplacian diffusion (Zhou et al. 2004 label spreading; Huang et al. 2021 C&S).

graph config: {"space": features-cfg, "k": 10, "s": 1.0, "sym": "max"|"min"}
Weights w_ij = exp(-d_ij^2 / (s * median kNN distance)^2) on cosine-normalized vectors,
symmetrized with max (union) or min (mutual kNN), S = D^-1/2 W D^-1/2.
Diffusion solves (I - alpha S) F = (1 - alpha) F0 exactly (LU factor cached per alpha).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import lu_factor, lu_solve
from sklearn.neighbors import NearestNeighbors

from .cache import memo
from .features import featurize

DEFAULT_GRAPH = {"space": {"scale": True, "pca": 64, "l2": True}, "k": 10, "s": 1.0, "sym": "max"}


class Graph:
    def __init__(self, Z_nodes, nodes, n_all, k, s, sym):
        self.nodes = nodes
        self.pos = np.full(n_all, -1)
        self.pos[nodes] = np.arange(len(nodes))
        self.Zn = Z_nodes / np.maximum(np.linalg.norm(Z_nodes, axis=1, keepdims=True), 1e-12)
        self.nn = NearestNeighbors(n_neighbors=k + 1).fit(self.Zn)
        dist, ind = self.nn.kneighbors(self.Zn)
        dist, ind = dist[:, 1:], ind[:, 1:]
        self.sigma = s * np.median(dist)
        n = len(nodes)
        rows = np.repeat(np.arange(n), k)
        W = sp.csr_matrix((np.exp(-(dist.ravel() ** 2) / self.sigma ** 2), (rows, ind.ravel())), shape=(n, n))
        W = W.maximum(W.T) if sym == "max" else W.minimum(W.T)
        deg = np.asarray(W.sum(1)).ravel()
        iso = deg <= 0
        if iso.any():                      # isolated nodes (possible with mutual kNN): self-loop
            W = W + sp.diags(iso.astype(float))
            deg = np.asarray(W.sum(1)).ravel()
        Dm = sp.diags(1 / np.sqrt(deg))
        self.S = (Dm @ W @ Dm).toarray()
        self.k = k
        self._lu = {}

    def propagate(self, F0, alpha):
        if alpha <= 0:
            return F0.copy()
        if alpha not in self._lu:
            self._lu[alpha] = lu_factor(np.eye(len(self.S)) - alpha * self.S)
        return (1 - alpha) * lu_solve(self._lu[alpha], F0)

    def extend(self, Z_rows, F_nodes):
        """Inductive extension for rows that are not graph nodes: RBF-weighted kNN average."""
        Zn = Z_rows / np.maximum(np.linalg.norm(Z_rows, axis=1, keepdims=True), 1e-12)
        dist, ind = self.nn.kneighbors(Zn, n_neighbors=self.k)
        w = np.exp(-dist ** 2 / self.sigma ** 2) + 1e-12
        return np.einsum("nk,nkc->nc", w, F_nodes[ind]) / w.sum(1, keepdims=True)


def get_graph(ctx, gcfg) -> Graph:
    g = dict(DEFAULT_GRAPH)
    g.update(gcfg or {})
    Z = featurize(ctx, g["space"])

    def build():
        return Graph(Z[ctx.unsup], ctx.unsup, ctx.D.n, int(g["k"]), float(g["s"]), g["sym"])

    return memo("graph", {k: g[k] for k in ("space", "k", "s", "sym")}, ctx.unsup, build), Z


def node_values(graph, Z, F_nodes, rows):
    """Read propagated values for arbitrary global rows (extend for non-nodes)."""
    out = np.zeros((len(rows), F_nodes.shape[1]))
    p = graph.pos[rows]
    m = p >= 0
    out[m] = F_nodes[p[m]]
    if (~m).any():
        out[~m] = graph.extend(Z[rows[~m]], F_nodes)
    return out


def label_spread(ctx, gcfg, alpha, train_rows, y_train, K, out_rows, base_all=None):
    """Pure label spreading (base_all=None) or the Smooth step of C&S (base_all = (n_all,K)
    base probabilities).  Training nodes are clamped to their one-hot labels in F0."""
    graph, Z = get_graph(ctx, gcfg)
    n = len(graph.nodes)
    F0 = np.zeros((n, K))
    if base_all is not None:
        F0 = base_all[graph.nodes].copy()
    p = graph.pos[train_rows]
    keep = p >= 0
    F0[p[keep]] = np.eye(K)[y_train[keep]]
    F = graph.propagate(F0, alpha)
    return node_values(graph, Z, F, out_rows)


def correct_and_smooth(ctx, gcfg, a1, a2, train_rows, y_train, P_oof_train, P_all, K, out_rows,
                       scale_mode="autoscale"):
    """Correct & Smooth (Huang et al. 2021).
    P_oof_train : base probabilities of the training rows, OUT-OF-FOLD (in-sample residuals ~0)
    P_all       : base probabilities for every global row (fit on all training rows)"""
    graph, Z = get_graph(ctx, gcfg)
    n = len(graph.nodes)
    p = graph.pos[train_rows]
    keep = p >= 0
    Ytr = np.eye(K)[y_train]
    base = P_all[graph.nodes].copy()
    if a1 > 0:
        E0 = np.zeros((n, K))
        E0[p[keep]] = (Ytr - P_oof_train)[keep]
        E = graph.propagate(E0, a1)
        if scale_mode == "autoscale":
            sigma = np.abs(E0[p[keep]]).sum(1).mean()
            scale = sigma / np.maximum(np.abs(E).sum(1, keepdims=True), 1e-12)
            base = base + scale * E
        else:
            base = base + float(scale_mode) * E
        base = np.clip(base, 0, None)
        base = base / np.maximum(base.sum(1, keepdims=True), 1e-12)
    F0 = base.copy()
    F0[p[keep]] = Ytr[keep]
    F = graph.propagate(F0, a2)
    out = node_values(graph, Z, F, out_rows)
    out = np.clip(out, 0, None) + 1e-9
    return out / out.sum(1, keepdims=True)
