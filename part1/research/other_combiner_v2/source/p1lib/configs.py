"""Named reference configurations (the report's five techniques, baselines and ensembles).

Every entry is a full experiment config: {"model": ..., "head": ..., "transductive": ...}.
Build new ones by copying and editing; `exp(model, head)` is a small helper.
"""
from __future__ import annotations

import copy

STD = {"scale": True}
PCA64 = {"scale": True, "pca": 64}
GRAPH = {"space": {"scale": True, "pca": 64, "l2": True}, "k": 10, "s": 1.0, "sym": "max"}

NO_HEAD = {"prior": "none", "decode": "map"}
SLD_HEAD = {"prior": "sld", "sld_a": 100, "decode": "map"}


def exp(model, head=None, transductive=True):
    return {"model": copy.deepcopy(model), "head": copy.deepcopy(head or NO_HEAD), "transductive": transductive}


def single(clf, features=STD, synth=None, labeler=None, temperature=False):
    return {"features": copy.deepcopy(features), "synth": copy.deepcopy(synth or []),
            "labeler": copy.deepcopy(labeler), "clf": copy.deepcopy(clf), "temperature": temperature}


LR = {"type": "lr", "C": 0.1}
LDA = {"type": "lda"}
OVR = {"type": "ovr", "C": 0.1}
KNN = {"type": "knn", "k": 10, "weights": "distance", "tau": 0.1, "beta": 1.0}

M_LR = single(LR)
M_LDA = single(LDA)
M_OVR = single(OVR)
M_KNN = single(KNN, features={"scale": True, "pca": 64, "l2": True})
M_CS = single({"type": "direct", "labeler": {"type": "cs", "graph": GRAPH, "a1": 0.5, "a2": 0.5}})
M_SPREAD = single({"type": "direct", "labeler": {"type": "spread", "graph": GRAPH, "alpha": 0.5}})
M_CLUSTER = single({"type": "direct", "labeler": {"type": "cluster", "algo": "kmeans", "k": 24, "m": 4}})
M_GMMCLUSTER = single({"type": "direct", "labeler": {"type": "cluster", "algo": "gmm", "k": 16, "cov": "diag", "m": 4}})

# T1 with SLD-filled pseudo-labels for the unlabeled pool
M_LR_SLDPL = single(LR, labeler={"type": "sld", "base": LR, "a": 100, "lam": 0.1})
# T2 C&S pseudo-labels feeding LR
M_LR_CSPL = single(LR, labeler={"type": "cs", "graph": GRAPH, "a1": 0.5, "a2": 0.5, "lam": 0.1})
# T4 cluster-then-label pseudo-labels feeding LR
M_LR_CLPL = single(LR, labeler={"type": "cluster", "algo": "kmeans", "k": 24, "m": 4, "lam": 0.1})
# T3 guarded self-training
M_LR_ST = single(LR, labeler={"type": "selftrain", "members": ["lr", "lda", "cs"], "rounds": 2, "tau": 0.75,
                              "lam": 0.3, "ramp": True, "align": "sld", "weighting": "softmatch",
                              "cluster_filter": True})
M_LR_ST_NAIVE = single(LR, labeler={"type": "selftrain", "members": ["lr"], "rounds": 3, "tau": 0.75,
                                    "lam": 0.5, "align": "none", "hard": True})
# synthetic-feature variants
M_LR_SYN = single(LR, synth=[{"type": "gmm", "k": 16, "cov": "diag"}, {"type": "knn_vote", "k": 10},
                             {"type": "graph", "mode": "spread", "graph": GRAPH, "alpha": 0.5},
                             {"type": "cluster_map", "algo": "kmeans", "k": 24, "m": 4}])
# T5 HGB stacker on nested-OOF meta-features
META = [{"type": "oof_model", "model": M_LR}, {"type": "oof_model", "model": M_LDA},
        {"type": "graph", "mode": "cs", "graph": GRAPH, "alpha": 0.5},
        {"type": "knn_vote", "k": 10}, {"type": "cluster_map", "algo": "kmeans", "k": 24, "m": 4}]
HGB = {"type": "hgb", "learning_rate": 0.05, "max_iter": 100, "max_leaf_nodes": 4, "max_depth": 2,
       "min_samples_leaf": 20, "l2_regularization": 1.0}
M_STACK_HGB = single(HGB, features=None, synth=META)
M_STACK_LR = single({"type": "lr", "C": 0.1}, features=None, synth=META)
M_HGB_RAW = single(HGB, features=STD)

# report's recommended ensemble: T1-LR + T1-LDA + T2-C&S, equal weights
M_ENS3 = {"ensemble": [M_LR, M_LDA, M_CS], "weights": None}

M_PRIOR = single({"type": "prior"})
M_NB = single({"type": "nb"}, features=PCA64)
SLD_EF1 = {"prior": "sld", "sld_a": 25, "decode": "expf1"}

REFERENCE = {
    "Majority": exp(M_PRIOR),
    "T1_LR": exp(M_LR),
    "T1_LR+SLD": exp(M_LR, SLD_HEAD),
    "T1_LDA": exp(M_LDA),
    "T1_LDA+SLD": exp(M_LDA, SLD_HEAD),
    "OvR_LR": exp(M_OVR),
    "kNN": exp(M_KNN),
    "T2_CS": exp(M_CS),
    "T2_CS+SLD": exp(M_CS, SLD_HEAD),
    "T2_spread": exp(M_SPREAD),
    "T4_kmeans_direct": exp(M_CLUSTER),
    "T4_gmm_direct": exp(M_GMMCLUSTER),
    "T1_LR+SLDpseudo": exp(M_LR_SLDPL),
    "T1_LR+SLDpseudo+SLD": exp(M_LR_SLDPL, SLD_HEAD),
    "T2_LR+CSpseudo": exp(M_LR_CSPL),
    "T4_LR+clusterpseudo": exp(M_LR_CLPL),
    "T3_selftrain_guarded": exp(M_LR_ST),
    "T3_selftrain_naive": exp(M_LR_ST_NAIVE),
    "LR+synth": exp(M_LR_SYN),
    "T5_stack_HGB": exp(M_STACK_HGB),
    "T5_stack_HGB+SLD": exp(M_STACK_HGB, SLD_HEAD),
    "stack_LR": exp(M_STACK_LR),
    "HGB_raw": exp(M_HGB_RAW),
    "ENS3": exp(M_ENS3),
    "ENS3+SLD": exp(M_ENS3, SLD_HEAD),
}
