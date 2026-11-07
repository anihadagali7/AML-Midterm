"""Fold-local changes to neighbor geometry, leaving LR/LDA coordinates fixed."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

VIEWS = ["pca30", "partial_white", "remove1", "remove3"]


def transform(packet, query, view, features):
    zq = packet["scaler"].transform(query[features.EMB].to_numpy())
    if view == "identity":
        return packet["z"], zq
    if "neighbor_geometry_pca" not in packet:
        pc = PCA(n_components=150, svd_solver="full").fit(packet["z"])
        packet["neighbor_geometry_pca"] = (pc, pc.transform(packet["z"]))
    pc, a = packet["neighbor_geometry_pca"]
    b = pc.transform(zq)
    if view == "pca30":
        return a[:, :30], b[:, :30]
    if view == "partial_white":
        scale = np.maximum(pc.explained_variance_, 1e-8)**.25
        return a/scale, b/scale
    remove = 1 if view == "remove1" else 3
    return a[:, remove:], b[:, remove:]


def predict(base, zq, transformed_query, transformed_fit):
    lr = np.zeros((len(zq), 5))
    lr[:, base["lr"].classes_] = base["lr"].predict_proba(zq)
    lda = np.zeros_like(lr)
    lda[:, base["lda"].classes_] = base["lda"].predict_proba(zq)
    similarity = normalize(transformed_query) @ normalize(transformed_fit).T
    nearest = np.argsort(-similarity, axis=1)[:, :10]
    weights = np.take_along_axis(similarity, nearest, axis=1).clip(0)**2
    knn = (weights[:, :, None]*base["targets"][nearest]).sum(axis=1)/(weights.sum(axis=1, keepdims=True)+1e-9)
    return .5*lr+.3*knn+.2*lda


def fit_predict(packet, fit, query, name, source, features):
    view, scope = name.rsplit("_", 1)
    a, b = transform(packet, query, view, features)
    key = (view, scope)
    cache = packet.setdefault("geometry_variants", {})
    if key not in cache:
        if scope == "final":
            revised = source.fitted_variant(packet, source.SETTINGS["revision_full"])
            cache[key] = revised
        else:
            z, gold, y = packet["z"], packet["gold"], packet["y"]
            unknown = np.setdiff1d(np.arange(len(z)), gold)
            resp = source.geometry(packet)["resp"]
            rows, targets = gold.copy(), y.copy()
            history = []
            for _ in range(2):
                base = source.fit_base(z[rows], targets, .1, .2)
                t = targets/targets.sum(axis=1, keepdims=True)
                rates = (resp[rows].T@t+t.mean(axis=0))/(resp[rows].sum(axis=0)[:, None]+1)
                probability = .8*predict(base, z[unknown], a[unknown], a[rows])+.2*(resp[unknown]@rates)
                keep = probability.max(axis=1)>=.8
                rows = np.r_[gold, unknown[keep]]
                targets = np.vstack([y, features.native.decide(probability, .3)[keep]])
                history.append(dict(accepted=int(keep.sum()), reason_counts=targets[len(y):].sum(axis=0).tolist()))
            cache[key] = dict(base=source.fit_base(z[rows], targets, .1, .2), rows=rows, targets=targets, history=history)
    fitted = cache[key]
    zq = packet["scaler"].transform(query[features.EMB].to_numpy())
    score = features.centered(predict(fitted["base"], zq, b, a[fitted["rows"]]))
    return score, score.copy(), dict(block=name, view=view, scope=scope, neighbor_dimensions=a.shape[1],
        fitting_ids=fit.ID.tolist(), neighbors_fitting_ids=fit.iloc[fitted["rows"]].ID.tolist(),
        history=fitted["history"], original_LR_LDA_coordinates=True, geometry_fit_includes_query=False)
