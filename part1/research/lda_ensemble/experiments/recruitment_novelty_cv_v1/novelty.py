"""Gold-reference novelty features with leave-group-out training distances."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.special import expit, logit

SETTINGS = {family+"_"+prior: (family, prior)
            for family in ["lr", "lda"] for prior in ["empirical", "balanced"]}


def distance_features(z, normals, other, normal_groups, other_groups, exclude=None):
    normalized = normalize(z)
    sim = normalized @ normalize(normals).T
    other_sim = normalized @ normalize(other).T
    if exclude is not None:
        sim[exclude[:, None] == normal_groups[None, :]] = -np.inf
        other_sim[exclude[:, None] == other_groups[None, :]] = -np.inf
    values = []
    for i in range(len(z)):
        eligible = np.sort(sim[i][np.isfinite(sim[i])])
        assert len(eligible) > 0
        top = [float(eligible[-1]), float(eligible[-min(3, len(eligible)):].mean()),
               float(eligible[-min(10, len(eligible)):].mean())]
        known_other = other_sim[i][np.isfinite(other_sim[i])]
        maximum_other = float(known_other.max()) if len(known_other) else 0.
        values.append([*top, maximum_other, top[0]-maximum_other, top[0]-top[2], float(np.linalg.norm(z[i]))])
    return np.array(values)


def fit_predict(packet, fit, query, name, source, features):
    family, prior = SETTINGS[name]
    original, _, _ = source.fit_addition(packet, fit, query, "revision_full")
    if "recruitment_novelty_data" not in packet:
        groups = pd.util.hash_pandas_object(fit[features.EMB].round(10), index=False).to_numpy()
        gold, y = packet["gold"], packet["y"]
        normal = gold[(y[:, 3] == 1)&(y[:, 4] == 0)]
        other = gold[(y[:, 4] == 1)&(y[:, 3] == 0)]
        rows = gold[y[:, 3] == 1]
        target = y[y[:, 3] == 1, 4]
        x = distance_features(packet["z"][rows], packet["z"][normal], packet["z"][other], groups[normal], groups[other], groups[rows])
        scaler = StandardScaler().fit(x)
        packet["recruitment_novelty_data"] = dict(rows=rows, normal=normal, other=other, groups=groups,
            target=target, x=scaler.transform(x), scaler=scaler)
    data = packet["recruitment_novelty_data"]
    cache = packet.setdefault("recruitment_novelty_heads", {})
    if name not in cache:
        y = data["target"]
        if len(np.unique(y)) < 2:
            model = float(y[0]) if len(y) else 0.
        elif family == "lr":
            model = LogisticRegression(C=1., max_iter=3000,
                class_weight="balanced" if prior=="balanced" else None).fit(data["x"], y)
        else:
            model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto",
                priors=[.5, .5] if prior=="balanced" else None).fit(data["x"], y)
        cache[name] = model
    model = cache[name]
    zq = packet["scaler"].transform(query[features.EMB].to_numpy())
    q = distance_features(zq, packet["z"][data["normal"]], packet["z"][data["other"]],
        data["groups"][data["normal"]], data["groups"][data["other"]])
    probability = np.full(len(query), model) if isinstance(model, float) else model.predict_proba(data["scaler"].transform(q))[:, 1]
    cutoff = .8 if prior=="balanced" else .5
    head = expit(logit(np.clip(probability, 1e-9, 1-1e-9))-logit(cutoff))
    result = original.copy()
    gate = original[:, 3] >= .5
    result[gate, 4] = np.maximum(original[gate, 4], head[gate])
    return result, np.column_stack([original[:, :4], probability]), dict(block=name,
        fitting_ids=fit.iloc[data["rows"]].ID.tolist(), reference_normal_ids=fit.iloc[data["normal"]].ID.tolist(),
        reference_other_ids=fit.iloc[data["other"]].ID.tolist(), conditional_training_rows=len(data["rows"]),
        conditional_positives=int(data["target"].sum()), training_distance_group_exclusion=True,
        query_references_all_permitted_fitting_gold=True, cutoff=cutoff, gated_query_rows=int(gate.sum()),
        pseudo_labels_used=False, feature_count=7)


def check_exclusion():
    z = np.array([[1., 0.], [1., 0.], [0., 1.]])
    groups = np.array([1, 1, 2])
    x = distance_features(z[:1], z, z[2:], groups, groups[2:], groups[:1])
    assert x[0, 0] == 0.
    q = distance_features(z[:1], z, z[2:], groups, groups[2:])
    assert q[0, 0] == 1.
