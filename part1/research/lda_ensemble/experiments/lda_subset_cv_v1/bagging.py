"""Grouped 80%-gold subset averaging for the latest LDA pipeline."""
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def fit_members(packet, fit, source, features):
    if "lda_subset_members" in packet:
        return packet["lda_subset_members"]
    groups = pd.util.hash_pandas_object(fit[features.EMB].round(10), index=False).to_numpy()
    gold = packet["gold"]
    unique, inverse = np.unique(groups[gold], return_inverse=True)
    y = packet["y"]
    group_y = np.array([y[inverse==i].max(axis=0) for i in range(len(unique))])
    splitter = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=270118)
    members = []
    for train, held in splitter.split(group_y, group_y):
        excluded = unique[held]
        rows = np.flatnonzero(~np.isin(groups, excluded))
        subset = fit.iloc[rows].reset_index(drop=True)
        assert not set(groups[rows]) & set(excluded)
        member = features.train_native(subset)
        revised = source.fitted_variant(member, source.SETTINGS["revision_full"])
        audit = dict(fit_ids=subset.ID.tolist(), gold_ids=subset.iloc[member["gold"]].ID.tolist(),
            excluded_gold_ids=fit.iloc[gold[np.isin(inverse, held)]].ID.tolist(),
            gold_rows=len(member["gold"]), masked_rows=len(subset)-len(member["gold"]),
            history=revised["history"], all_excluded_groups_purged=True)
        members.append((member, revised, audit))
    packet["lda_subset_members"] = members
    return members


def fit_predict(packet, fit, query, name, source, features):
    blend, scope = name.split("_", 1)
    full_weight = .5 if blend == "half" else 0.
    members = fit_members(packet, fit, source, features)
    raw = []
    for member, revised, audit in members:
        assert not set(audit["fit_ids"]) & set(query.ID)
        zq = member["scaler"].transform(query[features.EMB].to_numpy())
        raw.append(source.predict_base(revised["base"], zq))
    bagged_raw = np.mean(raw, axis=0)
    full = source.fitted_variant(packet, source.SETTINGS["revision_full"])
    full_raw = source.predict_base(full["base"], packet["scaler"].transform(query[features.EMB].to_numpy()))
    combined = features.centered(full_weight*full_raw+(1-full_weight)*bagged_raw)
    if scope == "other":
        result = features.centered(full_raw)
        result[:, 4] = combined[:, 4]
    else:
        result = combined
    return result, features.centered(bagged_raw), dict(block=name, full_weight=full_weight,
        scope=scope, subsets=5, subset_seed=270118, fitting_ids=fit.ID.tolist(),
        members=[member[2] for member in members], probability_averaging_before_centering=True)
