"""Recent-gold experts, trained only within each permitted Part 1 partition."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

CONFIG = {
    "recent5_half_all": (5, .5, False),
    "recent5_half_late": (5, .5, True),
    "recent5_quarter_late": (5, .25, True),
    "recent8_half_late": (8, .5, True),
}
SETTINGS = {name: dict(year_width=width, raw_probability_weight=weight,
    late_query_gate=gate, minimum_positive_support=3)
    for name, (width, weight, gate) in CONFIG.items()}


def subset(fit, features, width):
    known = fit[features.LABELS].ge(0).all(axis=1).to_numpy()
    gold = np.flatnonzero(known)
    year = fit.start_year.to_numpy(float)
    assert np.isfinite(year).all()
    truth = fit.iloc[gold][features.LABELS].to_numpy(int)
    latest = float(year[gold].max())
    cutoff = latest-width
    minimum = np.minimum(3, truth.sum(0))
    for label in range(5):
        positive_years = np.sort(year[gold][truth[:,label]==1])
        if len(positive_years):
            cutoff = min(cutoff, float(positive_years[-minimum[label]]))
    groups = pd.util.hash_pandas_object(fit[features.EMB].round(10), index=False).to_numpy()
    selected_groups = set(groups[gold[year[gold]>=cutoff]])
    selected = gold[np.array([g in selected_groups for g in groups[gold]])]
    excluded = np.setdiff1d(gold, selected)
    # Excluded gold is omitted, never converted to pseudo-label candidates.
    masked = np.flatnonzero(~known)
    masked = masked[~np.isin(groups[masked], groups[excluded])]
    indices = np.sort(np.r_[selected, masked])
    frame = fit.iloc[indices].reset_index(drop=True)
    counts = fit.iloc[selected][features.LABELS].sum().to_numpy(int)
    assert (counts>=minimum).all()
    assert not set(groups[selected]) & set(groups[excluded])
    assert not set(groups[masked]) & set(groups[excluded])
    audit = dict(latest_gold_year=latest, effective_cutoff=cutoff,
        expert_fitting_ids=frame.ID.tolist(), expert_gold_ids=fit.iloc[selected].ID.tolist(),
        excluded_gold_ids=fit.iloc[excluded].ID.tolist(),
        expert_gold_count=len(selected), expert_masked_count=len(masked),
        expert_reason_support=counts.tolist(), minimum_reason_support=minimum.tolist(),
        excluded_gold_groups_removed_from_masked=True, gold_targets_preserved=True)
    return frame, audit


def fit_predict(packet, fit, query, name, source, features):
    width, weight, gate = CONFIG[name]
    cache = packet.setdefault("recent_gold_experts", {})
    if width not in cache:
        frame, audit = subset(fit, features, width)
        scaler = StandardScaler().fit(frame[features.EMB].to_numpy())
        known = frame[features.LABELS].ge(0).all(axis=1).to_numpy()
        child = dict(scaler=scaler, z=scaler.transform(frame[features.EMB].to_numpy()),
            gold=np.flatnonzero(known), y=frame.loc[known,features.LABELS].to_numpy(int))
        trained = source.fitted_variant(child, source.SETTINGS["revision_full"])
        audit.update(history=trained["history"])
        cache[width] = dict(packet=child, model=trained, audit=audit)
    expert = cache[width]
    original = source.fitted_variant(packet, source.SETTINGS["revision_full"])
    raw = source.predict_base(original["base"], packet["scaler"].transform(query[features.EMB].to_numpy()))
    recent = source.predict_base(expert["model"]["base"], expert["packet"]["scaler"].transform(query[features.EMB].to_numpy()))
    active = query.start_year.to_numpy(float)>=expert["audit"]["latest_gold_year"]-1 if gate else np.ones(len(query),bool)
    probability = raw+weight*active[:,None]*(recent-raw)
    result = features.centered(probability)
    audit = dict(expert["audit"], block=name, settings=SETTINGS[name], fitting_ids=fit.ID.tolist(),
        query_inputs_used_for_fitting=False, late_query_gate_count=int(active.sum()),
        gate_year=expert["audit"]["latest_gold_year"]-1 if gate else None)
    return result, result.copy(), audit
