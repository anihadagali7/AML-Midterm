"""Decision cutoffs learned from grouped inner OOF ensemble predictions."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import three_way_transductive as partners

SETTINGS = {
    "global": dict(per_label=False, recent=False),
    "per_label": dict(per_label=True, recent=False),
    "global_recent": dict(per_label=False, recent=True),
    "per_label_recent": dict(per_label=True, recent=True),
}
GRID = np.array([.3, .25, .35, .2, .4])
SEED = 20261106


def centered(probability, threshold):
    p = np.clip(probability, 1e-9, 1-1e-9)
    t = np.asarray(threshold)
    return p*(1-t)/(p*(1-t)+(1-p)*t)


def inner_predictions(packet, fit, source, features):
    if "decision_inner" in packet:
        return packet["decision_inner"]
    gold = packet["gold"]
    truth = packet["y"]
    groups = pd.util.hash_pandas_object(fit[features.EMB].round(10), index=False).to_numpy()
    unique, inverse = np.unique(groups[gold], return_inverse=True)
    group_y = np.array([truth[inverse==i].max(0) for i in range(len(unique))])
    splits = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    probability = np.full((len(gold),5), np.nan)
    partner_score = np.full_like(probability,np.nan)
    masked = np.setdiff1d(np.arange(len(fit)),gold)
    audits = []
    for fold, (a,b) in enumerate(splits.split(group_y,group_y)):
        train = np.flatnonzero(np.isin(inverse,a))
        held = np.flatnonzero(np.isin(inverse,b))
        allowed_masked = masked[~np.isin(groups[masked],groups[gold[held]])]
        indices = np.sort(np.r_[gold[train],allowed_masked])
        fitting = fit.iloc[indices].reset_index(drop=True)
        query = fit.iloc[gold[held]].drop(columns=features.LABELS).reset_index(drop=True)
        assert not set(groups[indices]) & set(groups[gold[held]])
        scaler = StandardScaler().fit(fitting[features.EMB].to_numpy())
        known = fitting[features.LABELS].ge(0).all(axis=1).to_numpy()
        child = dict(scaler=scaler,z=scaler.transform(fitting[features.EMB].to_numpy()),
            gold=np.flatnonzero(known), y=fitting.loc[known,features.LABELS].to_numpy(int))
        fitted = source.fitted_variant(child,source.SETTINGS["revision_full"])
        probability[held] = source.predict_base(fitted["base"],scaler.transform(query[features.EMB].to_numpy()))
        tri, svm = partners.partners_in_frame(fitting,query)
        partner_score[held] = .15*tri+.15*svm
        audits.append(dict(fold=fold, fit_ids=fitting.ID.tolist(), query_ids=query.ID.tolist(),
            group_separation=True, masked_duplicates_excluded=len(masked)-len(allowed_masked)))
    assert np.isfinite(probability).all() and np.isfinite(partner_score).all()
    packet["decision_inner"] = dict(probability=probability, partners=partner_score, folds=audits,
        truth=truth, ids=fit.iloc[gold].ID.tolist(), years=fit.iloc[gold].start_year.to_numpy(float))
    return packet["decision_inner"]


def objective(threshold, inner, settings):
    scores = .7*centered(inner["probability"],threshold)+inner["partners"]
    predicted = (scores>=.5).astype(int)
    empty = predicted.sum(1)==0
    predicted[np.flatnonzero(empty),scores[empty].argmax(1)] = 1
    truth = inner["truth"]
    f1 = 2*(truth*predicted).sum(1)/(truth.sum(1)+predicted.sum(1))
    year = inner["years"]
    weights = np.exp2((year-year.max())/5.).clip(.1,1.) if settings["recent"] else np.ones(len(year))
    mean = float(np.average(f1,weights=weights))
    displacement = ((np.asarray(threshold)-.3)/.1)**2
    penalty = .01*float(displacement.sum() if settings["per_label"] else displacement.mean())
    return mean-penalty, mean, penalty


def select_thresholds(inner,settings):
    threshold = np.full(5,.3)
    best = objective(threshold,inner,settings)
    history = []
    for _ in range(2 if settings["per_label"] else 1):
        for label in range(5) if settings["per_label"] else [None]:
            winner = threshold.copy()
            for value in GRID:
                candidate = threshold.copy()
                if label is None:
                    candidate[:] = value
                else:
                    candidate[label] = value
                record = objective(candidate,inner,settings)
                if record[0]>best[0]+1e-12:
                    winner,best = candidate,record
            threshold = winner
            history.append(dict(label=label,threshold=threshold.tolist(),objective=best[0],
                inner_weighted_sample_f1=best[1],penalty=best[2]))
    return threshold,history


def fit_predict(packet,fit,query,name,source,features):
    inner = inner_predictions(packet,fit,source,features)
    cache = packet.setdefault("decision_thresholds",{})
    if name not in cache:
        cache[name] = select_thresholds(inner,SETTINGS[name])
    threshold,history = cache[name]
    fitted = source.fitted_variant(packet,source.SETTINGS["revision_full"])
    raw = source.predict_base(fitted["base"],packet["scaler"].transform(query[features.EMB].to_numpy()))
    result = centered(raw,threshold)
    audit = dict(block=name,fitting_ids=fit.ID.tolist(),settings=SETTINGS[name],thresholds=threshold.tolist(),
        history=history,inner_folds=inner["folds"],inner_gold_ids=inner["ids"],inner_truth=inner["truth"].tolist(),
        inner_years=inner["years"].tolist(),inner_source_probabilities=inner["probability"].tolist(),
        inner_partner_scores=inner["partners"].tolist(),inner_objective="ensemble sample F1 minus .01 squared displacement per free cutoff",
        query_inputs_used_for_fitting=False,outer_labels_used_for_cutoff_fitting=False)
    return result,result.copy(),audit
