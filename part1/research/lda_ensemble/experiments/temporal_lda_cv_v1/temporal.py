"""Fold-local temporal adjustments using supplied Part 1 years and embeddings."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

SETTINGS = {
    "global_trend_half": "Remove half of training-input linear embedding drift with year; rebuild source",
    "within_reason_half": "Remove half of gold within-reason linear drift; rebuild source",
    "recent_lr_5": "Final source LR with five-year recency half-life and preserved class mass",
    "recent_lr_10": "Final source LR with ten-year recency half-life and preserved class mass",
    "time_knn_final": "Time-distance weighted cosine neighbors at final prediction only",
    "time_knn_joint": "Same time-aware neighbors in teaching and final prediction",
}


def years(frame):
    result = frame.start_year.to_numpy(float)
    assert np.isfinite(result).all()
    return result


def drift_geometry(packet, fit, name):
    cache = packet.setdefault("temporal_drift", {})
    if name not in cache:
        year = years(fit)
        center, scale = float(year.mean()), max(float(year.std()), 1.)
        time = (year-center)/scale
        z = packet["z"]
        if name=="within_reason_half":
            rows = packet["gold"]
            design = np.column_stack([np.ones(len(rows)), packet["y"]])
            # Remove reason-class and intercept effects before estimating drift.
            time_residual = time[rows]-design@np.linalg.lstsq(design, time[rows], rcond=None)[0]
            z_residual = z[rows]-design@np.linalg.lstsq(design, z[rows], rcond=None)[0]
            slope = time_residual@z_residual/(time_residual@time_residual+10.)
            fitting_ids = fit.iloc[rows].ID.tolist()
        else:
            slope = time@z/(time@time+10.)
            fitting_ids = fit.ID.tolist()
        transformed = z-.5*time[:,None]*slope[None,:]
        child = dict(z=transformed, gold=packet["gold"], y=packet["y"])
        cache[name] = dict(packet=child, slope=slope, center=center, scale=scale, fitting_ids=fitting_ids)
    return cache[name]


def time_prediction(base, zq, query_year, fit_year):
    lr = np.zeros((len(zq), 5)); lda = np.zeros_like(lr)
    lr[:,base["lr"].classes_] = base["lr"].predict_proba(zq)
    lda[:,base["lda"].classes_] = base["lda"].predict_proba(zq)
    similarity = normalize(zq)@base["neighbors"].T
    # Five-year half-life applies to the usual positive squared cosine weights.
    weighted = similarity.clip(0)**2*np.exp2(-abs(query_year[:,None]-fit_year[None,:])/5.)
    nearest = np.argsort(-weighted, axis=1)[:,:10]
    weight = np.take_along_axis(weighted, nearest, axis=1)
    knn = (weight[:,:,None]*base["targets"][nearest]).sum(1)/(weight.sum(1,keepdims=True)+1e-9)
    return .5*lr+.3*knn+.2*lda


def weighted_lr(z, targets, year, half_life):
    rows, labels, weights = [], [], []
    for i, target in enumerate(targets):
        positive = np.flatnonzero(target)
        rows.extend([i]*len(positive)); labels.extend(positive); weights.extend([1/len(positive)]*len(positive))
    rows, labels, weights = np.array(rows), np.array(labels), np.array(weights)
    counts = np.bincount(labels, weights=weights, minlength=5)+1e-9
    native = weights*counts.mean()/counts[labels]
    age = np.exp2((year-year.max())/half_life).clip(.1, 1.)
    changed = native*age[rows]
    for label in np.unique(labels):
        mask = labels==label
        changed[mask] *= native[mask].sum()/changed[mask].sum()
    model = LogisticRegression(C=.1, max_iter=3000).fit(z[rows], labels, sample_weight=changed)
    audit = dict(weight_min=float(changed.min()), weight_max=float(changed.max()),
        class_weight_mass_original=np.bincount(labels, weights=native, minlength=5).tolist(),
        class_weight_mass_new=np.bincount(labels, weights=changed, minlength=5).tolist(),
        gold_and_pseudo_weighting=True, year_half_life=half_life)
    return model, audit


def fit_predict(packet, fit, query, name, source, features):
    zq = packet["scaler"].transform(query[features.EMB].to_numpy())
    fitting_year, query_year = years(fit), years(query)
    audit = dict(block=name, fitting_ids=fit.ID.tolist(), query_year_used=name in ["global_trend_half", "within_reason_half", "time_knn_final", "time_knn_joint"],
        query_inputs_used_for_fitting=False, metadata_field="start_year", fitting_year_range=[int(fitting_year.min()),int(fitting_year.max())])
    if name in ["global_trend_half", "within_reason_half"]:
        geometry = drift_geometry(packet, fit, name)
        q = zq-.5*((query_year-geometry["center"])/geometry["scale"])[:,None]*geometry["slope"][None,:]
        fitted = source.fitted_variant(geometry["packet"], source.SETTINGS["revision_full"])
        raw = source.predict_base(fitted["base"], q)
        audit.update(drift_fit_ids=geometry["fitting_ids"], slope_norm=float(np.linalg.norm(geometry["slope"])),
            history=fitted["history"], source_geometry_and_teacher_refitted=True)
    else:
        base = source.fitted_variant(packet, source.SETTINGS["revision_full"])
        if name.startswith("recent_lr_"):
            half_life = int(name.rsplit("_",1)[1])
            cache = packet.setdefault("temporal_lr", {})
            if name not in cache:
                cache[name] = weighted_lr(packet["z"][base["rows"]], base["targets"], fitting_year[base["rows"]], half_life)
            model, details = cache[name]
            old = np.zeros((len(query),5)); new = np.zeros_like(old)
            old[:,base["base"]["lr"].classes_] = base["base"]["lr"].predict_proba(zq)
            new[:,model.classes_] = model.predict_proba(zq)
            raw = source.predict_base(base["base"], zq)+.5*(new-old)
            audit.update(**details, classifier_fit_ids=fit.iloc[base["rows"]].ID.tolist())
        else:
            if name=="time_knn_joint":
                if "temporal_knn_joint" not in packet:
                    gold, y, z = packet["gold"],packet["y"],packet["z"]
                    unknown = np.setdiff1d(np.arange(len(z)),gold)
                    resp = source.geometry(packet)["resp"]
                    rows, targets, history = gold.copy(), y.copy(), []
                    for _ in range(2):
                        head = source.fit_base(z[rows], targets, .1, .2)
                        t = targets/targets.sum(1,keepdims=True)
                        rates = (resp[rows].T@t+t.mean(0))/(resp[rows].sum(0)[:,None]+1)
                        probability = .8*time_prediction(head,z[unknown],fitting_year[unknown],fitting_year[rows])+.2*resp[unknown]@rates
                        keep = probability.max(1)>=.8
                        rows = np.r_[gold,unknown[keep]]
                        targets = np.vstack([y,features.native.decide(probability,.3)[keep]])
                        history.append(dict(accepted=int(keep.sum()),reason_counts=targets[len(y):].sum(0).tolist()))
                    packet["temporal_knn_joint"] = dict(base=source.fit_base(z[rows],targets,.1,.2),rows=rows,history=history)
                base = packet["temporal_knn_joint"]
            raw = time_prediction(base["base"],zq,query_year,fitting_year[base["rows"]])
            audit.update(history=base["history"],neighbor_fit_ids=fit.iloc[base["rows"]].ID.tolist(),half_life=5)
    result = features.centered(raw.clip(0,1))
    return result, result.copy(), audit
