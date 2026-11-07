# Function definitions copied from the frozen LDA revision study; no runner code.
import numpy as np
from types import SimpleNamespace
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import models
f=SimpleNamespace(native=SimpleNamespace(decide=models.threshold))
def fit_base(x, y, C, lda_weight):
    rows, labels, weights = [], [], []
    for i, target in enumerate(y):
        positive = np.flatnonzero(target)
        rows.extend([i]*len(positive))
        labels.extend(positive)
        weights.extend([1./len(positive)]*len(positive))
    labels, weights = np.asarray(labels), np.asarray(weights)
    counts = np.bincount(labels, weights=weights, minlength=5)+1e-9
    weights = weights*counts.mean()/counts[labels]
    model = LogisticRegression(C=C, max_iter=3000).fit(x[rows], labels, sample_weight=weights)
    lda = None
    if lda_weight:
        k = len(np.unique(labels))
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=np.ones(k)/k).fit(x[rows], labels)
    return dict(lr=model, lda=lda, lda_weight=lda_weight,
                neighbors=normalize(x), targets=y/y.sum(axis=1, keepdims=True))

def predict_base(base, query):
    lr = np.zeros((len(query), 5))
    lr[:, base["lr"].classes_] = base["lr"].predict_proba(query)
    similarity = normalize(query) @ base["neighbors"].T
    nearest = np.argsort(-similarity, axis=1)[:, :10]
    weights = np.take_along_axis(similarity, nearest, axis=1).clip(0)**2
    knn = (weights[:, :, None]*base["targets"][nearest]).sum(axis=1)/(weights.sum(axis=1, keepdims=True)+1e-9)
    result = (1-.3-base["lda_weight"])*lr+.3*knn
    if base["lda"] is not None:
        lda = np.zeros_like(lr)
        lda[:, base["lda"].classes_] = base["lda"].predict_proba(query)
        result += base["lda_weight"]*lda
    return result

def geometry(packet):
    if "revision_geometry" not in packet:
        pc = PCA(30, random_state=0).fit(packet["z"])
        gmm = GaussianMixture(30, covariance_type="diag", n_init=3, random_state=0).fit(pc.transform(packet["z"]))
        assert gmm.converged_
        packet["revision_geometry"] = dict(pca=pc, gmm=gmm, resp=gmm.predict_proba(pc.transform(packet["z"])))
    return packet["revision_geometry"]

def fitted_variant(packet, settings):
    key = tuple(settings.values())
    cache = packet.setdefault("revision_models", {})
    if key in cache:
        return cache[key]
    z, gold, y = packet["z"], packet["gold"], packet["y"]
    unknown = np.setdiff1d(np.arange(len(z)), gold)
    resp = geometry(packet)["resp"]
    rows, targets = gold.copy(), y.copy()
    history = []
    for _ in range(2):
        base = fit_base(z[rows], targets, settings["C"], settings["teacher_lda"])
        t = targets/targets.sum(axis=1, keepdims=True)
        rates = (resp[rows].T@t+t.mean(axis=0))/(resp[rows].sum(axis=0)[:, None]+1)
        probability = .8*predict_base(base, z[unknown])+.2*(resp[unknown]@rates)
        keep = probability.max(axis=1)>=.8
        rows = np.r_[gold, unknown[keep]]
        targets = np.vstack([y, f.native.decide(probability, .3)[keep]])
        history.append(dict(accepted=int(keep.sum()), reason_counts=targets[len(y):].sum(axis=0).tolist()))
    result = dict(base=fit_base(z[rows], targets, settings["C"], settings["final_lda"]),
                  history=history, rows=rows, targets=targets)
    cache[key] = result
    return result
