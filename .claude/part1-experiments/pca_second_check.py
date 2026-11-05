"""Compares GMM inputs with and without a second PCA. Run from part1/."""
import warnings
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
LABELS = ["Safety", "Funding", "Efficacy", "Recruitment", "Other"]
EMB = [f"emb_{i}" for i in range(150)]
train, val = pd.read_csv("part1_train.csv"), pd.read_csv("part1_val.csv")
raw = train[EMB].values
scaler = StandardScaler().fit(raw)
X, X_val = scaler.transform(raw), scaler.transform(val[EMB].values)
labeled = (train["Safety"] != -100).values
Y, Y_val = train.loc[labeled, LABELS].values, val[LABELS].values

corr = np.abs(np.corrcoef(raw, rowvar=False)[np.triu_indices(150, 1)])
print(f"Raw dimension sd {raw.std(0).min():.3f}-{raw.std(0).max():.3f} | mean abs correlation {corr.mean():.3f}, max {corr.max():.3f}")
print("Standardized cumulative variance at 10/20/50/100:",
      np.cumsum(PCA(svd_solver="full").fit(X).explained_variance_ratio_)[[9, 19, 49, 99]].round(3))


def reduced(n):
    p = PCA(n_components=n, svd_solver="full", random_state=0).fit(X)
    return p.transform(X), p.transform(X_val)


def predict_with_fallback(T):
    X_unl, probs = X[~labeled], np.zeros((len(X_val), 5))
    for j in range(5):
        m = ~np.isnan(T[:, j])
        clf = LogisticRegression(C=1, max_iter=2000, class_weight="balanced")
        clf.fit(np.vstack([X[labeled], X_unl[m]]), np.r_[Y[:, j], T[m, j]].astype(int))
        probs[:, j] = clf.predict_proba(X_val)[:, 1]
    out = (probs >= .5).astype(int)
    none = out.sum(1) == 0
    out[none, probs[none].argmax(1)] = 1
    return out


inputs = {"standardized 150, no PCA": (X, X_val), "PCA keeping 150": reduced(150), "PCA 100": reduced(100),
          "PCA 50": reduced(50), "PCA 20": reduced(20), "PCA 10": reduced(10)}
rows = []
for name, (A, B) in inputs.items():
    for k in [2, 4, 8]:
        for seed in range(5):
            gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=seed).fit(A)
            P = gmm.predict_proba(A)
            R, base = P[labeled], Y.mean(0)
            rates = (R.T @ Y + 5 * base) / (R.sum(0, keepdims=True).T + 5)
            lp, vp = (P @ rates)[~labeled], gmm.predict_proba(B) @ rates
            T = np.full(lp.shape, np.nan)
            T[lp >= .7], T[lp <= base / 2] = 1, 0
            pred = predict_with_fallback(T)
            rows.append({"input": name, "k": k, "auc": np.mean([roc_auc_score(Y_val[:, j], vp[:, j]) for j in range(5)]),
                         "sample_f1": f1_score(Y_val, pred, average="samples", zero_division=0),
                         "macro_f1": f1_score(Y_val, pred, average="macro", zero_division=0)})
print(pd.DataFrame(rows).groupby(["k", "input"], sort=False).mean().round(3).to_string())
