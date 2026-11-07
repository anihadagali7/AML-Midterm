import warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
LABELS = ["Safety", "Funding", "Efficacy", "Recruitment", "Other"]
train = pd.read_csv("part1_train.csv")
val = pd.read_csv("part1_val.csv")
emb = [f"emb_{i}" for i in range(150)]
scaler = StandardScaler()
X_tr = scaler.fit_transform(train[emb])
X_va = scaler.transform(val[emb])
pca = PCA(n_components=20, random_state=0)
Z_tr = pca.fit_transform(X_tr)
labeled = (train["Safety"] != -100).values
strength = 5


def rates_and_probs(gmm, idx, Y_known):
    """Cluster label rates computed from known-label train rows only."""
    probs = gmm.predict_proba(Z_tr)
    R = probs[idx]
    br = Y_known.mean(axis=0)
    rates = (R.T @ Y_known + strength * br) / (R.sum(axis=0, keepdims=True).T + strength)
    return probs @ rates, br, rates, R


def clf(X, y, X_test):
    if len(np.unique(y)) < 2:
        return np.full(len(X_test), y[0])
    return LogisticRegression(C=1, max_iter=2000, class_weight="balanced").fit(X, y).predict(X_test)


# ---- What is k=2 doing? ----
idx = np.where(labeled)[0]
Y = train.loc[labeled, LABELS].values
gmm = GaussianMixture(n_components=2, covariance_type="diag", random_state=0).fit(Z_tr)
lp, br, rates, R = rates_and_probs(gmm, idx, Y)
print("k=2 weights", gmm.weights_.round(3), "labeled per cluster", R.sum(axis=0).round(1))
print("k=2 cluster rates\n", pd.DataFrame(rates, columns=LABELS).round(3).to_string())
print("base rates", dict(zip(LABELS, br.round(3))))
for j, l in enumerate(LABELS):
    print(l, "low", round(br[j] / 2, 3), "added 0s", int((~labeled & (lp[:, j] <= br[j] / 2)).sum()),
          "added 1s", int((~labeled & (lp[:, j] >= .7)).sum()))

# ---- Repeated 5-fold CV over all rows with true labels (157 train + 78 val) ----
n_lab, n_val = labeled.sum(), len(val)
Y_all = np.vstack([Y, val[LABELS].values])
print("\nlabeled rows", len(Y_all))
ks = [2, 3, 4, 5, 8, 10, 20]
gmms = {(k, s): GaussianMixture(n_components=k, covariance_type="diag", random_state=s).fit(Z_tr)
        for k in ks for s in range(3)}
X_unl = X_tr[~labeled]
X_all = np.vstack([X_tr[labeled], X_va])

records = []
for rep in range(5):
    oof = {}
    for tr_i, te_i in KFold(5, shuffle=True, random_state=rep).split(X_all):
        Y_fold = Y_all[tr_i]
        X_fold = X_all[tr_i]
        X_te = X_all[te_i]
        # baseline: true labels only
        oof.setdefault("baseline", np.zeros_like(Y_all))[te_i] = np.column_stack(
            [clf(X_fold, Y_fold[:, j], X_te) for j in range(5)])
        # Cluster rates use only training-fold labeled TRAIN rows
        tr_lab = tr_i[tr_i < n_lab]
        orig = idx[tr_lab]
        Y_known = Y_all[tr_lab]
        for (k, s), g in gmms.items():
            lp, br, _, _ = rates_and_probs(g, orig, Y_known)
            lp_u = lp[~labeled]
            for high_name, high in [("0.7", .7), ("zeros_only", np.inf)]:
                preds = []
                for j in range(5):
                    pos, neg = lp_u[:, j] >= high, lp_u[:, j] <= br[j] / 2
                    X_j = np.vstack([X_fold, X_unl[pos], X_unl[neg]])
                    y_j = np.concatenate([Y_fold[:, j], np.ones(pos.sum(), int), np.zeros(neg.sum(), int)])
                    preds.append(clf(X_j, y_j, X_te))
                oof.setdefault((k, s, high_name), np.zeros_like(Y_all))[te_i] = np.column_stack(preds)
    for key, P in oof.items():
        rec = {"rep": rep,
               "k": "baseline" if key == "baseline" else key[0],
               "seed": -1 if key == "baseline" else key[1],
               "high": "-" if key == "baseline" else key[2],
               "sample": f1_score(Y_all, P, average="samples", zero_division=0),
               "macro": f1_score(Y_all, P, average="macro", zero_division=0)}
        rec.update(zip(LABELS, f1_score(Y_all, P, average=None, zero_division=0)))
        records.append(rec)

df = pd.DataFrame(records)
pd.set_option("display.width", 250, "display.max_columns", None)
b = df[df.k == "baseline"].set_index("rep")
print("baseline", b[["sample", "macro"] + LABELS].mean().round(3).to_dict())
print("baseline sd across repeats", b[["sample", "macro"]].std().round(3).to_dict())
g = df[df.k != "baseline"].copy()
g["d_sample"] = g["sample"] - g.rep.map(b["sample"])
g["d_macro"] = g["macro"] - g.rep.map(b["macro"])
print(g.groupby(["high", "k"]).agg(sample=("sample", "mean"), macro=("macro", "mean"),
                                   d_sample=("d_sample", "mean"), d_sample_sd=("d_sample", "std"),
                                   d_macro=("d_macro", "mean"), d_macro_sd=("d_macro", "std"),
                                   **{l: (l, "mean") for l in LABELS}).round(3).to_string())
