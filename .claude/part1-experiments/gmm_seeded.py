import warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
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
Z_va = pca.transform(X_va)
labeled = (train["Safety"] != -100).values
idx = np.where(labeled)[0]
Y = train.loc[labeled, LABELS].values
Y_val = val[LABELS].values
strength = 5
pd.set_option("display.width", 250, "display.max_columns", None)


def seeded_fit(train_idx, Y_known, seed):
    """Fit a 5-component GMM starting each component at one label's mean embedding."""
    Z_known = Z_tr[train_idx]
    means = np.vstack([Z_known[Y_known[:, j] == 1].mean(axis=0) if (Y_known[:, j] == 1).any()
                       else Z_known.mean(axis=0) for j in range(5)])
    gmm = GaussianMixture(n_components=5, covariance_type="diag", means_init=means, random_state=seed).fit(Z_tr)
    return gmm, means


def rates_for(gmm, train_idx, Y_known):
    probs = gmm.predict_proba(Z_tr)
    R = probs[train_idx]
    br = Y_known.mean(axis=0)
    rates = (R.T @ Y_known + strength * br) / (R.sum(axis=0, keepdims=True).T + strength)
    return probs, R, br, rates


def clf(X, y, X_test):
    if len(np.unique(y)) < 2:
        return np.full(len(X_test), y[0])
    return LogisticRegression(C=1, max_iter=2000, class_weight="balanced").fit(X, y).predict(X_test)


def scores(Y_true, P):
    return {"sample": f1_score(Y_true, P, average="samples", zero_division=0),
            "macro": f1_score(Y_true, P, average="macro", zero_division=0),
            **dict(zip(LABELS, f1_score(Y_true, P, average=None, zero_division=0)))}


# ---- Cluster profiles and drift ----
for seed in [0, 1]:
    gmm, means = seeded_fit(idx, Y, seed)
    probs, R, br, rates = rates_for(gmm, idx, Y)
    d = pd.DataFrame(rates, columns=LABELS, index=[f"seed:{l}" for l in LABELS]).round(2)
    d.insert(0, "labeled", R.sum(axis=0).round(0))
    d.insert(0, "weight", gmm.weights_.round(3))
    d["drift"] = np.linalg.norm(gmm.means_ - means, axis=1).round(1)
    print(f"\nrandom_state={seed} (converged={gmm.converged_}, iters={gmm.n_iter_})\n", d.to_string())
print("\nTypical distance between starting means:",
      round(np.mean([np.linalg.norm(means[a] - means[b]) for a in range(5) for b in range(a + 1, 5)]), 1))

# ---- Validation, true-labels baseline vs seeded ----
base = np.column_stack([clf(X_tr[labeled], Y[:, j], X_va) for j in range(5)])
print("\nBASELINE val", {k: round(v, 3) for k, v in scores(Y_val, base).items()})

val_rows, gen_rows = [], []
for seed in range(5):
    gmm, _ = seeded_fit(idx, Y, seed)
    probs, R, br, rates = rates_for(gmm, idx, Y)
    lp = probs @ rates
    vp = gmm.predict_proba(Z_va) @ rates
    aucs = [roc_auc_score(Y_val[:, j], vp[:, j]) for j in range(5)]
    for hname, high in [("0.5", .5), ("0.7", .7), ("zeros_only", np.inf)]:
        preds = []
        for j, l in enumerate(LABELS):
            y = train[l].values.copy()
            y[~labeled & (lp[:, j] >= high)] = 1
            y[~labeled & (lp[:, j] <= br[j] / 2)] = 0
            k = y != -100
            preds.append(clf(X_tr[k], y[k], X_va))
            above, below = vp[:, j] >= high, vp[:, j] <= br[j] / 2
            gen_rows.append({"high": hname, "seed": seed, "label": l, "auc": aucs[j],
                             "added_pos": int((~labeled & (lp[:, j] >= high)).sum()),
                             "added_neg": int((~labeled & (lp[:, j] <= br[j] / 2)).sum()),
                             "val_above": int(above.sum()), "val_correct": int(Y_val[above, j].sum()),
                             "val_below": int(below.sum()), "val_missed": int(Y_val[below, j].sum())})
        val_rows.append({"high": hname, "seed": seed, **scores(Y_val, np.column_stack(preds))})

v = pd.DataFrame(val_rows)
print("\nSEEDED val mean by rule\n", v.drop(columns="seed").groupby("high").mean().round(3).to_string())
print("\nSEEDED val sd by rule\n", v.drop(columns="seed").groupby("high").std().round(3).to_string())
g = pd.DataFrame(gen_rows)
print("\nMean AUC per label\n", g[g.high == "0.5"].groupby("label")["auc"].mean().reindex(LABELS).round(3).to_string())
t = g.groupby(["high", "label"]).agg(added_pos=("added_pos", "mean"), added_neg=("added_neg", "mean"),
                                     val_above=("val_above", "sum"), val_correct=("val_correct", "sum"),
                                     val_below=("val_below", "sum"), val_missed=("val_missed", "sum"))
print("\nGenerated labels (counts pooled over 5 seeds for val columns)\n", t.to_string())

# ---- Repeated 5-fold CV over 235 rows with true labels; seeds use training-fold labels only ----
Y_all = np.vstack([Y, Y_val])
X_all = np.vstack([X_tr[labeled], X_va])
X_unl = X_tr[~labeled]
n_lab = labeled.sum()
records = []
for rep in range(5):
    oof = {}
    for tr_i, te_i in KFold(5, shuffle=True, random_state=rep).split(X_all):
        Y_fold, X_fold, X_te = Y_all[tr_i], X_all[tr_i], X_all[te_i]
        oof.setdefault("baseline", np.zeros_like(Y_all))[te_i] = np.column_stack(
            [clf(X_fold, Y_fold[:, j], X_te) for j in range(5)])
        tr_lab = tr_i[tr_i < n_lab]
        orig, Y_known = idx[tr_lab], Y_all[tr_lab]
        gmm, _ = seeded_fit(orig, Y_known, rep)
        probs, _, br, rates = rates_for(gmm, orig, Y_known)
        lp_u = (probs @ rates)[~labeled]
        for hname, high in [("0.5", .5), ("0.7", .7), ("zeros_only", np.inf)]:
            preds = []
            for j in range(5):
                pos, neg = lp_u[:, j] >= high, lp_u[:, j] <= br[j] / 2
                X_j = np.vstack([X_fold, X_unl[pos], X_unl[neg]])
                y_j = np.concatenate([Y_fold[:, j], np.ones(pos.sum(), int), np.zeros(neg.sum(), int)])
                preds.append(clf(X_j, y_j, X_te))
            oof.setdefault(hname, np.zeros_like(Y_all))[te_i] = np.column_stack(preds)
    for key, P in oof.items():
        records.append({"rep": rep, "rule": key, **scores(Y_all, P)})
cv = pd.DataFrame(records)
print("\nCV mean\n", cv.drop(columns="rep").groupby("rule").mean().round(3).to_string())
b = cv[cv.rule == "baseline"].set_index("rep")
o = cv[cv.rule != "baseline"].copy()
o["d_sample"] = o["sample"] - o.rep.map(b["sample"])
o["d_macro"] = o["macro"] - o.rep.map(b["macro"])
print("\nCV change vs baseline\n", o.groupby("rule").agg(d_sample=("d_sample", "mean"), d_sample_sd=("d_sample", "std"),
                                                       d_macro=("d_macro", "mean"), d_macro_sd=("d_macro", "std")).round(3).to_string())
