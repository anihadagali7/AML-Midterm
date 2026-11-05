import warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, silhouette_score
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
Y_all = np.vstack([Y, Y_val])
n_lab = labeled.sum()
X_all = np.vstack([X_tr[labeled], X_va])
X_unl = X_tr[~labeled]
strength = 5
SPACES = {"pca20": (Z_tr, Z_va), "emb150": (X_tr, X_va)}
KS = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40]
pd.set_option("display.width", 250, "display.max_columns", None, "display.max_rows", None)


def rates_from(assign, k, known_idx, Y_known):
    """Smoothed label rate per cluster, from known-label train rows only."""
    br = Y_known.mean(axis=0)
    a = assign[known_idx]
    counts = np.bincount(a, minlength=k).astype(float)
    pos = np.vstack([Y_known[a == c].sum(axis=0) if (a == c).any() else np.zeros(5) for c in range(k)])
    return (pos + strength * br) / (counts[:, None] + strength), counts, br


def clf(X, y, X_test):
    if len(np.unique(y)) < 2:
        return np.full(len(X_test), y[0])
    return LogisticRegression(C=1, max_iter=2000, class_weight="balanced").fit(X, y).predict(X_test)


def scores(Y_true, P):
    return {"sample": f1_score(Y_true, P, average="samples", zero_division=0),
            "macro": f1_score(Y_true, P, average="macro", zero_division=0),
            **dict(zip(LABELS, f1_score(Y_true, P, average=None, zero_division=0)))}


base = np.column_stack([clf(X_tr[labeled], Y[:, j], X_va) for j in range(5)])
print("BASELINE val", {k: round(float(v), 3) for k, v in scores(Y_val, base).items()})

# ---- Validation sweep ----
rows, gen = [], []
for sp, (A, B) in SPACES.items():
    for k in KS:
        for seed in range(5):
            km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(A)
            rates, counts, br = rates_from(km.labels_, k, idx, Y)
            lp, vp = rates[km.labels_], rates[km.predict(B)]
            aucs = [roc_auc_score(Y_val[:, j], vp[:, j]) for j in range(5)]
            r = {"space": sp, "k": k, "seed": seed, "silhouette": silhouette_score(A, km.labels_),
                 "inertia": km.inertia_, "min_labeled": counts.min(), "empty_of_labels": int((counts == 0).sum()),
                 "mean_auc": np.mean(aucs)}
            for hname, high in [("0.5", .5), ("0.7", .7)]:
                preds = []
                for j, l in enumerate(LABELS):
                    y = train[l].values.copy()
                    y[~labeled & (lp[:, j] >= high)] = 1
                    y[~labeled & (lp[:, j] <= br[j] / 2)] = 0
                    keep = y != -100
                    preds.append(clf(X_tr[keep], y[keep], X_va))
                    above, below = vp[:, j] >= high, vp[:, j] <= br[j] / 2
                    gen.append({"space": sp, "k": k, "seed": seed, "high": hname, "label": l, "auc": aucs[j],
                                "added_pos": int((~labeled & (lp[:, j] >= high)).sum()),
                                "added_neg": int((~labeled & (lp[:, j] <= br[j] / 2)).sum()),
                                "val_above": int(above.sum()), "val_correct": int(Y_val[above, j].sum()),
                                "val_below": int(below.sum()), "val_missed": int(Y_val[below, j].sum())})
                s = scores(Y_val, np.column_stack(preds))
                r[f"sample_{hname}"], r[f"macro_{hname}"] = s["sample"], s["macro"]
                for l in LABELS:
                    r[f"{l}_{hname}"] = s[l]
            rows.append(r)
df = pd.DataFrame(rows)
cols = ["silhouette", "inertia", "min_labeled", "empty_of_labels", "mean_auc", "sample_0.5", "macro_0.5", "sample_0.7", "macro_0.7"]
print("\nMean by space and k\n", df.groupby(["space", "k"])[cols].mean().round(3).to_string())
print("\nSD by space and k\n", df.groupby(["space", "k"])[["mean_auc", "sample_0.7", "macro_0.7"]].std().round(3).to_string())
print("\nPer-label F1, 0.7 rule\n", df.groupby(["space", "k"])[[f"{l}_0.7" for l in LABELS]].mean().round(3).to_string())
g = pd.DataFrame(gen)
for sp in SPACES:
    t = g[(g.space == sp) & (g.high == "0.7")].groupby(["k", "label"]).agg(
        pos=("added_pos", "mean"), a=("val_above", "sum"), c=("val_correct", "sum"))
    t["prec"] = (t.c / t.a).round(2)
    print(f"\n{sp}: mean generated 1s, 0.7 rule\n", t["pos"].unstack()[LABELS].round(0).to_string())
    print(f"{sp}: pooled validation precision of generated 1s\n", t["prec"].unstack()[LABELS].to_string())

# ---- Cluster profiles ----
for sp, k in [("pca20", 2), ("pca20", 5), ("pca20", 8), ("pca20", 15)]:
    A = SPACES[sp][0]
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(A)
    rates, counts, _ = rates_from(km.labels_, k, idx, Y)
    d = pd.DataFrame(rates, columns=LABELS).round(2)
    d.insert(0, "labeled", counts.astype(int))
    d.insert(0, "share", (np.bincount(km.labels_, minlength=k) / len(A)).round(3))
    print(f"\n{sp} k={k}\n", d.sort_values("share", ascending=False).to_string(index=False))

# ---- Repeated 5-fold CV over the 235 rows with true labels ----
ckpts = {(sp, k, s): KMeans(n_clusters=k, n_init=10, random_state=s).fit(SPACES[sp][0]).labels_
         for sp in SPACES for k in [2, 3, 4, 5, 8, 10, 20] for s in range(3)}
records = []
for rep in range(5):
    oof = {}
    for tr_i, te_i in KFold(5, shuffle=True, random_state=rep).split(X_all):
        Y_f, X_f, X_te = Y_all[tr_i], X_all[tr_i], X_all[te_i]
        oof.setdefault(("baseline", 0, 0), np.zeros_like(Y_all))[te_i] = np.column_stack(
            [clf(X_f, Y_f[:, j], X_te) for j in range(5)])
        tr_lab = tr_i[tr_i < n_lab]
        orig, Y_known = idx[tr_lab], Y_all[tr_lab]
        for (sp, k, s), assign in ckpts.items():
            rates, _, br = rates_from(assign, k, orig, Y_known)
            lp_u = rates[assign][~labeled]
            preds = []
            for j in range(5):
                pos, neg = lp_u[:, j] >= .7, lp_u[:, j] <= br[j] / 2
                X_j = np.vstack([X_f, X_unl[pos], X_unl[neg]])
                y_j = np.concatenate([Y_f[:, j], np.ones(pos.sum(), int), np.zeros(neg.sum(), int)])
                preds.append(clf(X_j, y_j, X_te))
            oof.setdefault((sp, k, s), np.zeros_like(Y_all))[te_i] = np.column_stack(preds)
    for key, P in oof.items():
        records.append({"rep": rep, "space": key[0], "k": key[1], "seed": key[2], **scores(Y_all, P)})
cv = pd.DataFrame(records)
b = cv[cv.space == "baseline"].set_index("rep")
print("\nCV BASELINE", b[["sample", "macro"] + LABELS].mean().round(3).to_dict())
o = cv[cv.space != "baseline"].copy()
o["d_sample"] = o["sample"] - o.rep.map(b["sample"])
o["d_macro"] = o["macro"] - o.rep.map(b["macro"])
print("\nCV change vs baseline, 0.7 rule\n", o.groupby(["space", "k"]).agg(
    sample=("sample", "mean"), macro=("macro", "mean"),
    d_sample=("d_sample", "mean"), d_sample_sd=("d_sample", "std"),
    d_macro=("d_macro", "mean"), d_macro_sd=("d_macro", "std"),
    **{l: (l, "mean") for l in LABELS}).round(3).to_string())
