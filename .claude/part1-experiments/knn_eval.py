import warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
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
Y = train.loc[labeled, LABELS].values
Y_val = val[LABELS].values
Y_all = np.vstack([Y, Y_val])
n_lab = labeled.sum()

SPACES = {
    "emb150_euclidean": (X_tr, X_va, "euclidean"),
    "emb150_cosine": (X_tr, X_va, "cosine"),
    "pca20_euclidean": (Z_tr, Z_va, "euclidean"),
}
pd.set_option("display.width", 250, "display.max_columns", None, "display.max_rows", None)


def knn_probs(ref, Y_ref, query, k, metric, weights):
    """Share of each row's k nearest labeled neighbors that have each label."""
    dist, ind = NearestNeighbors(n_neighbors=min(k, len(ref)), metric=metric).fit(ref).kneighbors(query)
    w = np.ones_like(dist) if weights == "uniform" else 1 / (dist + 1e-9)
    w = w / w.sum(axis=1, keepdims=True)
    return (w[:, :, None] * Y_ref[ind]).sum(axis=1)


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

# ---- 1. kNN as the classifier, on validation ----
rows = []
for sp, (A, B, metric) in SPACES.items():
    for k in [1, 3, 5, 10, 15, 25]:
        for wt in ["uniform", "distance"]:
            vp = knn_probs(A[labeled], Y, B, k, metric, wt)
            aucs = [roc_auc_score(Y_val[:, j], vp[:, j]) for j in range(5)]
            rows.append({"space": sp, "k": k, "weights": wt, "mean_auc": np.mean(aucs),
                         **scores(Y_val, (vp >= .5).astype(int))})
direct = pd.DataFrame(rows)
print("\nDIRECT kNN on validation (threshold 0.5), distance weights\n",
      direct[direct.weights == "distance"].round(3).to_string(index=False))
print("\nDIRECT kNN on validation, uniform weights\n",
      direct[direct.weights == "uniform"].round(3).to_string(index=False))

# ---- 2. kNN generated labels, then retrain LR, on validation ----
rows, gen = [], []
br = Y.mean(axis=0)
for sp, (A, B, metric) in SPACES.items():
    for k in [3, 5, 10, 15]:
        lp = knn_probs(A[labeled], Y, A[~labeled], k, metric, "distance")
        vp = knn_probs(A[labeled], Y, B, k, metric, "distance")
        for hname, high in [("0.5", .5), ("0.7", .7), ("0.9", .9)]:
            preds = []
            for j, l in enumerate(LABELS):
                pos, neg = lp[:, j] >= high, lp[:, j] <= br[j] / 2
                X_j = np.vstack([X_tr[labeled], X_tr[~labeled][pos], X_tr[~labeled][neg]])
                y_j = np.concatenate([Y[:, j], np.ones(pos.sum(), int), np.zeros(neg.sum(), int)])
                preds.append(clf(X_j, y_j, X_va))
                above, below = vp[:, j] >= high, vp[:, j] <= br[j] / 2
                gen.append({"space": sp, "k": k, "high": hname, "label": l,
                            "added_pos": int(pos.sum()), "added_neg": int(neg.sum()),
                            "val_above": int(above.sum()), "val_correct": int(Y_val[above, j].sum()),
                            "val_below": int(below.sum()), "val_missed": int(Y_val[below, j].sum()),
                            "f1": f1_score(Y_val[:, j], preds[-1], zero_division=0)})
            rows.append({"space": sp, "k": k, "high": hname, **scores(Y_val, np.column_stack(preds))})
pseudo = pd.DataFrame(rows)
print("\nkNN LABELS + LR retrain on validation\n", pseudo.round(3).to_string(index=False))
gen = pd.DataFrame(gen)

# ---- 3. Repeated 5-fold CV over the 235 rows with true labels ----
X_all = np.vstack([X_tr[labeled], X_va])
Z_all = np.vstack([Z_tr[labeled], Z_va])
ALL = {"emb150_euclidean": (X_all, X_tr[~labeled], "euclidean"),
       "emb150_cosine": (X_all, X_tr[~labeled], "cosine"),
       "pca20_euclidean": (Z_all, Z_tr[~labeled], "euclidean")}
X_unl = X_tr[~labeled]
records = []
for rep in range(5):
    oof = {}
    for tr_i, te_i in KFold(5, shuffle=True, random_state=rep).split(X_all):
        Y_f, X_f, X_te = Y_all[tr_i], X_all[tr_i], X_all[te_i]
        oof.setdefault(("baseline", "-", "-", "-"), np.zeros_like(Y_all))[te_i] = np.column_stack(
            [clf(X_f, Y_f[:, j], X_te) for j in range(5)])
        b = Y_f.mean(axis=0)
        for sp, (A, U, metric) in ALL.items():
            for k in [1, 3, 5, 10, 15, 25]:
                vp = knn_probs(A[tr_i], Y_f, A[te_i], k, metric, "distance")
                oof.setdefault(("direct", sp, k, "-"), np.zeros_like(Y_all))[te_i] = (vp >= .5).astype(int)
            for k in [3, 5, 10]:
                lp = knn_probs(A[tr_i], Y_f, U, k, metric, "distance")
                for hname, high in [("0.7", .7), ("0.9", .9)]:
                    preds = []
                    for j in range(5):
                        pos, neg = lp[:, j] >= high, lp[:, j] <= b[j] / 2
                        X_j = np.vstack([X_f, X_unl[pos], X_unl[neg]])
                        y_j = np.concatenate([Y_f[:, j], np.ones(pos.sum(), int), np.zeros(neg.sum(), int)])
                        preds.append(clf(X_j, y_j, X_te))
                    oof.setdefault(("pseudo", sp, k, hname), np.zeros_like(Y_all))[te_i] = np.column_stack(preds)
    for key, P in oof.items():
        records.append({"rep": rep, "mode": key[0], "space": key[1], "k": str(key[2]), "high": key[3], **scores(Y_all, P)})
cv = pd.DataFrame(records)
b = cv[cv["mode"] == "baseline"].set_index("rep")
print("\nCV BASELINE", b[["sample", "macro"] + LABELS].mean().round(3).to_dict())
o = cv[cv["mode"] != "baseline"].copy()
o["d_sample"] = o["sample"] - o.rep.map(b["sample"])
o["d_macro"] = o["macro"] - o.rep.map(b["macro"])
s = o.groupby(["mode", "space", "k", "high"]).agg(
    sample=("sample", "mean"), macro=("macro", "mean"),
    d_sample=("d_sample", "mean"), d_sample_sd=("d_sample", "std"),
    d_macro=("d_macro", "mean"), d_macro_sd=("d_macro", "std"),
    **{l: (l, "mean") for l in LABELS}).round(3)
print("\nCV results\n", s.to_string())

# ---- 4. Generated-label quality for the best CV pseudo-label config ----
best = s.loc["pseudo"].sort_values("d_macro", ascending=False).head(3)
print("\nTop pseudo configs by CV macro\n", best.to_string())
sp, k, high = best.index[0]
print(f"\nGenerated labels for {sp}, k={k}, high={high} (validation check, with labeled train rows as neighbors)")
print(gen[(gen.space == sp) & (gen.k == int(k)) & (gen.high == high)]
      .drop(columns=["space", "k", "high"]).round(3).to_string(index=False))
print("\nval row for that config\n", pseudo[(pseudo.space == sp) & (pseudo.k == int(k)) & (pseudo.high == high)].round(3).to_string(index=False))
