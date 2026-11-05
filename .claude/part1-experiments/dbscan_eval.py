import warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
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
SPACES = {"pca20_euclidean": (Z_tr, Z_va, "euclidean"),
          "emb150_euclidean": (X_tr, X_va, "euclidean"),
          "emb150_cosine": (X_tr, X_va, "cosine")}
pd.set_option("display.width", 250, "display.max_columns", None, "display.max_rows", None)


def rates_from(assign, n_c, known_idx, Y_known):
    """Smoothed label rate per cluster. Noise rows (-1) don't contribute."""
    br = Y_known.mean(axis=0)
    a = assign[known_idx]
    counts = np.array([(a == c).sum() for c in range(n_c)], float)
    pos = np.vstack([Y_known[a == c].sum(axis=0) if (a == c).any() else np.zeros(5) for c in range(n_c)])
    return (pos + strength * br) / (counts[:, None] + strength), counts, br


def probs_for(assign, rates, br):
    """Noise rows fall back to the overall rate, so they stay inside the uncertain band."""
    out = np.tile(br, (len(assign), 1))
    mask = assign >= 0
    out[mask] = rates[assign[mask]]
    return out


def assign_new(db, A, B, eps, metric):
    """DBSCAN has no predict: join the nearest core point's cluster if within eps."""
    core = db.core_sample_indices_
    if len(core) == 0:
        return np.full(len(B), -1)
    d, i = NearestNeighbors(n_neighbors=1, metric=metric).fit(A[core]).kneighbors(B)
    out = db.labels_[core][i[:, 0]].copy()
    out[d[:, 0] > eps] = -1
    return out


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

# ---- Sweep ----
rows, gen, fits = [], [], {}
for sp, (A, B, metric) in SPACES.items():
    for ms in [3, 5, 10, 20]:
        kd = NearestNeighbors(n_neighbors=ms, metric=metric).fit(A).kneighbors(A)[0][:, -1]
        print(f"{sp} min_samples={ms}: k-distance quantiles 10/50/90/99 =", np.quantile(kd, [.1, .5, .9, .99]).round(3))
        for q in [.1, .25, .4, .5, .6, .7, .8, .9, .95, .99]:
            eps = float(np.quantile(kd, q))
            if eps <= 0:
                continue
            db = DBSCAN(eps=eps, min_samples=ms, metric=metric).fit(A)
            n_c = db.labels_.max() + 1
            r = {"space": sp, "min_samples": ms, "q": q, "eps": eps, "clusters": n_c,
                 "noise_share": (db.labels_ == -1).mean()}
            if n_c < 2:
                rows.append(r)
                continue
            rates, counts, br = rates_from(db.labels_, n_c, idx, Y)
            lp = probs_for(db.labels_, rates, br)
            va = assign_new(db, A, B, eps, metric)
            vp = probs_for(va, rates, br)
            aucs = [roc_auc_score(Y_val[:, j], vp[:, j]) for j in range(5)]
            sizes = np.bincount(db.labels_[db.labels_ >= 0])
            core_mask = db.labels_ >= 0
            r.update(largest_share=sizes.max() / len(A), labeled_clustered=int((db.labels_[idx] >= 0).sum()),
                     val_noise=(va == -1).mean(), clusters_no_labels=int((counts == 0).sum()),
                     silhouette=silhouette_score(A[core_mask], db.labels_[core_mask], metric=metric),
                     mean_auc=np.mean(aucs))
            preds = []
            for j, l in enumerate(LABELS):
                y = train[l].values.copy()
                pos, neg = ~labeled & (lp[:, j] >= .7), ~labeled & (lp[:, j] <= br[j] / 2)
                y[pos], y[neg] = 1, 0
                keep = y != -100
                preds.append(clf(X_tr[keep], y[keep], X_va))
                above, below = vp[:, j] >= .7, vp[:, j] <= br[j] / 2
                gen.append({"space": sp, "min_samples": ms, "q": q, "label": l, "added_pos": int(pos.sum()),
                            "added_neg": int(neg.sum()), "val_above": int(above.sum()),
                            "val_correct": int(Y_val[above, j].sum()), "val_below": int(below.sum()),
                            "val_missed": int(Y_val[below, j].sum())})
            r.update(scores(Y_val, np.column_stack(preds)))
            rows.append(r)
            fits[(sp, ms, q)] = (db.labels_.copy(), n_c)

df = pd.DataFrame(rows)
print("\nFull sweep\n", df.round(3).to_string(index=False))
valid = df[df.clusters >= 2].copy()
print("\nvalid configs", len(valid), "of", len(df))

# ---- Cluster profile for the configs with most labeled rows clustered and the best AUC ----
top = valid.sort_values("mean_auc", ascending=False).head(10)
sel = pd.concat([top, valid.sort_values("macro", ascending=False).head(6),
                 valid.sort_values("sample", ascending=False).head(6)]).drop_duplicates(["space", "min_samples", "q"])
print("\nConfigs sent to CV\n", sel[["space", "min_samples", "q", "eps", "clusters", "noise_share", "labeled_clustered", "mean_auc", "sample", "macro"]].round(3).to_string(index=False))

for _, r in sel.head(3).iterrows():
    key = (r.space, int(r.min_samples), r.q)
    labs, n_c = fits[key]
    rates, counts, _ = rates_from(labs, n_c, idx, Y)
    d = pd.DataFrame(rates, columns=LABELS).round(2)
    d.insert(0, "labeled", counts.astype(int))
    d.insert(0, "rows", np.bincount(labs[labs >= 0], minlength=n_c))
    print(f"\nProfile {key}, noise rows {(labs == -1).sum()}\n", d.sort_values("rows", ascending=False).head(12).to_string(index=False))
    print(pd.DataFrame(gen).query("space == @key[0] and min_samples == @key[1] and q == @key[2]").drop(columns=["space", "min_samples", "q"]).to_string(index=False))

# ---- Repeated 5-fold CV ----
records = []
for rep in range(5):
    oof = {}
    for tr_i, te_i in KFold(5, shuffle=True, random_state=rep).split(X_all):
        Y_f, X_f, X_te = Y_all[tr_i], X_all[tr_i], X_all[te_i]
        oof.setdefault("baseline", np.zeros_like(Y_all))[te_i] = np.column_stack(
            [clf(X_f, Y_f[:, j], X_te) for j in range(5)])
        tr_lab = tr_i[tr_i < n_lab]
        orig, Y_known = idx[tr_lab], Y_all[tr_lab]
        for _, r in sel.iterrows():
            key = (r.space, int(r.min_samples), r.q)
            labs, n_c = fits[key]
            rates, _, br = rates_from(labs, n_c, orig, Y_known)
            lp_u = probs_for(labs, rates, br)[~labeled]
            preds = []
            for j in range(5):
                pos, neg = lp_u[:, j] >= .7, lp_u[:, j] <= br[j] / 2
                X_j = np.vstack([X_f, X_unl[pos], X_unl[neg]])
                y_j = np.concatenate([Y_f[:, j], np.ones(pos.sum(), int), np.zeros(neg.sum(), int)])
                preds.append(clf(X_j, y_j, X_te))
            oof.setdefault(key, np.zeros_like(Y_all))[te_i] = np.column_stack(preds)
    for key, P in oof.items():
        k = key if key != "baseline" else ("baseline", 0, 0)
        records.append({"rep": rep, "space": k[0], "min_samples": k[1], "q": k[2], **scores(Y_all, P)})
cv = pd.DataFrame(records)
b = cv[cv.space == "baseline"].set_index("rep")
print("\nCV BASELINE", b[["sample", "macro"] + LABELS].mean().round(3).to_dict())
o = cv[cv.space != "baseline"].copy()
o["d_sample"] = o["sample"] - o.rep.map(b["sample"])
o["d_macro"] = o["macro"] - o.rep.map(b["macro"])
res = o.groupby(["space", "min_samples", "q"]).agg(
    d_sample=("d_sample", "mean"), d_sample_sd=("d_sample", "std"),
    d_macro=("d_macro", "mean"), d_macro_sd=("d_macro", "std"),
    **{l: (l, "mean") for l in LABELS}).round(3).reset_index()
res = res.merge(sel[["space", "min_samples", "q", "clusters", "noise_share", "sample", "macro"]].round(3), on=["space", "min_samples", "q"])
print("\nCV change vs baseline, sorted by macro\n", res.sort_values("d_macro", ascending=False).to_string(index=False))
