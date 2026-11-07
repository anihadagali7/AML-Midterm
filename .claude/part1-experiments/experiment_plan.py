import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
OUT = os.path.dirname(os.path.abspath(__file__)) + "/"
LABELS = ["Safety", "Funding", "Efficacy", "Recruitment", "Other"]
SEEDS = range(5)
REPS = 5
STRENGTH = 5
pd.set_option("display.width", 250, "display.max_columns", None, "display.max_rows", None)

train = pd.read_csv("part1_train.csv")
val = pd.read_csv("part1_val.csv")
emb = [f"emb_{i}" for i in range(150)]
scaler = StandardScaler().fit(train[emb])
X_tr, X_va = scaler.transform(train[emb]), scaler.transform(val[emb])
pca20 = PCA(n_components=20, random_state=0).fit(X_tr)
Z_tr, Z_va = pca20.transform(X_tr), pca20.transform(X_va)
labeled = (train["Safety"] != -100).values
lab_idx = np.where(labeled)[0]
n_lab = len(lab_idx)
Y = train.loc[labeled, LABELS].values
Y_val = val[LABELS].values
Y_all = np.vstack([Y, Y_val])


# ---------- Clusterings, fit once on train embeddings without labels ----------
def one_hot(a, k):
    out = np.zeros((len(a), k))
    m = a >= 0
    out[np.where(m)[0], a[m]] = 1
    return out


CL = {}
for k in [2, 4, 8, 20]:
    for s in SEEDS:
        g = GaussianMixture(n_components=k, covariance_type="diag", random_state=s).fit(Z_tr)
        CL[("gmm", k, s)] = (g.predict_proba(Z_tr), g.predict_proba(Z_va))
for k in [4, 10]:
    for s in SEEDS:
        km = KMeans(n_clusters=k, n_init=10, random_state=s).fit(Z_tr)
        CL[("km", k, s)] = (one_hot(km.labels_, k), one_hot(km.predict(Z_va), k))
kd = NearestNeighbors(n_neighbors=3, metric="cosine").fit(X_tr).kneighbors(X_tr)[0][:, -1]
eps = float(np.quantile(kd, .5))
db = DBSCAN(eps=eps, min_samples=3, metric="cosine").fit(X_tr)
nc = db.labels_.max() + 1
d_, i_ = NearestNeighbors(n_neighbors=1, metric="cosine").fit(X_tr[db.core_sample_indices_]).kneighbors(X_va)
db_va = db.labels_[db.core_sample_indices_][i_[:, 0]].copy()
db_va[d_[:, 0] > eps] = -1
CL[("db", nc, 0)] = (one_hot(db.labels_, nc), one_hot(db_va, nc))
print(f"DBSCAN eps={eps:.3f} clusters={nc} noise={(db.labels_ == -1).mean():.3f}", flush=True)

# ---------- Exact-duplicate keys ----------
key_tr = train[emb].round(6).astype(str).agg("|".join, axis=1).values
key_va = val[emb].round(6).astype(str).agg("|".join, axis=1).values

# ---------- Feature blocks ----------
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def fingerprint(field):
    if pd.isna(field):
        return np.zeros(1024)
    fps = [gen.GetFingerprintAsNumPy(m) for s in str(field).split(";") if (m := Chem.MolFromSmiles(s)) is not None]
    return np.max(fps, axis=0) if fps else np.zeros(1024)


def block_pca(A, B, n=20):
    s = StandardScaler().fit(A)
    p = PCA(n_components=n, random_state=0).fit(s.transform(A))
    a, b = p.transform(s.transform(A)), p.transform(s.transform(B))
    f = np.sqrt(n / a.var(axis=0).sum())
    return a * f, b * f


def block_text(A, B, vec, n=20):
    Ma, Mb = vec.fit_transform(A), vec.transform(B)
    svd = TruncatedSVD(n_components=n, random_state=0).fit(Ma)
    a, b = svd.transform(Ma), svd.transform(Mb)
    s = StandardScaler().fit(a)
    return s.transform(a), s.transform(b)


semi = lambda x: [t.strip() for t in str(x).split(";") if t.strip()]
y_s = StandardScaler().fit(train[["start_year"]])
BLOCK = {
    "year": (y_s.transform(train[["start_year"]]), y_s.transform(val[["start_year"]])),
    "morgan": block_pca(np.vstack(train["drug_smiles"].apply(fingerprint)), np.vstack(val["drug_smiles"].apply(fingerprint))),
    "pathways": block_text(train["drug_pathways"].fillna(""), val["drug_pathways"].fillna(""),
                           CountVectorizer(tokenizer=semi, token_pattern=None, binary=True, min_df=5)),
    "diseases": block_text(train["diseases_cleaned"].fillna(""), val["diseases_cleaned"].fillna(""), TfidfVectorizer(min_df=3)),
    "criteria": block_text(train["criteria"].fillna(""), val["criteria"].fillna(""),
                           TfidfVectorizer(min_df=5, max_features=20000, stop_words="english")),
}
p50 = PCA(n_components=50, random_state=0).fit(X_tr)


def features(name, seed):
    if name == "emb":
        return X_tr, X_va
    if name == "emb_raw":
        return train[emb].values, val[emb].values
    if name == "pca20":
        return Z_tr, Z_va
    if name == "pca50":
        return p50.transform(X_tr), p50.transform(X_va)
    if name == "l2":
        return normalize(X_tr), normalize(X_va)
    if name.startswith("emb+gmm"):
        P = CL[("gmm", int(name[7:]), seed)]
        return np.hstack([X_tr, P[0]]), np.hstack([X_va, P[1]])
    parts = ["year", "morgan", "pathways", "diseases", "criteria"] if name == "emb+all" else [name[4:]]
    return (np.hstack([X_tr] + [BLOCK[p][0] for p in parts]), np.hstack([X_va] + [BLOCK[p][1] for p in parts]))


# ---------- Labelers ----------
def cluster_p(cid, known, Yk):
    """P(label | row) for every train and val row, plus masks of rows in no cluster."""
    Ptr, Pva = CL[cid]
    R = Ptr[known]
    b = Yk.mean(axis=0)
    rates = (R.T @ Yk + STRENGTH * b) / (R.sum(axis=0, keepdims=True).T + STRENGTH)
    return Ptr @ rates, Pva @ rates, Ptr.sum(1) == 0, Pva.sum(1) == 0, b


def hard_rule(p, b, noise=None):
    T = np.full(p.shape, np.nan)
    T[p >= .7] = 1
    T[p <= b / 2] = 0
    if noise is not None:
        T[noise] = np.nan
    return T


def knn_p(ref, Yr, q, k, metric):
    d, i = NearestNeighbors(n_neighbors=min(k, len(ref)), metric=metric).fit(ref).kneighbors(q)
    w = 1 / (d + 1e-9)
    w = w / w.sum(1, keepdims=True)
    return (w[:, :, None] * Yr[i]).sum(1)


def label_all(lab, seed, known, Yk):
    """Returns (T_train_all, T_val, soft_scale). T holds 1, 0, NaN, or a soft probability."""
    if lab == "none":
        return None, None, None
    if lab.startswith("soft"):
        _, k, s = lab.split("_")
        a, b, _, _, _ = cluster_p(("gmm", int(k), seed), known, Yk)
        return a, b, float(s)
    if lab.startswith(("gmm", "km")):
        m = "gmm" if lab.startswith("gmm") else "km"
        a, b, _, _, base = cluster_p((m, int(lab[len(m):]), seed), known, Yk)
        return hard_rule(a, base), hard_rule(b, base), None
    if lab == "dbscan":
        a, b, na, nb, base = cluster_p(("db", nc, 0), known, Yk)
        return hard_rule(a, base, na), hard_rule(b, base, nb), None
    if lab == "knn10":
        base = Yk.mean(axis=0)
        return (hard_rule(knn_p(X_tr[known], Yk, X_tr, 10, "euclidean"), base),
                hard_rule(knn_p(X_tr[known], Yk, X_va, 10, "euclidean"), base), None)
    if lab == "dup":
        m = {}
        for kk, y in zip(key_tr[known], Yk):
            m.setdefault(kk, []).append(tuple(y))
        good = {kk: v[0] for kk, v in m.items() if len(set(v)) == 1}

        def f(keys):
            T = np.full((len(keys), 5), np.nan)
            for i, kk in enumerate(keys):
                if kk in good:
                    T[i] = good[kk]
            return T
        return f(key_tr), f(key_va), None
    if lab == "dup+gmm2":
        a1, b1, _ = label_all("dup", seed, known, Yk)
        a2, b2, _ = label_all("gmm2", seed, known, Yk)
        return np.where(np.isnan(a1), a2, a1), np.where(np.isnan(b1), b2, b1), None
    if lab.startswith("consensus"):
        vs = [label_all(x, seed, known, Yk) for x in ["gmm2", "km4", "dbscan"]]
        out = []
        for idx in [0, 1]:
            S = np.stack([v[idx] for v in vs])
            ones, zeros = (S == 1).sum(0), (S == 0).sum(0)
            need = 3 if lab == "consensus3" else 2
            T = np.full(S[0].shape, np.nan)
            T[(ones >= need) & (zeros == 0)] = 1
            T[(zeros >= need) & (ones == 0)] = 0
            out.append(T)
        return out[0], out[1], None
    raise ValueError(lab)


# ---------- Classifiers ----------
def lr_proba(Fk, Yk, Fu, T, scale, q, c, cw):
    out = np.zeros((len(q), 5))
    for j in range(5):
        if T is None:
            X, y, w = Fk, Yk[:, j], None
        elif scale is None:
            m = ~np.isnan(T[:, j])
            X, y, w = np.vstack([Fk, Fu[m]]), np.r_[Yk[:, j], T[m, j]], None
        else:
            p = T[:, j]
            X = np.vstack([Fk, Fu, Fu])
            y = np.r_[Yk[:, j], np.ones(len(p)), np.zeros(len(p))]
            w = np.r_[np.ones(len(Yk)), scale * p, scale * (1 - p)]
        if len(np.unique(y)) < 2:
            out[:, j] = float(y[0])
            continue
        clf = LogisticRegression(C=c, max_iter=2000, class_weight="balanced" if cw else None)
        out[:, j] = clf.fit(X, y.astype(int), sample_weight=w).predict_proba(q)[:, list(clf.classes_).index(1)]
    return out


def knn_proba(Fk, Yk, Fu, T, scale, q, k, metric):
    out = np.zeros((len(q), 5))
    for j in range(5):
        if T is None:
            X, y, s = Fk, Yk[:, j], np.ones(len(Fk))
        elif scale is None:
            m = ~np.isnan(T[:, j])
            X, y, s = np.vstack([Fk, Fu[m]]), np.r_[Yk[:, j], T[m, j]], np.ones(len(Fk) + m.sum())
        else:
            X, y, s = np.vstack([Fk, Fu]), np.r_[Yk[:, j], T[:, j]], np.r_[np.ones(len(Fk)), np.full(len(Fu), scale)]
        d, i = NearestNeighbors(n_neighbors=min(k, len(X)), metric=metric).fit(X).kneighbors(q)
        w = s[i] / (d + 1e-9)
        out[:, j] = (w * y[i]).sum(1) / w.sum(1)
    return out


def predict(cfg, Fk, Yk, Fu, T, scale, q):
    if cfg["clf"] == "lr":
        return lr_proba(Fk, Yk, Fu, T, scale, q, cfg["C"], cfg["cw"]) >= .5
    if cfg["clf"] == "knn":
        return knn_proba(Fk, Yk, Fu, T, scale, q, cfg["k"], cfg["metric"]) >= cfg["thr"]
    a = lr_proba(Fk, Yk, Fu, T, scale, q, cfg["C"], cfg["cw"])
    b = knn_proba(Fk, Yk, Fu, T, scale, q, cfg["k"], cfg["metric"])
    return (a + b) / 2 >= .5


def score(Yt, P):
    P = P.astype(int)
    return {"sample": f1_score(Yt, P, average="samples", zero_division=0),
            "macro": f1_score(Yt, P, average="macro", zero_division=0),
            **dict(zip(LABELS, f1_score(Yt, P, average=None, zero_division=0)))}


def evaluate_config(cfg):
    lab, feat = cfg["labeler"], cfg["feat"]
    stochastic = lab.startswith(("gmm", "km", "soft", "dup+", "consensus")) or feat.startswith("emb+gmm")
    # Validation, using only labeled train rows as true labels
    vals = []
    for s in (SEEDS if stochastic else [0]):
        Ftr, Fva = features(feat, s)
        Ta, _, sc = label_all(lab, s, lab_idx, Y)
        T = None if Ta is None else Ta[~labeled]
        vals.append(score(Y_val, predict(cfg, Ftr[labeled], Y, Ftr[~labeled], T, sc, Fva)))
    v = pd.DataFrame(vals)
    # Repeated 5-fold CV over all 235 rows with true labels
    cv = []
    for rep in range(REPS):
        s = rep % 5
        Ftr, Fva = features(feat, s)
        pool, Fu = np.vstack([Ftr[labeled], Fva]), Ftr[~labeled]
        oof = np.zeros_like(Y_all)
        for tr_i, te_i in KFold(5, shuffle=True, random_state=rep).split(pool):
            known = lab_idx[tr_i[tr_i < n_lab]]
            Yk = Y_all[tr_i[tr_i < n_lab]]
            Ta, _, sc = label_all(lab, s, known, Yk)
            T = None if Ta is None else Ta[~labeled]
            oof[te_i] = predict(cfg, pool[tr_i], Y_all[tr_i], Fu, T, sc, pool[te_i])
        cv.append(score(Y_all, oof))
    c = pd.DataFrame(cv)
    return {**{k: cfg.get(k) for k in ["name", "labeler", "feat", "clf"]}, "cfg": cfg,
            "val_sample": v["sample"].mean(), "val_sample_sd": v["sample"].std(ddof=0),
            "val_macro": v["macro"].mean(), **{f"val_{l}": v[l].mean() for l in LABELS},
            "cv_sample": c["sample"].mean(), "cv_macro": c["macro"].mean(),
            "cv_list": c[["sample", "macro"]].values, **{f"cv_{l}": c[l].mean() for l in LABELS}}


def run(cfgs, title, base=None):
    res = Parallel(n_jobs=-1)(delayed(evaluate_config)(c) for c in cfgs)
    df = pd.DataFrame(res)
    if base is None:
        base = df.iloc[0]["cv_list"]
    df["cv_d_sample"] = [(r - base)[:, 0].mean() for r in df["cv_list"]]
    df["cv_d_sample_sd"] = [(r - base)[:, 0].std(ddof=1) for r in df["cv_list"]]
    df["cv_d_macro"] = [(r - base)[:, 1].mean() for r in df["cv_list"]]
    df["cv_d_macro_sd"] = [(r - base)[:, 1].std(ddof=1) for r in df["cv_list"]]
    show = ["name", "val_sample", "val_sample_sd", "val_macro", "cv_sample", "cv_d_sample", "cv_d_sample_sd", "cv_macro", "cv_d_macro", "cv_d_macro_sd"] + [f"val_{l}" for l in LABELS]
    print(f"\n===== {title} =====\n", df[show].sort_values(["val_sample", "cv_sample"], ascending=False).round(3).to_string(index=False), flush=True)
    return df


def pick(df):
    """Highest validation sample F1; among those within 0.01, highest CV sample F1."""
    near = df[df.val_sample >= df.val_sample.max() - 0.01]
    return near.sort_values("cv_sample", ascending=False).iloc[0]


# ---------- Stage 1: classifier tuning on true labels only ----------
base_cfg = {"name": "LR C=1 balanced", "labeler": "none", "feat": "emb", "clf": "lr", "C": 1, "cw": True}
s1 = [base_cfg]
for c in [.01, .1, 1, 10]:
    for cw in [True, False]:
        if (c, cw) != (1, True):
            s1.append({"name": f"LR C={c} {'balanced' if cw else 'unweighted'}", "labeler": "none", "feat": "emb", "clf": "lr", "C": c, "cw": cw})
for k in [1, 3, 5, 10]:
    for m in ["euclidean", "cosine"]:
        for t in [.5, .3]:
            s1.append({"name": f"kNN k={k} {m} thr={t}", "labeler": "none", "feat": "emb", "clf": "knn", "k": k, "metric": m, "thr": t})
d1 = run(s1, "STAGE 1: classifiers on true labels")
BASE = d1.iloc[0]["cv_list"]
best_lr = pick(d1[d1.clf == "lr"])["cfg"]
best_knn = pick(d1[d1.clf == "knn"])["cfg"]
print("Selected LR:", best_lr["name"], "| Selected kNN:", best_knn["name"], flush=True)

# ---------- Stage 2: labeling methods ----------
HARD = ["gmm2", "gmm4", "gmm8", "km4", "km10", "dbscan", "knn10", "dup", "dup+gmm2", "consensus3", "consensus2"]
SOFT = [f"soft_{k}_{s}" for k in [4, 8, 20] for s in [.1, .5, 1.0]]
s2 = []
for lab in ["none"] + HARD + SOFT:
    for clf_cfg in [best_lr, best_knn]:
        s2.append({**clf_cfg, "labeler": lab, "name": f"{lab} -> {clf_cfg['name']}"})
d2 = run(s2, "STAGE 2: labeling methods x classifiers", BASE)

print("\nGenerated-label quality on validation rows (seed 0)")
q = []
for lab in HARD:
    Ta, Tb, _ = label_all(lab, 0, lab_idx, Y)
    Tu = Ta[~labeled]
    for j, l in enumerate(LABELS):
        q.append({"labeler": lab, "label": l, "added_1": int((Tu[:, j] == 1).sum()), "added_0": int((Tu[:, j] == 0).sum()),
                  "val_1": int((Tb[:, j] == 1).sum()), "val_1_right": int(Y_val[Tb[:, j] == 1, j].sum()),
                  "val_0": int((Tb[:, j] == 0).sum()), "val_0_wrong": int(Y_val[Tb[:, j] == 0, j].sum())})
print(pd.DataFrame(q).to_string(index=False), flush=True)
n_dup = (~np.isnan(label_all("dup", 0, lab_idx, Y)[0][~labeled])).any(axis=1).sum()
print("Unlabeled rows covered by exact-duplicate copying:", int(n_dup), flush=True)

best_lab_row = pick(d2[d2.labeler != "none"])
best_lab = best_lab_row["labeler"]
print("Selected labeler:", best_lab, flush=True)

# ---------- Stage 3: classifier features ----------
FEATS = ["emb", "emb_raw", "pca20", "pca50", "l2", "emb+year", "emb+morgan", "emb+pathways", "emb+diseases", "emb+criteria", "emb+all", "emb+gmm4", "emb+gmm8", "emb+gmm20"]
s3 = []
for lab in ["none", best_lab]:
    for f in FEATS:
        for clf_cfg in [best_lr, best_knn]:
            s3.append({**clf_cfg, "labeler": lab, "feat": f, "name": f"{lab} | {f} | {clf_cfg['clf']}"})
d3 = run(s3, "STAGE 3: features", BASE)

# ---------- Stage 4: ensembles ----------
best_feat = pick(d3)["feat"]
print("Selected features:", best_feat, flush=True)
s4 = []
ens = {"clf": "ens", "C": best_lr["C"], "cw": best_lr["cw"], "k": best_knn["k"], "metric": best_knn["metric"]}
for lab in sorted({"none", "gmm2", "dbscan", best_lab}):
    for f in sorted({"emb", best_feat}):
        s4.append({**ens, "labeler": lab, "feat": f, "name": f"LR+kNN ensemble | {lab} | {f}"})
d4 = run(s4, "STAGE 4: ensembles", BASE)

# ---------- Final ranking and bootstrap against the original baseline ----------
allr = pd.concat([d1, d2, d3, d4]).drop_duplicates("name")
allr.drop(columns=["cfg", "cv_list"]).to_csv(OUT + "experiment_plan_results.csv", index=False)
cols = ["name", "val_sample", "val_macro", "cv_sample", "cv_d_sample", "cv_d_sample_sd", "cv_macro", "cv_d_macro", "cv_d_macro_sd"]
print("\n===== TOP 12 BY VALIDATION SAMPLE F1 =====\n", allr.sort_values("val_sample", ascending=False)[cols].head(12).round(3).to_string(index=False))
print("\n===== TOP 12 BY CV SAMPLE F1 =====\n", allr.sort_values("cv_sample", ascending=False)[cols].head(12).round(3).to_string(index=False))
winner = pick(allr)
print("\nSELECTED BY RULE:", winner["name"], "\n", winner.drop(["cfg", "cv_list"]).to_string(), flush=True)

cfg = winner["cfg"]
Ftr, Fva = features(cfg["feat"], 0)
Ta, _, sc = label_all(cfg["labeler"], 0, lab_idx, Y)
T = None if Ta is None else Ta[~labeled]
Pw = predict(cfg, Ftr[labeled], Y, Ftr[~labeled], T, sc, Fva).astype(int)
Pb = predict(base_cfg, X_tr[labeled], Y, X_tr[~labeled], None, None, X_va).astype(int)
rng = np.random.default_rng(0)
diffs = []
for _ in range(2000):
    i = rng.integers(0, len(Y_val), len(Y_val))
    diffs.append(f1_score(Y_val[i], Pw[i], average="samples", zero_division=0) - f1_score(Y_val[i], Pb[i], average="samples", zero_division=0))
print("Rows where prediction differs from baseline:", int((Pw != Pb).any(axis=1).sum()), "of", len(Pb))
print("Bootstrap sample-F1 difference vs baseline on validation: mean", round(np.mean(diffs), 3),
      "95% interval", np.percentile(diffs, [2.5, 97.5]).round(3), flush=True)
