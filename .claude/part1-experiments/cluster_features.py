import warnings
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, f1_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
LABELS = ["Safety", "Funding", "Efficacy", "Recruitment", "Other"]
train = pd.read_csv("part1_train.csv")
val = pd.read_csv("part1_val.csv")
emb = [f"emb_{i}" for i in range(150)]
labeled = (train["Safety"] != -100).values
Y = train.loc[labeled, LABELS].values
Y_val = val[LABELS].values
strength = 5
pd.set_option("display.width", 250, "display.max_columns", None, "display.max_rows", None)

print("Rows with a matched drug: train", round(train["drug_smiles"].notna().mean(), 3),
      "val", round(val["drug_smiles"].notna().mean(), 3))
print("Non-Recruitment rate among labeled rows, with and without a matched drug:",
      round(1 - Y[train.loc[labeled, "drug_smiles"].notna().values, 3].mean(), 3),
      round(1 - Y[train.loc[labeled, "drug_smiles"].isna().values, 3].mean(), 3))
print("Labeled start_year: mean", round(train.loc[labeled, "start_year"].mean(), 1),
      "| unlabeled mean", round(train.loc[~labeled, "start_year"].mean(), 1),
      "| val mean", round(val["start_year"].mean(), 1))

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def fingerprint(field):
    if pd.isna(field):
        return np.zeros(1024)
    fps = [gen.GetFingerprintAsNumPy(m) for s in str(field).split(";")
           if (m := Chem.MolFromSmiles(s)) is not None]
    return np.max(fps, axis=0) if fps else np.zeros(1024)


def block_pca(A, B, n=20):
    """Standardize, reduce to n dims, then rescale so every block has equal total variance."""
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
    a, b = s.transform(a), s.transform(b)
    return a, b


blocks = {}
blocks["emb"] = block_pca(train[emb].values, val[emb].values)
y_s = StandardScaler().fit(train[["start_year"]])
blocks["year"] = (y_s.transform(train[["start_year"]]), y_s.transform(val[["start_year"]]))
blocks["morgan"] = block_pca(np.vstack(train["drug_smiles"].apply(fingerprint)),
                             np.vstack(val["drug_smiles"].apply(fingerprint)))
semi = lambda s: [t.strip() for t in str(s).split(";") if t.strip()]
path_vec = CountVectorizer(tokenizer=semi, token_pattern=None, binary=True, min_df=5)
blocks["pathways"] = block_text(train["drug_pathways"].fillna(""), val["drug_pathways"].fillna(""), path_vec)
blocks["diseases"] = block_text(train["diseases_cleaned"].fillna(""), val["diseases_cleaned"].fillna(""),
                                TfidfVectorizer(min_df=3))
blocks["criteria"] = block_text(train["criteria"].fillna(""), val["criteria"].fillna(""),
                                TfidfVectorizer(min_df=5, max_features=20000, stop_words="english"))

SETS = {
    "emb only": ["emb"],
    "emb + year": ["emb", "year"],
    "emb + morgan": ["emb", "morgan"],
    "emb + pathways": ["emb", "pathways"],
    "emb + diseases": ["emb", "diseases"],
    "emb + criteria": ["emb", "criteria"],
    "emb + all": ["emb", "year", "morgan", "pathways", "diseases", "criteria"],
    "year only": ["year"],
    "morgan only": ["morgan"],
    "pathways only": ["pathways"],
    "diseases only": ["diseases"],
    "criteria only": ["criteria"],
}

scaler = StandardScaler().fit(train[emb])
X_tr, X_va = scaler.transform(train[emb]), scaler.transform(val[emb])


def clf(X, y, X_test):
    return LogisticRegression(C=1, max_iter=2000, class_weight="balanced").fit(X, y).predict(X_test)


rows, reference = [], {}
for name, parts in SETS.items():
    A = np.hstack([blocks[p][0] for p in parts])
    B = np.hstack([blocks[p][1] for p in parts])
    for k in [2, 4, 8]:
        for seed in range(5):
            gmm = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-4, random_state=seed).fit(A)
            hard = gmm.predict(A)
            if name == "emb only":
                reference[(k, seed)] = hard
            probs = gmm.predict_proba(A)
            R = probs[labeled]
            br = Y.mean(axis=0)
            rates = (R.T @ Y + strength * br) / (R.sum(axis=0, keepdims=True).T + strength)
            lp, vp = probs @ rates, gmm.predict_proba(B) @ rates
            aucs = [roc_auc_score(Y_val[:, j], vp[:, j]) for j in range(5)]
            preds = []
            for j, l in enumerate(LABELS):
                y = train[l].values.copy()
                y[~labeled & (lp[:, j] >= .7)] = 1
                y[~labeled & (lp[:, j] <= br[j] / 2)] = 0
                keep = y != -100
                preds.append(clf(X_tr[keep], y[keep], X_va))
            P = np.column_stack(preds)
            rows.append({"features": name, "k": k, "seed": seed, "mean_auc": np.mean(aucs),
                         **{f"auc_{l}": a for l, a in zip(LABELS, aucs)},
                         "ari_vs_emb": adjusted_rand_score(reference[(k, seed)], hard),
                         "best_rare_purity": rates[:, [0, 1, 2, 4]].max(),
                         "sample": f1_score(Y_val, P, average="samples", zero_division=0),
                         "macro": f1_score(Y_val, P, average="macro", zero_division=0)})
df = pd.DataFrame(rows)
order = list(SETS)
s = df.drop(columns="seed").groupby(["k", "features"]).mean().round(3)
sd = df.groupby(["k", "features"])[["mean_auc", "sample", "macro"]].std().round(3).add_suffix("_sd")
print(s.join(sd).reset_index().assign(features=lambda d: pd.Categorical(d.features, order, ordered=True))
      .sort_values(["k", "features"]).to_string(index=False))
