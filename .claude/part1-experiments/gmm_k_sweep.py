import warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

warnings.filterwarnings("ignore")
LABELS = ["Safety", "Funding", "Efficacy", "Recruitment", "Other"]
train = pd.read_csv("part1_train.csv")
val = pd.read_csv("part1_val.csv")
emb = [f"emb_{i}" for i in range(150)]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(train[emb])
X_val = scaler.transform(val[emb])
pca = PCA(n_components=20, random_state=0)
X_pca = pca.fit_transform(X_scaled)
X_val_pca = pca.transform(X_val)

labeled = (train["Safety"] != -100).values
Y = train.loc[labeled, LABELS].values
Y_val = val[LABELS].values
base_rate = Y.mean(axis=0)
strength = 5


def fit_clf(y, keep):
    clf = LogisticRegression(C=1, max_iter=2000, class_weight="balanced")
    return clf.fit(X_scaled[keep], y[keep]).predict(X_val)


# True labels only, the score to beat
base_preds = np.column_stack([fit_clf(train[l].values, labeled) for l in LABELS])
base_sample = f1_score(Y_val, base_preds, average="samples", zero_division=0)
base_macro = f1_score(Y_val, base_preds, average="macro", zero_division=0)
base_per = f1_score(Y_val, base_preds, average=None, zero_division=0)
print(f"baseline sample {base_sample:.3f} macro {base_macro:.3f}", dict(zip(LABELS, base_per.round(3))))

rows, per_label = [], []
for k in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40]:
    for seed in range(5):
        gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=seed).fit(X_pca)
        probs = gmm.predict_proba(X_pca)
        R = probs[labeled]
        rates = (R.T @ Y + strength * base_rate) / (R.sum(axis=0, keepdims=True).T + strength)
        lp = probs @ rates
        vp = gmm.predict_proba(X_val_pca) @ rates

        row = {"k": k, "seed": seed, "bic": gmm.bic(X_pca), "aic": gmm.aic(X_pca),
               "min_labeled_in_cluster": R.sum(axis=0).min()}
        aucs = [roc_auc_score(Y_val[:, j], vp[:, j]) for j in range(5)]
        row["mean_auc"] = np.mean(aucs)

        # Fixed rule, so each k is compared without per-k tuning on validation
        for high in [0.5, 0.7]:
            preds = []
            for j, l in enumerate(LABELS):
                y = train[l].values.copy()
                y[~labeled & (lp[:, j] >= high)] = 1
                y[~labeled & (lp[:, j] <= base_rate[j] / 2)] = 0
                preds.append(fit_clf(y, y != -100))
                above = vp[:, j] >= high
                per_label.append({"k": k, "seed": seed, "high": high, "label": l, "auc": aucs[j],
                                  "added_pos": int((~labeled & (lp[:, j] >= high)).sum()),
                                  "val_above": int(above.sum()),
                                  "val_correct": int(Y_val[above, j].sum()),
                                  "f1": f1_score(Y_val[:, j], preds[-1], zero_division=0)})
            preds = np.column_stack(preds)
            row[f"sample_{high}"] = f1_score(Y_val, preds, average="samples", zero_division=0)
            row[f"macro_{high}"] = f1_score(Y_val, preds, average="macro", zero_division=0)
        rows.append(row)

df = pd.DataFrame(rows)
pl = pd.DataFrame(per_label)
pd.set_option("display.width", 250, "display.max_columns", None, "display.max_rows", None)
summary = df.groupby("k").agg(
    bic=("bic", "mean"), aic=("aic", "mean"), min_labeled=("min_labeled_in_cluster", "mean"),
    mean_auc=("mean_auc", "mean"), auc_sd=("mean_auc", "std"),
    sample_05=("sample_0.5", "mean"), sample_05_sd=("sample_0.5", "std"),
    macro_05=("macro_0.5", "mean"), macro_05_sd=("macro_0.5", "std"),
    sample_07=("sample_0.7", "mean"), sample_07_sd=("sample_0.7", "std"),
    macro_07=("macro_0.7", "mean"), macro_07_sd=("macro_0.7", "std"),
).round(3)
print(summary.to_string())

print("\nPer-label mean F1 by k, high=0.5")
print(pl[pl.high == 0.5].pivot_table(index="k", columns="label", values="f1")[LABELS].round(3).to_string())
print("\nPer-label mean F1 by k, high=0.7")
print(pl[pl.high == 0.7].pivot_table(index="k", columns="label", values="f1")[LABELS].round(3).to_string())
print("\nPer-label mean AUC by k")
print(pl[pl.high == 0.5].pivot_table(index="k", columns="label", values="auc")[LABELS].round(3).to_string())

for high in [0.5, 0.7]:
    g = pl[pl.high == high].groupby(["k", "label"])[["added_pos", "val_above", "val_correct"]].sum()
    g["precision"] = (g.val_correct / g.val_above).round(3)
    g["added_pos"] = (g.added_pos / 5).round(0)
    print(f"\nPooled validation precision of generated 1s across 5 seeds, high={high}")
    print(g["precision"].unstack("label")[LABELS].to_string())
    print("Mean generated positives")
    print(g["added_pos"].unstack("label")[LABELS].to_string())
