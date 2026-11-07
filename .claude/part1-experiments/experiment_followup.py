"""Follow-up to experiment_plan.py: labeler x C interaction, real ensembles, fair preprocessing checks."""
import os
SCRATCH = os.path.dirname(os.path.abspath(__file__)) + "/"
src = open(SCRATCH + "experiment_plan.py").read().split("# ---------- Stage 1: classifier tuning")[0]
exec(src)

base = {"name": "BASE none | LR C=1 balanced", "labeler": "none", "feat": "emb", "clf": "lr", "C": 1, "cw": True}
BASE = evaluate_config(base)["cv_list"]


def lr(lab, c, cw=True, feat="emb"):
    return {"name": f"{lab} | {feat} | LR C={c} {'bal' if cw else 'unw'}", "labeler": lab, "feat": feat, "clf": "lr", "C": c, "cw": cw}


# A. Hard labelers crossed with C
a = [lr(l, c) for l in ["none", "gmm2", "km4", "km10", "dbscan", "knn10", "dup", "consensus2", "consensus3"] for c in [.01, .1, 1, 10]]
dA = run(a, "A: labelers x LR C", BASE)

# B. Soft labels: smaller weights and both class-weight settings
b = [lr(f"soft_{k}_{s}", c, cw) for k in [4, 8, 20] for s in [.02, .1] for c in [.1, 1] for cw in [True, False]]
dB = run(b, "B: soft GMM labels", BASE)

# C. Real ensembles: kNN with k>1 so its probabilities aren't only 0 or 1
c_cfg = []
for lab in ["none", "knn10", "gmm2", "dbscan"]:
    for c in [.01, 1]:
        for k in [5, 10]:
            c_cfg.append({"name": f"ENS {lab} | LR C={c} + kNN k={k} cosine", "labeler": lab, "feat": "emb", "clf": "ens", "C": c, "cw": True, "k": k, "metric": "cosine"})
dC = run(c_cfg, "C: LR + kNN ensembles", BASE)

# D. Preprocessing at C values suited to its scale
d = [lr("none", c, True, "l2") for c in [1, 10, 100]] + [lr("none", c, True, "emb_raw") for c in [100, 1000, 10000]]
dD = run(d, "D: L2-normalized and raw embeddings", BASE)

# E. Added features at C=1, since stage 3 only used C=0.01
e = [lr(l, 1, True, f) for l in ["none", "knn10"] for f in ["emb+year", "emb+morgan", "emb+pathways", "emb+diseases", "emb+criteria", "emb+all"]]
dE = run(e, "E: added features at C=1", BASE)

allr = pd.concat([dA, dB, dC, dD, dE]).drop_duplicates("name")
allr.drop(columns=["cfg", "cv_list"]).to_csv(SCRATCH + "experiment_followup_results.csv", index=False)
cols = ["name", "val_sample", "val_macro", "cv_sample", "cv_d_sample", "cv_d_sample_sd", "cv_macro", "cv_d_macro", "cv_d_macro_sd"]
print("\n===== TOP 15 BY VALIDATION SAMPLE F1 =====\n", allr.sort_values("val_sample", ascending=False)[cols].head(15).round(3).to_string(index=False))
print("\n===== TOP 15 BY CV SAMPLE F1 =====\n", allr.sort_values("cv_sample", ascending=False)[cols].head(15).round(3).to_string(index=False))
print("\n===== TOP 15 BY CV MACRO F1 =====\n", allr.sort_values("cv_macro", ascending=False)[cols].head(15).round(3).to_string(index=False))


def rows_and_boot(cfg, label):
    Ftr, Fva = features(cfg["feat"], 0)
    Ta, _, sc = label_all(cfg["labeler"], 0, lab_idx, Y)
    T = None if Ta is None else Ta[~labeled]
    Pw = predict(cfg, Ftr[labeled], Y, Ftr[~labeled], T, sc, Fva).astype(int)
    Pb = predict(base, X_tr[labeled], Y, X_tr[~labeled], None, None, X_va).astype(int)
    rng = np.random.default_rng(0)
    ds, dm = [], []
    for _ in range(2000):
        i = rng.integers(0, len(Y_val), len(Y_val))
        ds.append(f1_score(Y_val[i], Pw[i], average="samples", zero_division=0) - f1_score(Y_val[i], Pb[i], average="samples", zero_division=0))
        dm.append(f1_score(Y_val[i], Pw[i], average="macro", zero_division=0) - f1_score(Y_val[i], Pb[i], average="macro", zero_division=0))
    print(label, "| rows differing:", int((Pw != Pb).any(axis=1).sum()), "of", len(Pb),
          "| sample diff", np.round(np.mean(ds), 3), np.percentile(ds, [2.5, 97.5]).round(3),
          "| macro diff", np.round(np.mean(dm), 3), np.percentile(dm, [2.5, 97.5]).round(3), flush=True)


print("\n===== BOOTSTRAP vs ORIGINAL BASELINE (validation, seed 0) =====")
for cfg_ in [lr("none", .01), lr("knn10", .01), lr("knn10", 1), lr("gmm2", 1), lr("dbscan", 1), lr("none", .01, True, "emb+morgan"), lr("knn10", .01, True, "emb+criteria")]:
    rows_and_boot(cfg_, cfg_["name"])
