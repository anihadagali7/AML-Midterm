"""Blend Michael's AnchoredReasonClassifier with cand1 / cand5 on the framework's exact folds (R=10, seed=1).
Blend = weighted average of per-label scores (ours: combo posterior after head -> label marginals Q@CM;
Michael: his 5 selection scores). Decode with Michael's rule: argmax label + any label >= t."""
import json, sys, warnings
import numpy as np
sys.path.insert(0, "../midterm/part1")
from joblib import Parallel, delayed
from p1lib.data import load, metrics, row_f1, LABELS
from p1lib.evaluate import make_folds, fold_outputs, paired_se, evaluate
from p1lib.head import apply_head
from anchored_solution import AnchoredReasonClassifier
warnings.simplefilter("ignore")

D = load(); X = D.df[[f"emb_{i}" for i in range(150)]].to_numpy(float)
folds = make_folds(D, ("A", "B", "C"), R=10, seed=1); unl = D.idx["unl"]

def mich(f):
    warnings.simplefilter("ignore")
    rows = np.sort(np.r_[f["train"], unl]); Y = np.full((len(rows), 5), -100)
    lab = np.isin(rows, f["train"]); Y[lab] = D.Y[rows[lab]]
    return AnchoredReasonClassifier().fit(X[rows], Y).predict_proba(X[f["eval"]])
S = {"michael": Parallel(n_jobs=16)(delayed(mich)(f) for f in folds)}
native = {}
for c in ["cand1_selected_1se", "cand5_lr_cs_sld_robust"]:
    cfg = json.load(open(f"../midterm/part1/submissions/{c}.manifest.json"))["config"]
    outs = fold_outputs(cfg, D, folds, n_jobs=16)
    S[c] = []
    for f, o in zip(folds, outs):
        P = o["P_eval"].astype(float); P /= P.sum(1, keepdims=True)
        Pt = o.get("P_target"); Pt = None if Pt is None else Pt.astype(float) / Pt.astype(float).sum(1, keepdims=True)
        _, Q, _ = apply_head(P, o["prior"], D.CM, cfg.get("head", {}), Pt)
        S[c].append(Q @ D.CM)
    native[c] = evaluate(cfg, D, R=10, seed=1, n_jobs=16, bootstrap=False)

def decode(M, t):
    Y = (M >= t).astype(int); Y[np.arange(len(M)), M.argmax(1)] = 1; return Y

def score(preds):
    res = {}; A = [i for i, f in enumerate(folds) if f["protocol"] == "A"]
    rows = np.unique(np.concatenate([folds[i]["eval"] for i in A])); pos = {r: k for k, r in enumerate(rows)}; Yt = D.Y[rows]
    reps = []
    for r in sorted({folds[i]["repeat"] for i in A}):
        Yp = np.zeros((len(rows), 5), int)
        for i in A:
            if folds[i]["repeat"] == r: Yp[[pos[j] for j in folds[i]["eval"]]] = preds[i]
        reps.append(Yp)
    rf = np.mean([row_f1(Yt, p) for p in reps], axis=0)
    res["A"] = {"sample_f1": float(np.mean([metrics(Yt, p)["sample_f1"] for p in reps])), "sample_f1_se": float(rf.std(ddof=1) / np.sqrt(len(rf))),
                "macro_f1": float(np.mean([metrics(Yt, p)["macro_f1"] for p in reps]))}
    res["rowf1_A"] = rf.tolist()
    B = [i for i, f in enumerate(folds) if f["protocol"] == "B"]
    res["B"] = {"sample_f1": float(np.mean([metrics(D.Y[folds[i]["eval"]], preds[i])["sample_f1"] for i in B]))}
    res["rowf1_B"] = sum([row_f1(D.Y[folds[i]["eval"]], preds[i]).tolist() for i in B], []); res["B_ncuts"] = len(B)
    iC = [i for i, f in enumerate(folds) if f["protocol"] == "C"][0]; res["C"] = metrics(D.Y[folds[iC]["eval"]], preds[iC])
    res["J"] = 0.5 * res["A"]["sample_f1"] + 0.5 * res["B"]["sample_f1"]
    return res

def blend(ws, t=0.3):
    return [decode(sum(w * S[k][i] for k, w in ws.items()) / sum(ws.values()), t) for i in range(len(folds))]

R = {}
R["michael (native)"] = score(blend({"michael": 1}))
for c in native: R[f"{c} (native)"] = native[c]
R["cand1 marginals, thr0.3"] = score(blend({"cand1_selected_1se": 1}))
R["cand5 marginals, thr0.3"] = score(blend({"cand5_lr_cs_sld_robust": 1}))
R["== M+cand1 equal"] = score(blend({"michael": 1, "cand1_selected_1se": 1}))
R["== M+cand5 equal"] = score(blend({"michael": 1, "cand5_lr_cs_sld_robust": 1}))
R["== M+cand1+cand5 equal"] = score(blend({"michael": 1, "cand1_selected_1se": 1, "cand5_lr_cs_sld_robust": 1}))
for w in [0.25, 0.75]:
    R[f"M+cand1 w_ours={w}"] = score(blend({"michael": 1 - w, "cand1_selected_1se": w}))
    R[f"M+cand5 w_ours={w}"] = score(blend({"michael": 1 - w, "cand5_lr_cs_sld_robust": w}))
for t in [0.4, 0.5]:
    R[f"M+cand1 equal, thr{t}"] = score(blend({"michael": 1, "cand1_selected_1se": 1}, t))
    R[f"M+cand1+cand5 equal, thr{t}"] = score(blend({"michael": 1, "cand1_selected_1se": 1, "cand5_lr_cs_sld_robust": 1}, t))

ref_m, ref_c1 = R["michael (native)"], R["cand1_selected_1se (native)"]
print(f"{'config':34s} {'A':>6s} {'B':>6s} {'C':>6s} {'J':>6s} {'macroC':>7s}  dJ_vs_M(SE)   dJ_vs_cand1(SE)")
for k, r in R.items():
    print(f"{k:34s} {r['A']['sample_f1']:.3f}  {r['B']['sample_f1']:.3f}  {r['C']['sample_f1']:.3f}  {r['J']:.3f}  {r['C']['macro_f1']:.3f}"
          f"   {r['J']-ref_m['J']:+.3f}({paired_se(r, ref_m):.3f})   {r['J']-ref_c1['J']:+.3f}({paired_se(r, ref_c1):.3f})")
json.dump({k: {p: r[p] for p in ("A", "B", "C", "J")} for k, r in R.items()}, open("ensemble_results.json", "w"), indent=2, default=float)
