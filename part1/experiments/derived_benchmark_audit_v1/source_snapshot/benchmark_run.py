"""Benchmark Michael's AnchoredReasonClassifier (part1/anchored_solution.py) on the framework's
exact A/B/C folds (R=10, seed=1, same as the v2 leaderboard), paired against cand1 and cand5."""
import json, sys, time, warnings
import numpy as np
sys.path.insert(0, "../midterm/part1")
from joblib import Parallel, delayed
from p1lib.data import load, metrics, row_f1, LABELS
from p1lib.evaluate import make_folds, evaluate, paired_se
from anchored_solution import AnchoredReasonClassifier

D = load()
EMB = [f"emb_{i}" for i in range(150)]
X = D.df[EMB].to_numpy(float)
folds = make_folds(D, ("A", "B", "C"), R=10, seed=1)
unl = D.idx["unl"]

def run(f):
    warnings.simplefilter("ignore")
    fit_rows = np.sort(np.r_[f["train"], unl])            # permitted rows only, original CSV order
    Y = np.full((len(fit_rows), 5), -100)
    is_lab = np.isin(fit_rows, f["train"])
    Y[is_lab] = D.Y[fit_rows[is_lab]]
    t = time.time()
    try:
        m = AnchoredReasonClassifier().fit(X[fit_rows], Y)
        return f["name"], m.predict(X[f["eval"]]), time.time() - t, None
    except Exception as e:
        return f["name"], None, time.time() - t, repr(e)

outs = {n: (yp, s, err) for n, yp, s, err in Parallel(n_jobs=16)(delayed(run)(f) for f in folds)}
errs = {n: e for n, (_, _, e) in outs.items() if e}
print("fold errors:", errs if errs else "none")

res = {}
A = [f for f in folds if f["protocol"] == "A"]
rows = np.unique(np.concatenate([f["eval"] for f in A])); pos = {r: i for i, r in enumerate(rows)}
Yt = D.Y[rows]; preds = []
for r in sorted({f["repeat"] for f in A}):
    Yp = np.zeros((len(rows), 5), int)
    for f in A:
        if f["repeat"] == r: Yp[[pos[i] for i in f["eval"]]] = outs[f["name"]][0]
    preds.append(Yp)
rm = [metrics(Yt, p) for p in preds]
rf = np.mean([row_f1(Yt, p) for p in preds], axis=0)
res["A"] = {k: float(np.mean([m[k] for m in rm])) for k in rm[0] if k != "n"}
res["A"]["sample_f1_se"] = float(rf.std(ddof=1) / np.sqrt(len(rf))); res["rowf1_A"] = rf.tolist()
bm, rB = [], []
for f in [f for f in folds if f["protocol"] == "B"]:
    m = metrics(D.Y[f["eval"]], outs[f["name"]][0]); m["cut"] = f["cut"]; bm.append(m)
    rB.extend(row_f1(D.Y[f["eval"]], outs[f["name"]][0]).tolist())
res["B"] = {k: float(np.mean([m[k] for m in bm])) for k in bm[0] if k not in ("n", "cut")}
res["B_cuts"] = {m["cut"]: m["sample_f1"] for m in bm}; res["rowf1_B"] = rB; res["B_ncuts"] = len(bm)
fC = [f for f in folds if f["protocol"] == "C"][0]
res["C"] = metrics(D.Y[fC["eval"]], outs["C"][0])
res["J"] = 0.5 * res["A"]["sample_f1"] + 0.5 * res["B"]["sample_f1"]
res["J_se"] = 0.5 * float(np.hypot(res["A"]["sample_f1_se"], res["B"]["sample_f1_se"]))
res["fit_secs_mean"] = float(np.mean([s for _, s, _ in outs.values()]))

cands = {}
for c in ["cand1_selected_1se", "cand5_lr_cs_sld_robust"]:
    cfg = json.load(open(f"../midterm/part1/submissions/{c}.manifest.json"))["config"]
    cands[c] = evaluate(cfg, D, R=10, seed=1, n_jobs=16, bootstrap=False)

def line(name, r):
    return (f"{name:28s} A={r['A']['sample_f1']:.3f}±{r['A']['sample_f1_se']:.3f}  B={r['B']['sample_f1']:.3f}  "
            f"C={r['C']['sample_f1']:.3f}  J={r['J']:.3f}  macroA={r['A']['macro_f1']:.3f}  macroC={r['C']['macro_f1']:.3f}")
print(line("michael_anchored", res)); print("  B by cut:", {k: round(v, 3) for k, v in res["B_cuts"].items()})
print("  C per-label:", {l: round(res['C'][f'f1_{l}'], 3) for l in LABELS})
for c, r in cands.items():
    print(line(c, r)); print(f"  J diff vs michael: {r['J'] - res['J']:+.3f} (paired SE {paired_se(r, res):.3f})")
    print(f"  A diff: {r['A']['sample_f1'] - res['A']['sample_f1']:+.3f}  B diff: {r['B']['sample_f1'] - res['B']['sample_f1']:+.3f}  C diff: {r['C']['sample_f1'] - res['C']['sample_f1']:+.3f}")
json.dump({"michael_anchored": {k: v for k, v in res.items() if not k.startswith("rowf1")},
           **{c: {k: r[k] for k in ("A", "B", "C", "J", "J_se")} for c, r in cands.items()}},
          open("results.json", "w"), indent=2, default=float)
