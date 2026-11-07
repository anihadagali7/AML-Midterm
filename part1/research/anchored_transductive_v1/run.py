"""Register, run and independently verify the fixed transductive comparison."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

STUDY = Path(__file__).resolve().parent
PART1 = STUDY.parents[1]
ROOT = PART1.parent
sys.path.insert(0, str(PART1))
from anchored_solution import AnchoredReasonClassifier, select_reasons
from models import GeometryOnlyAnchored, fit_predict

REFERENCE = PART1 / "experiments/current_leaders_grouped_v1"
LABELS = ["Safety", "Funding", "Efficacy", "Recruitment", "Other"]
FEATURES = [f"emb_{i}" for i in range(150)]
MODES = ["inductive", "geometry_only", "target_self_training"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def groups(frame):
    return pd.util.hash_pandas_object(frame[FEATURES].round(10), index=False).to_numpy()


def metrics(y, prediction):
    precision, recall, f1, support = precision_recall_fscore_support(y, prediction, zero_division=0)
    return dict(sample_f1=float(f1_score(y, prediction, average="samples", zero_division=0)),
                macro_f1=float(f1.mean()), **{
                    f"{kind}_{label}": float(values[j])
                    for kind, values in [("precision", precision), ("recall", recall),
                                         ("f1", f1), ("support", support)]
                    for j, label in enumerate(LABELS)})


def sources():
    paths = [STUDY/"models.py", Path(__file__), PART1/"anchored_solution.py",
             PART1/"data/part1_train.csv", PART1/"data/part1_val.csv",
             PART1/"data/part1_test.csv", PART1/"TESTING_POLICY.md", ROOT/"Midterm_Project.pdf",
             REFERENCE/"anchored_train.npz", REFERENCE/"anchored_pooled.npz",
             REFERENCE/"diagnostic_predictions.csv",
             PART1/"submissions/archive/submitted/anchored_reason_trainval235__kaggle_0.89636.csv",
             PART1/"experiments/paired_improvement/candidate/manifest.json",
             PART1/"research/lda_ensemble/research/ani_snapshots/a548721/part1_best_model.ipynb"]
    paths += [REFERENCE/"snapshots"/(name+".json")
              for name in ["train3", "train5", "pooled3", "pooled5", "chronology"]]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def register():
    path = STUDY/"protocol.json"
    assert not path.exists(), "Never overwrite a registered study."
    save(path, dict(
        version="anchored-transductive-v1", registered_at=datetime.now(timezone.utc).isoformat(),
        parent="Unchanged anchored classifier; user-reported Kaggle 0.89636; inductive final235 fit.",
        hypothesis="Target embeddings may improve geometry; confident target-generated labels may add support or reinforce errors.",
        modes={
            "inductive": "Frozen parent fit on eligible gold and originally masked training inputs only.",
            "geometry_only": "Append target embeddings only when fitting StandardScaler, PCA30 and diagonal GMM30; target rows cannot receive generated training labels.",
            "target_self_training": "Append target embeddings with five -100 labels to the exact frozen parent fit; geometry and confidence-filtered target-generated labels may influence fitting."
        },
        settings="All parent features, random seeds, C, priors, weights, thresholds, two generated-label passes and decoder fixed; no tuning.",
        schedule="Predeclared full 8 repeats x 5 folds on both saved grouped pools plus validation78 and three chronological windows for EVERY mode: 252 scored fits.",
        supplementary_exception="Run all four supplementary fits even if CV fails, to answer this focused access question without post-result selection.",
        target_access="Only each evaluated batch's embeddings enter transductive fits. Target truth never enters fit_predict. Actual Kaggle test inputs are hashed for protection but not loaded or used in evaluation.",
        group_policy="Same gold/masked fitting IDs, original order and group purges as the frozen parent. Only the explicit target batch is added. Train157 excludes every validation-source row.",
        chronology="Same earlier fitting IDs and later query IDs; transductive variants deliberately see later query embeddings. These are transductive chronological diagnostics, not strict future-input-excluded forecasts.",
        original_ani_difference="Ani's final fit used test inputs for geometry and generated labels. Its old CV used globally fitted geometry and actual Kaggle inputs; this study instead refits each fold and uses its held-out batch as the target analogue.",
        advancement="Positive paired mean sample-F1 delta and at least 5/8 repeat wins in BOTH pools. Supplementary regressions remain visible; no automatic replacement, final test fit, upload or merge.",
        metrics="Complete OOF sample-F1 per repeat, mean, population SD, paired repeat deltas/wins/ties/losses, macro-F1 and every label precision/recall/F1/support; chronology pooled over 98 rows.",
        controls="Refit parent on all 84 cases; require every CV score array within 1e-7 and all CV/supplementary predicted labels identical to frozen records. Empty-target geometry implementation must match parent on the first fold.",
        limitations=["Reused development splits are not an untouched test.",
                     "CV/chronology target batches have fewer rows than the 312-row final test; a local effect may not transfer.",
                     "The PDF describes supplied unlabeled data, unsupervised augmentation and no test-label reverse engineering; it does not explicitly settle transductive submission permission."],
        retention="Keep code, protocol, aggregate/per-repeat/per-label results and verification. Full predictions only for a qualifying candidate or useful matched control; no fitted models or per-fold datasets.",
        versions={"python": platform.python_version(), "numpy": np.__version__,
                  "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        sources=sources()))
    print("Registered", path)


def cases():
    train = pd.read_csv(PART1/"data/part1_train.csv")
    val = pd.read_csv(PART1/"data/part1_val.csv")
    full = pd.concat([train, val], ignore_index=True)
    masked = train[train[LABELS].eq(-100).all(axis=1)].reset_index(drop=True)
    result = []
    for pool, frame in [("train", train), ("pooled", full)]:
        gold = frame[frame[LABELS].ge(0).all(axis=1)].reset_index(drop=True)
        plans = [json.loads((REFERENCE/"snapshots"/f"{pool}{n}.json").read_text()) for n in [3, 5]]
        assert plans[0]["gold_ids"] == plans[1]["gold_ids"] == gold.ID.tolist()
        seeds = plans[0]["seeds"] + plans[1]["seeds"]
        order = {v: i for i, v in enumerate(frame.ID)}
        gg, ug = groups(gold), groups(masked)
        for f in plans[0]["folds"] + plans[1]["folds"]:
            a, b = np.array(f["train_gold_positions"]), np.array(f["heldout_gold_positions"])
            assert set(a).isdisjoint(b) and len(a)+len(b) == len(gold)
            assert set(gg[a]).isdisjoint(gg[b]) and gold.iloc[b].ID.tolist() == f["heldout_ids"]
            keep = ~np.isin(ug, gg[b])
            assert int((~keep).sum()) == f["masked_duplicates_excluded"]
            fit = pd.concat([gold.iloc[a], masked[keep]], ignore_index=True)
            fit = fit.iloc[np.argsort([order[v] for v in fit.ID])].reset_index(drop=True)
            query = gold.iloc[b].reset_index(drop=True)
            assert set(groups(fit)).isdisjoint(groups(query)) and set(fit.ID).isdisjoint(query.ID)
            if pool == "train":
                assert set(fit.ID).isdisjoint(val.ID) and set(query.ID).isdisjoint(val.ID)
            result.append(dict(key=f"{pool}_{f['seed']}_{f['fold']}", pool=pool, seed=f["seed"],
                               fold=f["fold"], repeat_index=seeds.index(f["seed"]), positions=b.tolist(),
                               purged_masked=int((~keep).sum()), fit=fit, query=query))
    result.append(dict(key="validation", pool="supplementary", seed=-1, fold=-1,
                       repeat_index=-1, positions=[], purged_masked=0, fit=train, query=val))
    chrono = json.loads((REFERENCE/"snapshots/chronology.json").read_text())
    indexed = full.set_index("ID", drop=False)
    for w in chrono["chronological_splits"]:
        fit, query = indexed.loc[w["training_ids"]], indexed.loc[w["heldout_ids"]]
        assert fit[LABELS].ge(0).all(axis=1).sum() == w["gold_rows"]
        assert fit[LABELS].eq(-100).all(axis=1).sum() == w["masked_rows"]
        assert len(query) == w["heldout_rows"] and set(groups(fit)).isdisjoint(groups(query))
        assert set(fit.ID).isdisjoint(query.ID)
        result.append(dict(key=w["name"], pool="supplementary", seed=-1, fold=-1,
                           repeat_index=-1, positions=[], purged_masked=w["purged_earlier_masked"],
                           fit=fit.reset_index(drop=True), query=query.reset_index(drop=True)))
    assert len(result) == 84
    return result


def evaluate_case(case):
    start = time.monotonic()
    fit, query = case["fit"], case["query"]
    x, y = fit[FEATURES].to_numpy(float), fit[LABELS].to_numpy(int)
    qx = query[FEATURES].to_numpy(float)  # Label-free interface for every mode.
    scores, audits = [], []
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for mode in MODES:
            score, audit = fit_predict(x, y, qx, mode)
            assert np.isfinite(score).all() and score.shape == (len(qx), 5)
            scores.append(score)
            audits.append(audit)
        equivalence = None
        if case["key"] == "train_20261126_0":
            empty = GeometryOnlyAnchored().fit(x, y, np.empty((0, 150))).predict_proba(qx)
            np.testing.assert_allclose(empty, scores[0], atol=1e-7, rtol=0)
            np.testing.assert_array_equal(select_reasons(empty), select_reasons(scores[0]))
            equivalence = float(np.max(np.abs(empty-scores[0])))
    data = dict(ids=query.ID.to_numpy(str), truth=query[LABELS].to_numpy(int), scores=np.stack(scores))
    data["predictions"] = np.stack([select_reasons(s) for s in scores])
    audit = {k: v for k, v in case.items() if k not in ["fit", "query"]}
    audit.update(fit_ids_sha256=hashlib.sha256("\n".join(fit.ID).encode()).hexdigest(),
                 query_ids_sha256=hashlib.sha256("\n".join(query.ID).encode()).hexdigest(),
                 modes=audits, seconds=time.monotonic()-start, empty_target_max_error=equivalence)
    return case["key"], data, audit


def run(output, workers):
    protocol = json.loads((STUDY/"protocol.json").read_text())
    assert protocol["sources"] == sources(), "Registered sources changed. Version the study instead."
    output = Path(output).resolve()
    assert not output.exists(), "Choose a new output directory."
    output.mkdir(parents=True)
    (output/"predictions").mkdir()
    job_cases = cases()
    audits = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(evaluate_case, c) for c in job_cases]
        for future in as_completed(futures):
            key, data, audit = future.result()
            np.savez_compressed(output/"predictions"/(key+".npz"), **data)
            audits.append(audit)
            print(json.dumps(dict(completed=len(audits), total=84, case=key,
                                  seconds=round(audit["seconds"], 2))), flush=True)
    save(output/"fit_audits.json", sorted(audits, key=lambda a: a["key"]))
    assert sources() == protocol["sources"]
    verify(output)


def verify(output):
    output = Path(output).resolve()
    protocol = json.loads((STUDY/"protocol.json").read_text())
    assert sources() == protocol["sources"]
    job_cases = cases()
    refs = {}
    for pool in ["train", "pooled"]:
        with np.load(REFERENCE/f"anchored_{pool}.npz", allow_pickle=True) as z:
            refs[pool] = {k: z[k] for k in z.files}
    diagnostic = pd.read_csv(REFERENCE/"diagnostic_predictions.csv")
    diagnostic = diagnostic[diagnostic.name == "anchored_candidate"]
    entries, score_rows, check_errors = [], [], []
    for c in job_cases:
        with np.load(output/"predictions"/(c["key"]+".npz"), allow_pickle=False) as z:
            np.testing.assert_array_equal(z["ids"], c["query"].ID)
            np.testing.assert_array_equal(z["truth"], c["query"][LABELS])
            assert z["scores"].shape == (3, len(c["query"]), 5)
            for i in range(3):
                np.testing.assert_array_equal(select_reasons(z["scores"][i]), z["predictions"][i])
            if c["pool"] != "supplementary":
                ref = refs[c["pool"]]
                b = c["positions"]
                np.testing.assert_array_equal(z["ids"], ref["ids"][b])
                np.testing.assert_array_equal(z["truth"], ref["truth"][b])
                old = ref["scores"][c["repeat_index"], b]
                np.testing.assert_allclose(z["scores"][0], old, atol=1e-7, rtol=0)
                np.testing.assert_array_equal(z["predictions"][0], ref["predictions"][c["repeat_index"], b])
                check_errors.append(float(np.max(np.abs(z["scores"][0]-old))))
            else:
                old = diagnostic[diagnostic["case"] == c["key"]].set_index("ID").loc[z["ids"]]
                np.testing.assert_array_equal(z["truth"], old[["actual_"+s for s in LABELS]])
                np.testing.assert_array_equal(z["predictions"][0], old[["predicted_"+s for s in LABELS]])
            for i, mode in enumerate(MODES):
                entry = pd.DataFrame(dict(ID=z["ids"], pool=c["pool"], seed=c["seed"],
                                          case=c["key"], mode=mode))
                for j, label in enumerate(LABELS):
                    entry["actual_"+label] = z["truth"][:, j]
                    entry["predicted_"+label] = z["predictions"][i, :, j]
                entries.append(entry)
                if c["pool"] == "supplementary":
                    score_rows.append(dict(case=c["key"], mode=mode, **metrics(z["truth"], z["predictions"][i])))
    frame = pd.concat(entries, ignore_index=True)
    truth_cols, pred_cols = [[prefix+s for s in LABELS] for prefix in ["actual_", "predicted_"]]
    repeats = []
    for (pool, seed, mode), g in frame[frame.pool != "supplementary"].groupby(["pool", "seed", "mode"]):
        assert g.ID.is_unique and len(g) == (157 if pool == "train" else 235)
        assert set(g.ID) == set(refs[pool]["ids"])
        repeats.append(dict(pool=pool, seed=int(seed), mode=mode, **metrics(g[truth_cols], g[pred_cols])))
    for mode, g in frame[(frame.pool == "supplementary") & (frame["case"] != "validation")].groupby("mode"):
        assert len(g) == 98 and g.ID.is_unique
        score_rows.append(dict(case="chronology98", mode=mode, **metrics(g[truth_cols], g[pred_cols])))
    repeats = pd.DataFrame(repeats)
    assert len(repeats) == 48
    repeats.to_csv(output/"repeat_scores.csv", index=False)
    summary = repeats.groupby(["pool", "mode"]).agg(
        mean_sample_f1=("sample_f1", "mean"), population_sd=("sample_f1", lambda s: s.std(ddof=0)),
        mean_macro_f1=("macro_f1", "mean")).reset_index()
    summary.to_csv(output/"cv_summary.csv", index=False)
    pairs, deltas, qualified = [], [], {m: True for m in MODES[1:]}
    for pool in ["train", "pooled"]:
        pivot = repeats[repeats.pool == pool].pivot(index="seed", columns="mode", values="sample_f1")
        for mode in MODES[1:]:
            delta = pivot[mode] - pivot.inductive
            wins, ties, losses = int((delta > 1e-12).sum()), int((delta.abs() <= 1e-12).sum()), int((delta < -1e-12).sum())
            passed = bool(delta.mean() > 0 and wins >= 5)
            qualified[mode] &= passed
            pairs.append(dict(pool=pool, mode=mode, mean_delta=float(delta.mean()), wins=wins,
                              ties=ties, losses=losses, meets_pool_rule=passed))
            deltas += [dict(pool=pool, mode=mode, seed=int(seed), sample_f1_delta=float(d)) for seed, d in delta.items()]
    pd.DataFrame(pairs).to_csv(output/"paired_summary.csv", index=False)
    pd.DataFrame(deltas).to_csv(output/"paired_repeat_deltas.csv", index=False)
    supplementary = pd.DataFrame(score_rows)
    supplementary.to_csv(output/"supplementary_scores.csv", index=False)
    audits = json.loads((output/"fit_audits.json").read_text())
    assert len(audits) == 84 and {a["key"] for a in audits} == {c["key"] for c in job_cases}
    for a in audits:
        assert len(a["modes"]) == 3
        for m in a["modes"]:
            assert m["target_labels_supplied"] is False
            assert m["gold_rows"] == m["rounds"][0]["known"]
            if m["mode"] != "target_self_training":
                assert m["target_generated_rows_in_final_fit"] == 0
    check = dict(passed=True, full_cv_fits=240, supplementary_fits=12,
                 implementation_equivalence_fits=1, frozen_parent_cv_checks=80,
                 frozen_parent_supplementary_checks=4, parent_max_score_error=max(check_errors),
                 empty_target_max_error=next(a["empty_target_max_error"] for a in audits if a["empty_target_max_error"] is not None),
                 repeats_scored=48, qualified_cv=qualified, protected_hashes_unchanged=True,
                 query_truth_passed_to_model=False, actual_kaggle_test_inputs_used=False,
                 new_test_predictions=False, protocol_sha256=sha(STUDY/"protocol.json"))
    save(output/"verification.json", check)
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(pairs).to_string(index=False), flush=True)
    print(supplementary[supplementary["case"].isin(["validation", "chronology98"])][["case", "mode", "sample_f1", "macro_f1"]].to_string(index=False), flush=True)
    print(json.dumps(check, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register", "run", "verify"])
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "register":
        register()
    elif args.command == "run":
        assert args.output and args.workers > 0
        run(args.output, args.workers)
    else:
        assert args.output
        verify(args.output)
