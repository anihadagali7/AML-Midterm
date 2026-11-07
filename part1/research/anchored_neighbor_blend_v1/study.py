"""Fixed neighbor-vote and 75/25 ensemble experiments on all saved Part 1 splits."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
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
from threadpoolctl import threadpool_limits

STUDY = Path(__file__).resolve().parent
PART1, ROOT = STUDY.parents[1], STUDY.parents[2]
FROZEN_HELPER = STUDY.parent/"anchored_transductive_v1"
sys.path[:0] = [str(PART1), str(FROZEN_HELPER)]
spec = importlib.util.spec_from_file_location("frozen_anchored_benchmark", FROZEN_HELPER/"run.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
from anchored_solution import AnchoredReasonClassifier
from neighbor_models import NeighborAdjustedAnchored, centered, decide, embedding_groups

REFERENCE = PART1/"experiments/current_leaders_grouped_v1"
LDA_ROOT = PART1/"research/lda_ensemble/experiments/ani_lda_revision_cv_v1"
LABELS, FEATURES = benchmark.LABELS, benchmark.FEATURES
MODES = ["anchored", "grouped_neighbors", "pseudo_neighbor_quarter", "anchored75_lda25", "lda_control"]
CANDIDATES = MODES[1:4]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def sources():
    paths = [Path(__file__), STUDY/"neighbor_models.py", STUDY/"check_neighbor_math.py", FROZEN_HELPER/"run.py", FROZEN_HELPER/"models.py",
             PART1/"anchored_solution.py", PART1/"TESTING_POLICY.md",
             PART1/"data/part1_train.csv", PART1/"data/part1_val.csv", PART1/"data/part1_test.csv",
             REFERENCE/"diagnostic_predictions.csv", REFERENCE/"comparison.csv",
             PART1/"submissions/results/anchored-trainval235.json",
             PART1/"submissions/results/lda-ensemble-trainval235.json",
             PART1/"submissions/archive/submitted/anchored_reason_trainval235__kaggle_0.89636.csv",
             PART1/"research/lda_ensemble/experiments/final_lda_ensemble_trainval235_v1/frozen_recipe.json"]
    for pool in ["train", "pooled"]:
        paths += [REFERENCE/f"anchored_{pool}.npz", REFERENCE/"snapshots"/f"lda_ensemble_{pool}.npz"]
    paths += [REFERENCE/"snapshots"/(s+".json") for s in ["train3", "train5", "pooled3", "pooled5", "chronology"]]
    paths += [LDA_ROOT/"supplementary"/f"window{i}_ensemble__revision_full.npz" for i in range(4)]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def register():
    p = STUDY/"protocol.json"
    assert not p.exists()
    save(p, dict(
        version="anchored-neighbor-blend-v1", registered_at=datetime.now(timezone.utc).isoformat(),
        parent="Frozen anchored 0.89636 recipe and predictions, unchanged inductive fitting.",
        authorization="User requested all three proposed changes: duplicate neighbor influence, generated-label vote weighting, and a fixed anchored/LDA blend.",
        candidates={
            "grouped_neighbors": "At each existing generated-label pass and final prediction, collapse the current neighbor bank by the saved rounded raw-embedding group. Use the first member's vector, average member binary label vectors, preserve first-occurrence order, and retrieve ten distinct groups with one cosine-squared weight each. LR, LDA and preprocessing still fit the unchanged row population.",
            "pseudo_neighbor_quarter": "Keep ordinary ten-row neighbor retrieval. Multiply each generated-row neighbor's cosine-squared weight by 0.25, gold by 1, then normalize by their summed weights. Apply in teaching and final prediction. All other settings fixed.",
            "anchored75_lda25": "0.75 * expit(logit(clip(anchored_score,1e-9,1-1e-9))-logit(0.3)) + 0.25 * frozen LDA ensemble score. Cutoff 0.5 plus nonempty argmax. Scores share a decision center; they are not asserted to be calibrated probabilities."
        },
        hypotheses=["Repeated embeddings may crowd distinct evidence out of a ten-row neighborhood.",
                    "Reduced generated-label influence may help neighbors avoid reinforcing mistakes, as anchored LR already does.",
                    "The two competitive submitted recipes may correct complementary errors."],
        controls="Fresh unchanged anchored fit on every case. Frozen LDA scores are reused only after exact saved-plan, target-ID, truth, shape and decoder checks; standalone LDA metrics must match the common comparison.",
        schedule="All eight saved five-fold repeats in BOTH pools and validation plus three chronological windows for all three candidates, without screening or tuning. 252 new scored fits (84 each for parent and two neighbor variants); blend and standalone LDA use 84 matched saved LDA evaluations.",
        implementation_checks="One first-fold row-control subclass refit must match the parent; synthetic checks ensure duplicate collapse, conflicting-label averaging and quarter-weight normalization implement the stated changes.",
        supplementary_exception="Evaluate every candidate on all four supplementary cases regardless of CV outcome, declared before scoring.",
        advancement="Positive mean paired sample-F1 difference and at least 5/8 wins against anchored in BOTH pools. Report validation and every chronological window without pooling unlike checks. Qualification only warrants review; no automatic leader replacement, final235 test fit, upload, merge or push.",
        access="Strict inductive: original saved fitting IDs/order and masked-group purges; Train157 excludes validation-source rows. No query/test inputs or labels enter fitting; actual test file is hashed only.",
        scoring="Complete OOF sample F1 for each repeat, population SD, paired repeat wins/ties/losses, macro F1 and all label precision/recall/F1/support. Chronology pooled across the same 98 held-out predictions.",
        no_search="Exactly three proposed arms. No combined neighbor variant, other vote weights, other blending weights, decoder search or selected seeds.",
        limitations="Reused development folds and small later-year samples; local improvement need not improve Kaggle. The registered 75/25 blend is evaluated on existing cross-fitted components, not fitted to outer labels.",
        retention="Keep complete predictions for qualifying candidates/useful controls; otherwise verify and retain compact recipes, all metrics, fit-access summaries and failure notes, then discard temporary trial-level outputs.",
        versions={"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        sources=sources()))
    print("Registered", p)


def lda_for(case):
    if case["pool"] != "supplementary":
        path = REFERENCE/"snapshots"/f"lda_ensemble_{case['pool']}.npz"
        with np.load(path, allow_pickle=True) as z:
            b, ri = case["positions"], case["repeat_index"]
            data = dict(ids=z["ids"][b], truth=z["truth"][b], scores=z["scores"][ri, b], predictions=z["predictions"][ri, b])
    else:
        windows = json.loads((REFERENCE/"snapshots/chronology.json").read_text())["chronological_splits"]
        names = ["validation"]+[w["name"] for w in windows]
        path = LDA_ROOT/"supplementary"/f"window{names.index(case['key'])}_ensemble__revision_full.npz"
        with np.load(path, allow_pickle=True) as z:
            positions = {v: i for i, v in enumerate(z["ids"])}
            b = [positions[v] for v in case["query"].ID]
            data = {k: z[k][b] for k in ["ids", "truth", "scores", "predictions"]}
    np.testing.assert_array_equal(data["ids"], case["query"].ID)
    np.testing.assert_array_equal(data["truth"], case["query"][LABELS])
    np.testing.assert_array_equal(decide(data["scores"]), data["predictions"])
    return data["scores"]


def evaluate(case):
    start = time.monotonic()
    fit, query = case["fit"], case["query"]
    x, y, qx = fit[FEATURES].to_numpy(float), fit[LABELS].to_numpy(int), query[FEATURES].to_numpy(float)
    lda = lda_for(case)
    audits, scores = [], []
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        parent = AnchoredReasonClassifier().fit(x, y)
        raw = parent.predict_proba(qx)
        scores.append(centered(raw))
        audits.append(dict(mode="anchored", rounds=parent.round_audit_))
        np.testing.assert_array_equal(decide(scores[0]), parent.predict(qx))
        for mode in CANDIDATES[:2]:
            model = NeighborAdjustedAnchored(mode).fit(x, y)
            scores.append(centered(model.predict_proba(qx)))
            audits.append(dict(mode=mode, rounds=model.round_audit_, neighbors=model.neighbor_audit_))
        error = None
        if case["key"] == "train_20261126_0":
            control = NeighborAdjustedAnchored("row_control").fit(x, y).predict_proba(qx)
            np.testing.assert_allclose(control, raw, atol=1e-10, rtol=0)
            error = float(np.max(np.abs(control-raw)))
    scores += [.75*scores[0]+.25*lda, lda]
    scores = np.stack(scores)
    predictions = np.stack([decide(s) for s in scores])
    packet = dict(ids=query.ID.to_numpy(str), truth=query[LABELS].to_numpy(int), scores=scores,
                  predictions=predictions, parent_raw_scores=raw)
    assert np.isfinite(scores).all()
    audit = {k: v for k, v in case.items() if k not in ["fit", "query"]}
    audit.update(models=audits, input_rows=len(fit), gold_rows=int(fit[LABELS].ge(0).all(axis=1).sum()),
                 fit_groups=len(np.unique(embedding_groups(x))), target_rows=len(query),
                 fit_ids_sha256=hashlib.sha256("\n".join(fit.ID).encode()).hexdigest(),
                 query_inputs_used_for_fit=False, query_labels_used_for_fit=False,
                 row_control_max_error=error, seconds=time.monotonic()-start)
    return case["key"], packet, audit


def run(output, workers):
    protocol = json.loads((STUDY/"protocol.json").read_text())
    assert sources() == protocol["sources"]
    output = Path(output).resolve()
    assert not output.exists(), "Choose a new output directory."
    (output/"predictions").mkdir(parents=True)
    audits = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(evaluate, c) for c in benchmark.cases()]
        for future in as_completed(futures):
            key, packet, audit = future.result()
            np.savez_compressed(output/"predictions"/(key+".npz"), **packet)
            audits.append(audit)
            print(json.dumps(dict(completed=len(audits), total=84, case=key)), flush=True)
    save(output/"fit_audits.json", sorted(audits, key=lambda a: a["key"]))
    verify(output)


def verify(output):
    output = Path(output).resolve()
    assert sources() == json.loads((STUDY/"protocol.json").read_text())["sources"]
    refs = {}
    for pool in ["train", "pooled"]:
        with np.load(REFERENCE/f"anchored_{pool}.npz", allow_pickle=True) as z:
            refs[pool] = {k: z[k] for k in z.files}
    old_diagnostic = pd.read_csv(REFERENCE/"diagnostic_predictions.csv")
    old_diagnostic = old_diagnostic[old_diagnostic.name == "anchored_candidate"]
    entries, supplementary, errors = [], [], []
    for c in benchmark.cases():
        with np.load(output/"predictions"/(c["key"]+".npz"), allow_pickle=False) as z:
            np.testing.assert_array_equal(z["ids"], c["query"].ID)
            np.testing.assert_array_equal(z["truth"], c["query"][LABELS])
            assert z["scores"].shape == (len(MODES), len(c["query"]), 5)
            np.testing.assert_allclose(z["scores"][4], lda_for(c), atol=0, rtol=0)
            np.testing.assert_allclose(z["scores"][3], .75*centered(z["parent_raw_scores"])+.25*z["scores"][4], atol=0, rtol=0)
            if c["pool"] != "supplementary":
                b, ri, ref = c["positions"], c["repeat_index"], refs[c["pool"]]
                np.testing.assert_array_equal(z["ids"], ref["ids"][b])
                np.testing.assert_allclose(z["parent_raw_scores"], ref["scores"][ri, b], atol=1e-7, rtol=0)
                np.testing.assert_array_equal(z["predictions"][0], ref["predictions"][ri, b])
                errors.append(float(np.max(np.abs(z["parent_raw_scores"]-ref["scores"][ri, b]))))
            else:
                old = old_diagnostic[old_diagnostic["case"] == c["key"]].set_index("ID").loc[z["ids"]]
                np.testing.assert_array_equal(z["predictions"][0], old[["predicted_"+r for r in LABELS]])
            for i, mode in enumerate(MODES):
                np.testing.assert_array_equal(z["predictions"][i], decide(z["scores"][i]))
                frame = pd.DataFrame(dict(ID=z["ids"], pool=c["pool"], seed=c["seed"], case=c["key"], mode=mode))
                for j, label in enumerate(LABELS):
                    frame["actual_"+label] = z["truth"][:, j]
                    frame["predicted_"+label] = z["predictions"][i, :, j]
                entries.append(frame)
                if c["pool"] == "supplementary":
                    supplementary.append(dict(case=c["key"], mode=mode, **benchmark.metrics(z["truth"], z["predictions"][i])))
    combined = pd.concat(entries, ignore_index=True)
    truth, prediction = [[prefix+r for r in LABELS] for prefix in ["actual_", "predicted_"]]
    records = []
    for (pool, seed, mode), g in combined[combined.pool != "supplementary"].groupby(["pool", "seed", "mode"]):
        assert g.ID.is_unique and set(g.ID) == set(refs[pool]["ids"])
        records.append(dict(pool=pool, seed=int(seed), mode=mode, **benchmark.metrics(g[truth], g[prediction])))
    for mode, g in combined[(combined.pool == "supplementary") & (combined["case"] != "validation")].groupby("mode"):
        assert len(g) == 98 and g.ID.is_unique
        supplementary.append(dict(case="chronology98", mode=mode, **benchmark.metrics(g[truth], g[prediction])))
    repeats = pd.DataFrame(records)
    assert len(repeats) == 80
    summary = repeats.groupby(["pool", "mode"]).agg(mean_sample_f1=("sample_f1", "mean"),
             population_sd=("sample_f1", lambda s: s.std(ddof=0)), mean_macro_f1=("macro_f1", "mean")).reset_index()
    pairs, differences, qualifies = [], [], {m: True for m in CANDIDATES}
    for pool in ["train", "pooled"]:
        pivot = repeats[repeats.pool == pool].pivot(index="seed", columns="mode", values="sample_f1")
        for mode in CANDIDATES:
            d = pivot[mode]-pivot.anchored
            wins, ties, losses = int((d > 1e-12).sum()), int((d.abs() <= 1e-12).sum()), int((d < -1e-12).sum())
            passes = bool(d.mean() > 0 and wins >= 5)
            qualifies[mode] &= passes
            pairs.append(dict(pool=pool, mode=mode, mean_delta=float(d.mean()), wins=wins, ties=ties,
                              losses=losses, meets_pool_rule=passes))
            differences += [dict(pool=pool, mode=mode, seed=int(s), sample_f1_delta=float(v)) for s, v in d.items()]
    supp = pd.DataFrame(supplementary)
    # Independently establish that the reused LDA records are the published comparator.
    common = pd.read_csv(REFERENCE/"comparison.csv").set_index("name")
    for pool in ["train", "pooled"]:
        actual = summary[(summary.pool == pool) & (summary["mode"] == "lda_control")].mean_sample_f1.item()
        np.testing.assert_allclose(actual, common.loc["lda_ensemble", pool], atol=1e-12, rtol=0)
    for case in ["validation", "chronology98"]:
        actual = supp[(supp["case"] == case) & (supp["mode"] == "lda_control")].sample_f1.item()
        np.testing.assert_allclose(actual, common.loc["lda_ensemble", case], atol=1e-12, rtol=0)
    repeats.to_csv(output/"repeat_scores.csv", index=False)
    summary.to_csv(output/"cv_summary.csv", index=False)
    pd.DataFrame(pairs).to_csv(output/"paired_summary.csv", index=False)
    pd.DataFrame(differences).to_csv(output/"paired_repeat_deltas.csv", index=False)
    supp.to_csv(output/"supplementary_scores.csv", index=False)
    audits = json.loads((output/"fit_audits.json").read_text())
    assert len(audits) == 84 and all(not a["query_inputs_used_for_fit"] and not a["query_labels_used_for_fit"] for a in audits)
    check = dict(passed=True, new_cv_fits=240, new_supplementary_fits=12, implementation_control_fits=1,
                 reused_lda_cv_evaluations=80, reused_lda_supplementary_evaluations=4,
                 blend_evaluated_cases=84, parent_frozen_cv_checks=80, parent_supplementary_checks=4,
                 parent_max_score_error=max(errors), row_control_max_error=next(a["row_control_max_error"] for a in audits if a["row_control_max_error"] is not None),
                 repeats_scored=80, qualified_cv=qualifies, all_source_hashes_unchanged=True,
                 lda_standalone_matches_common_comparison=True, actual_test_inputs_used=False,
                 protocol_sha256=sha(STUDY/"protocol.json"))
    save(output/"verification.json", check)
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(pairs).to_string(index=False), flush=True)
    print(supp[supp["case"].isin(["validation", "chronology98"])][["case", "mode", "sample_f1", "macro_f1"]].to_string(index=False), flush=True)
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
