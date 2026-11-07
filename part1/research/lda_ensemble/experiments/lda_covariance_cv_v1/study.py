"""Matched covariance-head follow-ups to the newly confirmed Ani LDA revision."""
import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(variable, "1")
from pathlib import Path
import sys
import datetime
import importlib.util
from types import SimpleNamespace
import numpy as np
from scipy.special import softmax
from scipy.linalg import cho_factor, cho_solve
from sklearn.covariance import LedoitWolf

OUT = Path(__file__).resolve().parent
PRIOR = OUT.parent / "feature_gap_cv_v1"
REVISION = OUT.parent / "ani_lda_revision_cv_v1"
sys.path.insert(0, str(PRIOR))
import run as runner
import features as f
ORIGINALS = {key: getattr(runner, key) for key in ["OUT", "m", "prepare", "guard", "parity", "shortlist", "report"]}
spec = importlib.util.spec_from_file_location("revision_helpers", REVISION / "study.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
for key, value in ORIGINALS.items():
    setattr(runner, key, value)
ORIGINAL_REPORT, ORIGINAL_COMPARE, ORIGINAL_STAGE = runner.report, runner.comparisons, runner.run_stage
BASE = runner.read(REVISION / "protocol.json")
BLOCKS = {"gold_lda": "Replace final LDA with a gold-only LDA",
          "half_gold_lda": "Average final gold-only and augmented LDA",
          "gold_rda": "Gold-only regularized class-specific covariance in PCA30",
          "augmented_rda": "Augmented regularized class-specific covariance in PCA30"}


def rda_fit(x, y):
    means, covariances = [], []
    for j in range(5):
        points = x[y[:, j] == 1]
        assert len(points) >= 2
        estimator = LedoitWolf().fit(points)
        means.append(estimator.location_)
        covariances.append(estimator.covariance_)
    pooled = np.mean(covariances, axis=0)
    ridge = max(np.trace(pooled)/len(pooled), 1.)*1e-6
    factors, logdet = [], []
    for covariance in covariances:
        matrix = .5*covariance+.5*pooled+ridge*np.eye(len(pooled))
        factors.append(cho_factor(matrix, lower=True))
        logdet.append(float(np.linalg.slogdet(matrix)[1]))
    return dict(means=np.array(means), factors=factors, logdet=logdet)


def rda_predict(model, query):
    scores = []
    for j in range(5):
        delta = query-model["means"][j]
        quadratic = np.sum(delta*cho_solve(model["factors"][j], delta.T).T, axis=1)
        scores.append(-.5*(quadratic+model["logdet"][j]))
    return softmax(np.array(scores).T, axis=1)


def fit_addition(packet, fit, query, name):
    if name == "revision_full":
        return source.fit_addition(packet, fit, query, name)
    revised = source.fitted_variant(packet, source.SETTINGS["revision_full"])
    zq = packet["scaler"].transform(query[f.EMB].to_numpy())
    raw = source.predict_base(revised["base"], zq)
    old_lda = np.zeros_like(raw)
    old = revised["base"]["lda"]
    old_lda[:, old.classes_] = old.predict_proba(zq)
    cache = packet.setdefault("covariance_heads", {})
    if name in ["gold_lda", "half_gold_lda"]:
        if "gold_lda" not in cache:
            cache["gold_lda"] = source.fit_base(packet["z"][packet["gold"]], packet["y"], .1, .2)["lda"]
        model = cache["gold_lda"]
        head = np.zeros_like(raw)
        head[:, model.classes_] = model.predict_proba(zq)
        if name == "half_gold_lda":
            head = .5*head+.5*old_lda
        fitting_rows = packet["gold"]
    else:
        fitting_rows = packet["gold"] if name == "gold_rda" else revised["rows"]
        targets = packet["y"] if name == "gold_rda" else revised["targets"]
        pca = source.geometry(packet)["pca"]
        if name not in cache:
            cache[name] = rda_fit(pca.transform(packet["z"][fitting_rows]), targets)
        head = rda_predict(cache[name], pca.transform(zq))
    result = f.centered(np.clip(raw+.2*(head-old_lda), 0, 1))
    return result, f.centered(head), dict(block=name, fitting_ids=fit.iloc[fitting_rows].ID.tolist(),
        gold_rows=len(packet["gold"]), head_rows=len(fitting_rows), head_weight=.2,
        pseudo_label_teacher_unchanged=True, LR_and_neighbors_unchanged=True,
        class_specific_covariance_mix=.5 if name.endswith("rda") else None)


def guard():
    p = runner.read(OUT / "protocol.json")
    for path, digest in p["source_sha256"].items():
        assert runner.sha(runner.PART1/path) == digest, path
    assert p["registry_sha256"] == runner.sha(OUT / "registry.json")
    return p


def prepare():
    if (OUT / "protocol.json").exists():
        guard()
        return
    sources = dict(BASE["source_sha256"])
    for path in [Path(__file__), REVISION/"protocol.json", REVISION/"confirmation.json", REVISION/"summary.json"]:
        sources[str(path.relative_to(runner.PART1))] = runner.sha(path)
    registry = [dict(name=parent+"__"+name, parent=parent, block=name, description=description)
                for name, description in BLOCKS.items() for parent in runner.CONTROL_NAMES]
    runner.save(OUT/"registry.json", registry)
    p = dict(BASE)
    p.update(version="lda-covariance-cv-v1", frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sources, registry_sha256=runner.sha(OUT/"registry.json"), candidates=len(registry), hypotheses=BLOCKS,
        parent="Latest inductive Ani LDA revision and its ensemble. Original Ani/ensemble retained as additional references.",
        fixed_recipe="Latest C.1/.5LR/.3kNN/.2LDA pseudo-label teacher, LR and neighbors unchanged. Replace final .2 LDA head with gold-only LDA, half gold/augmented LDA, or regularized class-specific Gaussian head fitted on gold or augmented targets. RDA uses fitting PCA30, per-class LedoitWolf covariance, half class-specific plus half mean covariance, uniform priors, trace-scaled1e-6 ridge. Gold multi-reason examples belong to each positive class; gold labels unchanged.",
        initial="All eight candidates run all eight repeats, predeclared. Matched revision_full controls always run in both pools.",
        shortlist="All eight candidates; no pruning after initial scores.",
        confirmation="Positive paired means and >=5/8 wins in BOTH pools against corresponding latest revision control, not merely the older original parent.",
        supplementary="At most three confirmed candidates, or at most two diagnostics if none confirm, selected after all-eight comparison against latest revision.",
        metadata_diagnostic="Final discriminant-head scores; not metadata-only.")
    runner.save(OUT/"protocol.json", p)
    print("Frozen eight covariance-head follow-ups with latest-revision matched controls.", flush=True)


def run_stage(stage, blocks):
    return ORIGINAL_STAGE(stage, list(dict.fromkeys([*blocks, "revision_full"])))


def comparisons(extra=False):
    rows = ORIGINAL_COMPARE(extra)
    for row in rows:
        matched = row["parent"]+"__revision_full"
        for pool in ["train", "pooled"]:
            reference = []
            for stage in [pool+"3", pool+"5"] if extra else [pool+"3"]:
                entry = next(r for r in runner.read(OUT/stage/"index.json") if r["candidate"] == matched)
                reference.extend(r["sample_f1"] for r in entry["repeat_scores"])
            delta = np.array(row[pool+"_repeats"])-reference
            row[pool+"_vs_revision_delta"] = float(delta.mean())
            row[pool+"_vs_revision_wins"] = int((delta>1e-12).sum())
            row[pool+"_vs_revision_ties"] = int((abs(delta)<=1e-12).sum())
            row[pool+"_vs_revision_deltas"] = delta.tolist()
        row["matched_revision_control"] = matched
        row["minimum_original_parent_gain"] = row["minimum_parent_gain"]
        row["minimum_parent_gain"] = min(row[pool+"_vs_revision_delta"] for pool in ["train", "pooled"])
        row["qualifies"] = bool(row["minimum_parent_gain"]>1e-12 and all(
            row[pool+"_vs_revision_wins"] >= (5 if extra else 2) for pool in ["train", "pooled"]))
    return sorted(rows, key=lambda r: (-r["minimum_parent_gain"], r["candidate"]))


def parity():
    guard()
    rng = np.random.default_rng(12)
    x = rng.normal(size=(30, 4))
    y = np.eye(5, dtype=int)[np.arange(30)%5]
    model = rda_fit(x, y)
    np.testing.assert_allclose(rda_predict(model, x).sum(axis=1), 1., atol=1e-12)
    frame = runner.frame_for("pooled")
    p, positions, groups, masked = runner.plan("pooled3", frame)
    fit, query, h = runner.fold_inputs(frame, positions, groups, masked, p["folds"][0])
    packet = f.train_native(fit)
    np.testing.assert_allclose(f.native_prediction(packet, query), runner.controls("pooled3", frame.iloc[positions])["ani"][0,h], atol=1e-10, rtol=0)
    revised, _, _ = fit_addition(packet, fit, query, "revision_full")
    with np.load(REVISION/"pooled3/ani__revision_full.npz") as saved:
        np.testing.assert_allclose(revised, saved["scores"][0,h], atol=1e-10, rtol=0)
    for name in BLOCKS:
        full, _, audit = fit_addition(packet, fit, query, name)
        first, _, _ = fit_addition(packet, fit, query.iloc[:1], name)
        np.testing.assert_allclose(full[:1], first, atol=1e-10, rtol=0)
        assert not set(audit["fitting_ids"]) & set(query.ID)
        if name != "augmented_rda":
            assert set(audit["fitting_ids"]) == set(fit.iloc[packet["gold"]].ID)
    runner.save(OUT/"parity.json", dict(passed=True, native_control_parity=True, latest_revision_control_parity=True,
        query_batch_invariance=True, covariance_normalization_checked=True, head_training_boundaries_verified=True))


def shortlist():
    rows = comparisons()
    runner.save(OUT/"screen_comparison.json", rows)
    runner.pd.DataFrame(rows).to_csv(OUT/"screen_comparison.csv", index=False)
    runner.save(OUT/"shortlist.json", dict(names=[r["name"] for r in runner.read(OUT/"registry.json")],
        diagnostic_only=False, selection_rule="All candidates, predeclared before scoring"))


def report():
    ORIGINAL_REPORT()
    path = OUT/"report.html"
    body = path.read_text().replace("Part 1 feature gap experiments", "LDA covariance follow-ups")
    body = body.replace("Part 1: new feature structure and earlier-feature retries", "Covariance heads versus the latest Ani LDA revision")
    start = body.index("<p>Nine blocks")
    stop = body.index("</p>", start)+4
    body = body[:start]+'<p>Keep the newly confirmed LDA revision teacher, logistic head and neighbors fixed. Compare gold-only LDA, gold/augmented LDA averaging, and regularized class-specific Gaussian covariance. Confirmation is measured against the latest LDA revision, whose matched control runs in every fold. Original Ani/ensemble scores are additional references. Only supplied Part 1 inputs are used.</p>'+body[stop:]
    body = body.replace("Metadata-only diagnostics", "Discriminant-head diagnostics")
    body += '<p>Latest-revision paired deltas and wins are in final_comparison.csv; original-parent deltas in the default table are supplementary references.</p>'
    path.write_text(body)
    (OUT/"metadata_only.csv").rename(OUT/"head_diagnostics.csv")


def independent_verify():
    import verify_saved
    verify_saved.HERE = OUT
    verify_saved.main()


runner.OUT = OUT
runner.m = SimpleNamespace(BLOCKS=BLOCKS, LABELS=f.LABELS, EMB=f.EMB,
    train_native=f.train_native, native_prediction=f.native_prediction, fit_addition=fit_addition)
runner.prepare, runner.guard, runner.parity = prepare, guard, parity
runner.run_stage, runner.comparisons = run_stage, comparisons
runner.shortlist, runner.report = shortlist, report
if __name__ == "__main__":
    if len(sys.argv)==2 and sys.argv[1]=="independent_verify":
        independent_verify()
    else:
        runner.main()
