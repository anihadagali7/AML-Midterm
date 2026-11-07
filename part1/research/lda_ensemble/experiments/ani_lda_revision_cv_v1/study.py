"""Inductive evaluation of the previously unbenchmarked 8a8c3fc Ani revision."""
import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(variable, "1")
from pathlib import Path
import sys
import ast
import json
import datetime
from types import SimpleNamespace
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

OUT = Path(__file__).resolve().parent
PRIOR = OUT.parent / "feature_gap_cv_v1"
sys.path.insert(0, str(PRIOR))
import run as runner
import features as f
BASE = runner.read(PRIOR / "protocol.json")
NOTEBOOK = runner.PART1 / "research/ani_snapshots/8a8c3fc/part1_best_model.ipynb"
ORIGINAL_REPORT = runner.report
SETTINGS = {
    "revision_full": dict(C=.1, teacher_lda=.2, final_lda=.2),
    "regularization_only": dict(C=.1, teacher_lda=0., final_lda=0.),
    "lda_both_c03": dict(C=.3, teacher_lda=.2, final_lda=.2),
    "lda_final_only": dict(C=.3, teacher_lda=0., final_lda=.2),
}
BLOCKS = {name: str(settings) for name, settings in SETTINGS.items()}


def fit_base(x, y, C, lda_weight):
    rows, labels, weights = [], [], []
    for i, target in enumerate(y):
        positive = np.flatnonzero(target)
        rows.extend([i]*len(positive))
        labels.extend(positive)
        weights.extend([1./len(positive)]*len(positive))
    labels, weights = np.asarray(labels), np.asarray(weights)
    counts = np.bincount(labels, weights=weights, minlength=5)+1e-9
    weights = weights*counts.mean()/counts[labels]
    model = LogisticRegression(C=C, max_iter=3000).fit(x[rows], labels, sample_weight=weights)
    lda = None
    if lda_weight:
        k = len(np.unique(labels))
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=np.ones(k)/k).fit(x[rows], labels)
    return dict(lr=model, lda=lda, lda_weight=lda_weight,
                neighbors=normalize(x), targets=y/y.sum(axis=1, keepdims=True))


def predict_base(base, query):
    lr = np.zeros((len(query), 5))
    lr[:, base["lr"].classes_] = base["lr"].predict_proba(query)
    similarity = normalize(query) @ base["neighbors"].T
    nearest = np.argsort(-similarity, axis=1)[:, :10]
    weights = np.take_along_axis(similarity, nearest, axis=1).clip(0)**2
    knn = (weights[:, :, None]*base["targets"][nearest]).sum(axis=1)/(weights.sum(axis=1, keepdims=True)+1e-9)
    result = (1-.3-base["lda_weight"])*lr+.3*knn
    if base["lda"] is not None:
        lda = np.zeros_like(lr)
        lda[:, base["lda"].classes_] = base["lda"].predict_proba(query)
        result += base["lda_weight"]*lda
    return result


def geometry(packet):
    if "revision_geometry" not in packet:
        pc = PCA(30, random_state=0).fit(packet["z"])
        gmm = GaussianMixture(30, covariance_type="diag", n_init=3, random_state=0).fit(pc.transform(packet["z"]))
        assert gmm.converged_
        packet["revision_geometry"] = dict(pca=pc, gmm=gmm, resp=gmm.predict_proba(pc.transform(packet["z"])))
    return packet["revision_geometry"]


def fitted_variant(packet, settings):
    key = tuple(settings.values())
    cache = packet.setdefault("revision_models", {})
    if key in cache:
        return cache[key]
    z, gold, y = packet["z"], packet["gold"], packet["y"]
    unknown = np.setdiff1d(np.arange(len(z)), gold)
    resp = geometry(packet)["resp"]
    rows, targets = gold.copy(), y.copy()
    history = []
    for _ in range(2):
        base = fit_base(z[rows], targets, settings["C"], settings["teacher_lda"])
        t = targets/targets.sum(axis=1, keepdims=True)
        rates = (resp[rows].T@t+t.mean(axis=0))/(resp[rows].sum(axis=0)[:, None]+1)
        probability = .8*predict_base(base, z[unknown])+.2*(resp[unknown]@rates)
        keep = probability.max(axis=1)>=.8
        rows = np.r_[gold, unknown[keep]]
        targets = np.vstack([y, f.native.decide(probability, .3)[keep]])
        history.append(dict(accepted=int(keep.sum()), reason_counts=targets[len(y):].sum(axis=0).tolist()))
    result = dict(base=fit_base(z[rows], targets, settings["C"], settings["final_lda"]),
                  history=history, rows=rows, targets=targets)
    cache[key] = result
    return result


def fit_addition(packet, fit, query, name):
    model = fitted_variant(packet, SETTINGS[name])
    result = f.centered(predict_base(model["base"], packet["scaler"].transform(query[f.EMB].to_numpy())))
    return result, result.copy(), dict(block=name, settings=SETTINGS[name], history=model["history"],
        original_notebook_commit="8a8c3fc", access="inductive adaptation; all transforms fit permitted training rows",
        gold_rows=len(packet["gold"]), query_rows=len(query), no_test_inputs=True)


def guard():
    p = runner.read(OUT / "protocol.json")
    for path, digest in p["source_sha256"].items():
        assert runner.sha(runner.PART1 / path) == digest, path
    assert p["registry_sha256"] == runner.sha(OUT / "registry.json")
    return p


def prepare():
    if (OUT / "protocol.json").exists():
        guard()
        return
    sources = dict(BASE["source_sha256"])
    for path in [Path(__file__), PRIOR / "protocol.json", NOTEBOOK]:
        sources[str(path.relative_to(runner.PART1))] = runner.sha(path)
    registry = [dict(name=parent+"__"+name, parent=parent, block=name, description=description)
                for name, description in BLOCKS.items() for parent in runner.CONTROL_NAMES]
    runner.save(OUT / "registry.json", registry)
    p = dict(BASE)
    p.update(version="ani-lda-revision-cv-v1", frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sources, registry_sha256=runner.sha(OUT / "registry.json"), candidates=len(registry), hypotheses=BLOCKS,
        fixed_recipe="Latest Ani source: C.1 multinomial LR, .5 LR/.3 fractional cosine kNN/.2 shrinkage LDA, uniform LDA class priors and duplicated multi-reason labels. Same mixture in two pseudo-label rounds with .2 GMM contribution, confidence.8 and threshold.3. Fit every transform inductively on allowed rows. Controls isolate C.1 without LDA, LDA with original C.3, and LDA only in final prediction with original C.3 and pseudo-labels.",
        initial="Focused eight-repeat comparison of all four fixed recipes in both parent pipelines; all advance regardless of first-three results.",
        shortlist="All eight parent-specific candidates, declared before scoring.",
        metadata_diagnostic="Diagnostic duplicates the new Ani source, not a metadata-only model.",
        limitations="The original notebook fits input transforms transductively and uses different validation folds. This is an inductive, grouped-CV adaptation; its scores cannot be assigned to the notebook's unscored submission CSV.")
    runner.save(OUT / "protocol.json", p)
    print("Frozen eight parent-specific comparisons of latest Ani and matched ablations.", flush=True)


def parity():
    guard()
    frame = runner.frame_for("pooled")
    p, positions, groups, masked = runner.plan("pooled3", frame)
    fit, query, h = runner.fold_inputs(frame, positions, groups, masked, p["folds"][0])
    packet = f.train_native(fit)
    zq = packet["scaler"].transform(query[f.EMB].to_numpy())
    original = f.native_prediction(packet, query)
    np.testing.assert_allclose(original, runner.controls("pooled3", frame.iloc[positions])["ani"][0, h], atol=1e-10, rtol=0)
    control = fitted_variant(packet, dict(C=.3, teacher_lda=0., final_lda=0.))
    np.testing.assert_array_equal(control["rows"], packet["rows"])
    np.testing.assert_array_equal(control["targets"], packet["targets"])
    np.testing.assert_allclose(f.centered(predict_base(control["base"], zq)), original, atol=1e-10, rtol=0)
    # Execute function definitions only. Never execute notebook loading, global
    # transductive transforms, evaluation, or export cells.
    notebook = json.loads(NOTEBOOK.read_text())
    source = next("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code" and "def final_model(" in "".join(c["source"]))
    tree = ast.parse(source)
    assert all(isinstance(node, ast.FunctionDef) for node in tree.body)
    geom = geometry(packet)
    scope = dict(np=np, normalize=normalize, LogisticRegression=LogisticRegression,
        LinearDiscriminantAnalysis=LinearDiscriminantAnalysis, EMB=f.EMB, pca30=geom["pca"], gmm=geom["gmm"])
    exec(compile(tree, str(NOTEBOOK)+":function_definitions", "exec"), scope)
    unknown = np.setdiff1d(np.arange(len(fit)), packet["gold"])
    notebook_score = scope["final_model"](packet["z"][packet["gold"]], packet["y"], zq, packet["z"][unknown])
    adapted, _, _ = fit_addition(packet, fit, query, "revision_full")
    np.testing.assert_allclose(adapted, f.centered(notebook_score), atol=1e-10, rtol=0)
    for name in BLOCKS:
        full, _, _ = fit_addition(packet, fit, query, name)
        first, _, _ = fit_addition(packet, fit, query.iloc[:1], name)
        np.testing.assert_allclose(full[:1], first, atol=1e-10, rtol=0)
    runner.save(OUT / "parity.json", dict(passed=True, native_control_reproduced=True,
        original_notebook_functions_reproduced_on_same_allowed_inputs=True, query_batch_invariance=True,
        native_augmentation_rows_and_labels_identical=True, notebook_data_loading_cells_executed=False))


def shortlist():
    rows = runner.comparisons()
    runner.save(OUT / "screen_comparison.json", rows)
    runner.pd.DataFrame(rows).to_csv(OUT / "screen_comparison.csv", index=False)
    runner.save(OUT / "shortlist.json", dict(names=[r["name"] for r in runner.read(OUT / "registry.json")],
        diagnostic_only=False, selection_rule="All recipes, predeclared before scoring"))


def report():
    ORIGINAL_REPORT()
    path = OUT / "report.html"
    body = path.read_text().replace("Part 1 feature gap experiments", "Latest Ani LDA revision")
    body = body.replace("Part 1: new feature structure and earlier-feature retries", "Inductive comparison of the latest Ani revision")
    start = body.index("<p>Nine blocks")
    stop = body.index("</p>", start)+4
    body = body[:start]+'<p>The previously unbenchmarked 8a8c3fc notebook adds shrinkage LDA and changes logistic C from .3 to .1. Test the exact mixture in an inductive adaptation plus controls isolating regularization, LDA integration and pseudo-label teaching. All eight recipes were predeclared for all eight repeats in both pools. Only supplied Part 1 inputs were used.</p>'+body[stop:]
    body = body.replace("Metadata-only diagnostics", "Source-score diagnostics")
    path.write_text(body)
    (OUT / "metadata_only.csv").rename(OUT / "source_diagnostics.csv")


def independent_verify():
    import verify_saved
    verify_saved.HERE = OUT
    verify_saved.main()


runner.OUT = OUT
runner.m = SimpleNamespace(BLOCKS=BLOCKS, LABELS=f.LABELS, EMB=f.EMB,
    train_native=f.train_native, native_prediction=f.native_prediction, fit_addition=fit_addition)
runner.prepare, runner.guard, runner.parity = prepare, guard, parity
runner.shortlist, runner.report = shortlist, report
if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "independent_verify":
        independent_verify()
    else:
        runner.main()
