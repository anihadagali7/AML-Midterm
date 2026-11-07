"""Ablate fractional label normalization during pseudo-label teaching only."""
import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(variable, "1")
from pathlib import Path
import sys
import datetime
from types import SimpleNamespace
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

OUT = Path(__file__).resolve().parent
PRIOR = OUT.parent / "feature_gap_cv_v1"
sys.path.insert(0, str(PRIOR))
import run as runner
import features as f
BASE = runner.read(PRIOR / "protocol.json")
ORIGINAL_REPORT = runner.report
OPTIONS = {"gmm_binary": (True, False), "neighbors_binary": (False, True), "both_binary": (True, True)}
BLOCKS = {name + "_" + scope: f"Independent teacher votes in {name}; predict {scope}"
          for name in OPTIONS for scope in ["all", "other"]}


def augment(packet, gmm_binary, neighbors_binary):
    key = (gmm_binary, neighbors_binary)
    cache = packet.setdefault("teacher_variants", {})
    if key in cache:
        return cache[key]
    z, gold, y = packet["z"], packet["gold"], packet["y"]
    if "teacher_responsibilities" not in packet:
        pc = PCA(30, random_state=0).fit(z)
        gmm = GaussianMixture(30, covariance_type="diag", n_init=3, random_state=0).fit(pc.transform(z))
        assert gmm.converged_
        packet["teacher_responsibilities"] = gmm.predict_proba(pc.transform(z))
    resp = packet["teacher_responsibilities"]
    unknown = np.setdiff1d(np.arange(len(z)), gold)
    rows, targets = gold.copy(), y.copy()
    history = []
    for _ in range(2):
        base = f.native.fit_ani_base(z[rows], targets)
        mapping = targets if gmm_binary else targets / targets.sum(axis=1, keepdims=True)
        rates = (resp[rows].T @ mapping + mapping.mean(axis=0)) / (resp[rows].sum(axis=0)[:, None] + 1)
        if neighbors_binary:
            base = (base[0], base[1], targets.astype(float))
        probability = .8*f.native.ani_base_predict(base, z[unknown]) + .2*(resp[unknown] @ rates)
        assert (probability >= -1e-12).all() and (probability <= 1+1e-12).all()
        keep = probability.max(axis=1) >= .8
        generated = f.native.decide(probability, .3)[keep]
        rows = np.r_[gold, unknown[keep]]
        targets = np.vstack([y, generated])
        history.append(dict(accepted=int(keep.sum()), multi_reason_rows=int((generated.sum(axis=1)>1).sum()),
                            reason_positive_counts=generated.sum(axis=0).tolist()))
    result = dict(base=f.native.fit_ani_base(z[rows], targets), rows=rows, targets=targets, history=history)
    cache[key] = result
    return result


def fit_addition(packet, fit, query, name):
    variant, scope = name.rsplit("_", 1)
    generated = augment(packet, *OPTIONS[variant])
    zq = packet["scaler"].transform(query[f.EMB].to_numpy())
    result = f.centered(f.native.ani_base_predict(generated["base"], zq))
    if scope == "other":
        original = f.native_prediction(packet, query)
        original[:, 4] = result[:, 4]
        result = original
    audit = dict(block=name, teacher_settings=dict(zip(["binary_gmm_mapping", "binary_neighbor_targets"], OPTIONS[variant])),
                 history=generated["history"], gold_multi_reason_rows=int((packet["y"].sum(axis=1)>1).sum()),
                 confidence=.8, positive_label_cutoff=.3, rounds=2, final_fractional_classifier_unchanged=True)
    return result, result.copy(), audit


def guard():
    p = runner.read(OUT / "protocol.json")
    for name, digest in p["source_sha256"].items():
        assert runner.sha(runner.PART1/name) == digest, name
    assert p["registry_sha256"] == runner.sha(OUT / "registry.json")
    return p


def prepare():
    if (OUT / "protocol.json").exists():
        guard()
        return
    sources = dict(BASE["source_sha256"])
    for path in [Path(__file__), PRIOR / "protocol.json", OUT.parent / "neighbor_labels_cv_v1/confirmation.json"]:
        sources[str(path.relative_to(runner.PART1))] = runner.sha(path)
    rows = [dict(name=parent + "__" + block, parent=parent, block=block, description=description)
            for block, description in BLOCKS.items() for parent in runner.CONTROL_NAMES]
    runner.save(OUT / "registry.json", rows)
    p = dict(BASE)
    p.update(version="multilabel-teacher-cv-v1", frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sources, registry_sha256=runner.sha(OUT / "registry.json"), hypotheses=BLOCKS, candidates=len(rows),
        hypothesis_detail="Native teacher scores sum to at most1. Confidence>=.8 and label cutoff.3 therefore cannot generate two positives. Test independent binary label values in GMM mapping, teaching kNN, or both; keep confidence and cutoff unchanged.",
        fixed_recipe="Scaler, PCA30, diagonal GMM30, gold labels, two rounds, confidence.8, cutoff.3, final multinomial LR and fractional kNN unchanged. Only pseudo-label teaching uses binary rather than normalized multi-reason targets in the declared component. Apply retrained source to all reasons or Other only.",
        initial="Focused all-eight-repeat comparison in both pools, declared before any candidate scores.",
        shortlist="Every fixed recipe advances to the five additional saved repeats. No screening-based pruning.",
        metadata_diagnostic="Diagnostic duplicates the source variant; no metadata-only model.",
        limitations="The native restriction may be beneficial conservative pseudo-labeling, not necessarily a defect. Additional multi-reason pseudo-labels can be wrong; counts do not establish accuracy. No masked or test truth is available.")
    runner.save(OUT / "protocol.json", p)
    print("Frozen three teacher-normalization ablations, two scopes, two parents; eight repeats.", flush=True)


def parity():
    guard()
    frame = runner.frame_for("pooled")
    p, pos, groups, masked = runner.plan("pooled3", frame)
    fit, query, h = runner.fold_inputs(frame, pos, groups, masked, p["folds"][0])
    packet = f.train_native(fit)
    unchanged = augment(packet, False, False)
    np.testing.assert_array_equal(unchanged["rows"], packet["rows"])
    np.testing.assert_array_equal(unchanged["targets"], packet["targets"])
    assert all(step["multi_reason_rows"] == 0 for step in unchanged["history"])
    original = f.native_prediction(packet, query)
    np.testing.assert_allclose(original, runner.controls("pooled3", frame.iloc[pos])["ani"][0, h], atol=1e-10, rtol=0)
    for name in BLOCKS:
        full, _, audit = fit_addition(packet, fit, query, name)
        first, _, _ = fit_addition(packet, fit, query.iloc[:1], name)
        np.testing.assert_allclose(full[:1], first, atol=1e-10, rtol=0)
        if name.endswith("_other"):
            np.testing.assert_array_equal(full[:, :4], original[:, :4])
    runner.save(OUT / "parity.json", dict(passed=True, native_augmentation_targets_and_rows_identical=True,
        native_generated_multilabel_rows=0, query_invariance=True, scope_checks=True))


def shortlist():
    rows = runner.comparisons()
    runner.save(OUT / "screen_comparison.json", rows)
    runner.pd.DataFrame(rows).to_csv(OUT / "screen_comparison.csv", index=False)
    runner.save(OUT / "shortlist.json", dict(names=[r["name"] for r in runner.read(OUT / "registry.json")],
        diagnostic_only=False, selection_rule="All recipes as declared before scoring"))


def report():
    ORIGINAL_REPORT()
    path = OUT / "report.html"
    body = path.read_text().replace("Part 1 feature gap experiments", "Multi-reason pseudo-label teaching")
    body = body.replace("Part 1: new feature structure and earlier-feature retries", "Independent reason values during pseudo-label teaching")
    start, stop = body.index("<p>Nine blocks"), body.index("</p>", body.index("<p>Nine blocks")) + 4
    body = body[:start] + '<p>Three controlled changes to the pseudo-label teacher: binary GMM mapping, binary kNN votes, or both. Native final LR and kNN remain fractional, isolating generated-label changes. Every recipe was predeclared for eight repeats in both pools. Multi-label acceptance counts are diagnostic, not evidence that generated labels are correct. Part 1 inputs only.</p>' + body[stop:]
    body = body.replace("Metadata-only diagnostics", "Source-variant diagnostics")
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
