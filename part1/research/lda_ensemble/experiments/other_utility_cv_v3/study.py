"""Sample-F1-utility weighted, cross-fitted Other corrections."""
import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(variable, "1")
from pathlib import Path
import sys
import datetime
from types import SimpleNamespace
import numpy as np
from scipy.special import expit
from scipy.optimize import check_grad

OUT = Path(__file__).resolve().parent
RESIDUAL = OUT.parent / "other_residual_cv_v1"
GAP = OUT.parent / "feature_gap_cv_v1"
sys.path.insert(0, str(RESIDUAL))
import corrector as model
sys.path.insert(0, str(GAP))
import run as runner
f = model.f
BASE = runner.read(RESIDUAL / "protocol.json")
ORIGINAL_REPORT = runner.report
ORIGINAL_LOSS = model.loss_gradient
BLOCKS = {view + "_ridge" + str(ridge): f"Utility weighted Other: {view}, ridge {ridge}"
          for view in ["context", "drug_targets", "criteria_text"] for ridge in [1, 10]}

SCIPY_MINIMIZE = model.minimize
OPTIMIZATION_AUDITS = []

def robust_minimize(fun, x0, args, method, jac, options):
    options = dict(options, maxls=100)
    result = SCIPY_MINIMIZE(fun, x0, args=args, method=method, jac=jac, options=options)
    retried = False
    if not result.success:
        retried = True
        result = SCIPY_MINIMIZE(fun, result.x, args=args, method=method, jac=jac,
                               options=dict(options, ftol=1e-10))
    gradient = float(np.abs(fun(result.x, *args)[1]).max())
    if not result.success and gradient < 1e-6:
        result.success = True
        result.message = 'Stationary point verified independently despite line-search status'
    assert result.success and gradient < 1e-4, (result.message, gradient)
    OPTIMIZATION_AUDITS.append(dict(retried=retried, gradient_max=gradient, message=str(result.message)))
    return result

model.minimize = robust_minimize


def utility_weights(truth, scores):
    without = runner.decide(scores)
    without[:, 4] = 0
    empty = without.sum(axis=1) == 0
    without[np.flatnonzero(empty), scores[empty, :4].argmax(axis=1)] = 1
    with_other = without.copy()
    with_other[:, 4] = 1
    utility = runner.sample_values(truth, with_other) - runner.sample_values(truth, without)
    assert np.array_equal(utility > 0, truth[:, 4] == 1)
    assert (utility[truth[:, 4] == 0] <= 0).all()
    return np.abs(utility) / max(np.abs(utility).mean(), 1e-12), utility


def weighted_loss(w, x, y, offset, ridge, sample_weight):
    z = offset + x @ w
    objective = np.sum(sample_weight*(np.logaddexp(0, z)-y*z)) + .5*ridge*(w@w)
    gradient = x.T @ (sample_weight*(expit(z)-y)) + ridge*w
    return float(objective), gradient


def fit_addition(packet, fit, query, name):
    OPTIMIZATION_AUDITS.clear()
    inner = model.inner_features(packet, fit)
    weights, utility = utility_weights(packet["y"], inner["scores"])
    unweighted, _, _ = model.fit_addition(packet, fit, query, name)
    assert model.loss_gradient is ORIGINAL_LOSS
    try:
        model.loss_gradient = lambda w, x, y, offset, ridge: weighted_loss(w, x, y, offset, ridge, weights)
        result, _, audit = model.fit_addition(packet, fit, query, name)
    finally:
        model.loss_gradient = ORIGINAL_LOSS
    audit.update(optimizer_audits=list(OPTIMIZATION_AUDITS), utility_weighting=True, mean_weight=float(weights.mean()),
                 positive_weight_mean=float(weights[packet["y"][:, 4] == 1].mean()),
                 negative_weight_mean=float(weights[packet["y"][:, 4] == 0].mean()),
                 utility_values=sorted(np.unique(utility).tolist()),
                 target="Inner-OOF sample F1 difference between nonempty predictions with and without Other")
    return result, unweighted, audit


def guard():
    p = runner.read(OUT / "protocol.json")
    for name, digest in p["source_sha256"].items():
        assert runner.sha(runner.PART1 / name) == digest, name
    assert p["registry_sha256"] == runner.sha(OUT / "registry.json")
    return p


def prepare():
    if (OUT / "protocol.json").exists():
        guard()
        return
    sources = dict(BASE["source_sha256"])
    for path in [Path(__file__), RESIDUAL / "protocol.json", RESIDUAL / "confirmation.json"]:
        sources[str(path.relative_to(runner.PART1))] = runner.sha(path)
    rows = [dict(name=parent + "__" + block, parent=parent, block=block, description=description)
            for block, description in BLOCKS.items() for parent in runner.CONTROL_NAMES]
    runner.save(OUT / "registry.json", rows)
    p = dict(BASE)
    p.update(optimizer="L-BFGS-B maxls100; retry from last iterate with ftol1e-10 on failure; require final gradient infinity norm below1e-4; permit failed status only if below1e-6. Same fixed objective. Every fold rerun.", version="other-utility-cv-v3", frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sources, registry_sha256=runner.sha(OUT / "registry.json"), candidates=len(rows),
        hypotheses=BLOCKS,
        fixed_recipe="Same offset-logistic correction and inner group boundaries as residual study. Weight each gold example by absolute sample-F1 change from adding Other to its inner-OOF prediction without Other; if removal empties the set, select highest-scored non-Other label. Zero-utility rows receive zero weight; normalize nonzero total weights to mean1. Positive/negative sign agrees with the true Other label. Penalties1/10 and decoder remain fixed.",
        initial="Focused all-eight-repeat comparison of six utility-weighted heads with both original parents. First three saved separately; no screening-based pruning.",
        shortlist="All twelve fixed recipes advance to the five additional saved repeats, declared before scoring.",
        metadata_diagnostic="Diagnostic scores are matched unweighted Other correctors, not metadata-only models; independently refitted on identical gold rows and source predictions.",
        controls="Original Ani and ensemble plus matched unweighted correctors. Metric-weighting effects and per-label tradeoffs are reported separately.")
    runner.save(OUT / "protocol.json", p)
    print("Frozen twelve utility-weighted recipes on all eight repeats.", flush=True)


def parity():
    guard()
    truth = np.array([[0, 0, 0, 0, 1], [0, 0, 0, 1, 1], [0, 0, 0, 1, 0], [1, 0, 0, 0, 0]])
    score = np.array([[.1, .1, .1, .9, .1]]*4)
    weights, values = utility_weights(truth, score)
    np.testing.assert_allclose(values, [2/3, 1/3, -1/3, 0])
    rng = np.random.default_rng(43)
    x, y, offset, w, sw = rng.normal(size=(15, 4)), rng.integers(0, 2, 15), rng.normal(size=15), rng.normal(size=4), rng.uniform(.1, 3, 15)
    gradient_error = check_grad(lambda b: weighted_loss(b, x, y, offset, 1, sw)[0], lambda b: weighted_loss(b, x, y, offset, 1, sw)[1], w)
    assert gradient_error < 1e-5
    frame = runner.frame_for("pooled")
    p, positions, groups, masked = runner.plan("pooled3", frame)
    fit, query, h = runner.fold_inputs(frame, positions, groups, masked, p["folds"][0])
    packet = f.train_native(fit)
    original = f.native_prediction(packet, query)
    np.testing.assert_allclose(original, runner.controls("pooled3", frame.iloc[positions])["ani"][0, h], atol=1e-10, rtol=0)
    for name in BLOCKS:
        full, _, audit = fit_addition(packet, fit, query, name)
        first, _, audit1 = fit_addition(packet, fit, query.iloc[:1], name)
        np.testing.assert_allclose(full[:1], first, rtol=0, atol=1e-9)
        assert audit["coefficients_sha256"] == audit1["coefficients_sha256"]
        np.testing.assert_array_equal(full[:, :4], original[:, :4])
    runner.save(OUT / "parity.json", dict(passed=True, gradient_error=float(gradient_error),
        verified_utility_examples=True, query_invariance=True, native_control_parity=True, objective_restored_after_each_fit=True))


def shortlist():
    rows = runner.comparisons()
    runner.save(OUT / "screen_comparison.json", rows)
    runner.pd.DataFrame(rows).to_csv(OUT / "screen_comparison.csv", index=False)
    runner.save(OUT / "shortlist.json", dict(names=[r["name"] for r in runner.read(OUT / "registry.json")],
        diagnostic_only=False, selection_rule="Every recipe, as declared before scoring"))


def report():
    ORIGINAL_REPORT()
    path = OUT / "report.html"
    body = path.read_text().replace("Part 1 feature gap experiments", "Sample F1 utility corrections")
    body = body.replace("Part 1: new feature structure and earlier-feature retries", "Correcting Other with sample-F1-derived training weights")
    start, stop = body.index("<p>Nine blocks"), body.index("</p>", body.index("<p>Nine blocks")) + 4
    body = body[:start] + '<p>Gold-only offset-logistic correctors learn from grouped inner-OOF source predictions. Training-example weights reflect the sample-F1 benefit or harm of adding Other, using outer-training labels only. Score-context, protein targets and criteria-text representations each use two frozen penalties. All recipes were predeclared for eight repeats in both pools. Only Part 1 inputs are used.</p>' + body[stop:]
    body = body.replace("Metadata-only diagnostics", "Matched unweighted-corrector diagnostics")
    path.write_text(body)
    (OUT / "metadata_only.csv").rename(OUT / "unweighted_corrector_diagnostics.csv")


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
