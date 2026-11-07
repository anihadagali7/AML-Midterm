"""Versioned cross-fitted Other correction study; Part 1 inputs only."""
import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(variable, "1")
from pathlib import Path
import sys
import datetime
from types import SimpleNamespace
import numpy as np
from scipy.optimize import check_grad

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import corrector as model
sys.path.insert(0, str(model.PRIOR))
import run as runner

f = model.f
BASE = runner.read(model.PRIOR / "protocol.json")
OLD_REPORT = runner.report


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
    for path in [Path(__file__), OUT / "corrector.py", model.PRIOR / "protocol.json",
                 OUT.parent / "neighbor_labels_cv_v1/confirmation.json"]:
        sources[str(path.relative_to(runner.PART1))] = runner.sha(path)
    rows = [dict(name=parent + "__" + block, parent=parent, block=block, description=description)
            for block, description in model.BLOCKS.items() for parent in runner.CONTROL_NAMES]
    runner.save(OUT / "registry.json", rows)
    p = dict(BASE)
    p.update(version="other-residual-cv-v1", frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sources, registry_sha256=runner.sha(OUT / "registry.json"), candidates=len(rows),
        hypotheses=model.BLOCKS,
        fixed_recipe="Only Other is corrected. Fit an offset binary logistic model on outer-training gold using grouped inner-OOF native Ani decision scores. Offset is logit(native centered Other score); learned correction starts at zero and is ridge-penalized including intercept. Penalties1/10 are frozen. No outer-score threshold selection.",
        representations="Other-logit calibration; all-score context (five logits, normalized entropy and margins); context plus gold-neighbor geometry; context plus separate target/disease/criteria metadata blocks. Context standardized on inner OOF rows; blocks normalized by RMS; no ID or year features.",
        inner_boundary="Three iterative multilabel-stratified group folds, seed270201. Each inner fit excludes inner held-out gold inputs and matching masked groups, in addition to the outer purge. Source pseudo-labels are regenerated inside each inner fit. Corrector fits gold labels only. Outer query inputs enter prediction only.",
        metadata_diagnostic="Diagnostic output duplicates corrected Ani scores; it is neither metadata-only nor an extra candidate.",
        controls="Both original parents are retained. Calibration/context variants show whether metadata or geometry adds beyond correcting the source scores. All advancement uses the unchanged parent-specific screen/confirmation rules.",
        limitations="Reused labels and folds; inner OOF inputs reduce in-sample stacking bias but do not create an independent holdout. Strong corrections may damage calibrated decisions; chronology and original validation remain visible.")
    runner.save(OUT / "protocol.json", p)
    print("Frozen six corrective representations x two ridge strengths x two parents.", flush=True)


def parity():
    guard()
    rng = np.random.default_rng(42)
    x, y, offset, w = rng.normal(size=(15, 4)), rng.integers(0, 2, 15), rng.normal(size=15), rng.normal(size=4)
    error = check_grad(lambda b: model.loss_gradient(b, x, y, offset, 1)[0],
                       lambda b: model.loss_gradient(b, x, y, offset, 1)[1], w)
    assert error < 1e-5, error
    frame = runner.frame_for("pooled")
    p, positions, groups, masked = runner.plan("pooled3", frame)
    fit, query, h = runner.fold_inputs(frame, positions, groups, masked, p["folds"][0])
    packet = f.train_native(fit)
    original = f.native_prediction(packet, query)
    refs = runner.controls("pooled3", frame.iloc[positions])
    np.testing.assert_allclose(original, refs["ani"][0, h], rtol=0, atol=1e-10)
    checks = []
    for view in model.VIEWS:
        name = view + "_ridge1"
        all_scores, _, audit = model.fit_addition(packet, fit, query, name)
        first, _, second_audit = model.fit_addition(packet, fit, query.iloc[:1], name)
        np.testing.assert_allclose(all_scores[:1], first, rtol=0, atol=1e-9)
        assert audit["coefficients_sha256"] == second_audit["coefficients_sha256"]
        np.testing.assert_array_equal(all_scores[:, :4], original[:, :4])
        for split in audit["inner_splits"]:
            assert not (set(split["fit_gold_ids"]) | set(split["fit_masked_ids"])) & set(split["query_ids"])
            assert not set(query.ID) & (set(split["fit_gold_ids"]) | set(split["fit_masked_ids"]))
        checks.append(dict(view=view, query_batch_invariance=True, query_does_not_change_coefficients=True,
                           inner_and_outer_id_separation=True, non_other_scores_unchanged=True))
    runner.save(OUT / "parity.json", dict(passed=True, analytic_gradient_error=float(error),
                 native_control_parity=True, corrective_checks=checks))
    print("Gradient, cross-fitting boundaries, native parity and query invariance checks passed.", flush=True)


def report():
    OLD_REPORT()
    path = OUT / "report.html"
    body = path.read_text().replace("Part 1 feature gap experiments", "Cross-fitted Other corrections")
    body = body.replace("Part 1: new feature structure and earlier-feature retries", "Learning Other corrections from inner-held-out predictions")
    start, stop = body.index("<p>Nine blocks"), body.index("</p>", body.index("<p>Nine blocks")) + 4
    body = body[:start] + '<p>Six corrective representations and two fixed ridge strengths, evaluated in Ani and the existing three-way ensemble. Correctors use only outer-training verified labels and predictions generated by three grouped inner folds. All source fits purge held-out input groups, including matching masked rows. Only the Other score changes. Part 1 inputs only.</p>' + body[stop:]
    body = body.replace("Metadata-only diagnostics", "Corrected-source diagnostics (same scores as Ani variants)")
    path.write_text(body)
    (OUT / "metadata_only.csv").rename(OUT / "corrected_source_diagnostics.csv")


def independent_verify():
    import verify_saved
    verify_saved.HERE = OUT
    verify_saved.main()


runner.OUT = OUT
runner.m = SimpleNamespace(BLOCKS=model.BLOCKS, LABELS=f.LABELS, EMB=f.EMB,
             train_native=f.train_native, native_prediction=f.native_prediction, fit_addition=model.fit_addition)
runner.prepare, runner.guard, runner.parity, runner.report = prepare, guard, parity, report

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "independent_verify":
        independent_verify()
    else:
        runner.main()
