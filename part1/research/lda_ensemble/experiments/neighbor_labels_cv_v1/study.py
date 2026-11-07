"""Focused eight-repeat test of independent multi-label neighbor votes.

Native pseudo-labeling and logistic regression remain fixed. Only final neighbor
votes change. Historical files are imported read-only and all output stays here.
"""
import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(variable, "1")
from pathlib import Path
import sys
import datetime
from types import SimpleNamespace
import numpy as np
from sklearn.preprocessing import normalize

OUT = Path(__file__).resolve().parent
PRIOR = OUT.parent / "feature_gap_cv_v1"
sys.path.insert(0, str(PRIOR))
import run as runner
import features as f
BASE = runner.read(PRIOR / "protocol.json")
OLD_REPORT = runner.report
SETTINGS = {
    "augmented_binary10": dict(gold_only=False, binary=True, k=10),
    "gold_fractional10": dict(gold_only=True, binary=False, k=10),
    "gold_binary10": dict(gold_only=True, binary=True, k=10),
    "gold_binary3": dict(gold_only=True, binary=True, k=3),
}
BLOCKS = {key + "_" + scope: f"{key}, {scope} score correction"
          for key in SETTINGS for scope in ["all", "other"]}


def vote(x, y, q, k, binary):
    sim = normalize(q) @ normalize(x).T
    indices = np.argsort(-sim, axis=1)[:, :min(k, len(x))]
    weights = np.take_along_axis(sim, indices, axis=1).clip(0) ** 2
    targets = y if binary else y / y.sum(axis=1, keepdims=True)
    return (weights[:, :, None] * targets[indices]).sum(axis=1) / (weights.sum(axis=1, keepdims=True) + 1e-9)


def fit_addition(packet, fit, query, name):
    key, scope = name.rsplit("_", 1)
    config = SETTINGS[key]
    zq = packet["scaler"].transform(query[f.EMB].to_numpy())
    rows = packet["gold"] if config["gold_only"] else packet["rows"]
    targets = packet["y"] if config["gold_only"] else packet["targets"]
    neighbors = vote(packet["z"][rows], targets, zq, config["k"], config["binary"])
    model = packet["base"][0]
    lr = np.zeros((len(query), 5))
    lr[:, model.classes_] = model.predict_proba(zq)
    new = f.centered(.7 * lr + .3 * neighbors)
    if scope == "other":
        original = f.native_prediction(packet, query)
        original[:, 4] = new[:, 4]
        new = original
    audit = dict(block=name, neighbor_rows=len(rows), independent_labels=config["binary"],
                 gold_only=config["gold_only"], k=config["k"], scope=scope,
                 final_LR_unchanged=True, pseudo_label_generation_unchanged=True)
    return new, f.centered(neighbors), audit


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
    for path in [Path(__file__), PRIOR / "protocol.json", OUT.parent / "other_transfer_cv_v1/confirmation.json"]:
        sources[str(path.relative_to(runner.PART1))] = runner.sha(path)
    rows = [dict(name=parent + "__" + block, parent=parent, block=block, description=description)
            for block, description in BLOCKS.items() for parent in runner.CONTROL_NAMES]
    runner.save(OUT / "registry.json", rows)
    p = dict(BASE)
    p.update(version="neighbor-labels-cv-v1", frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             source_sha256=sources, registry_sha256=runner.sha(OUT / "registry.json"),
             hypotheses=BLOCKS, candidates=len(rows),
             fixed_recipe="Final LR, GMM, two pseudo-label rounds, scalers, ensemble weights and decoders unchanged. Replace only final kNN with augmented binary votes, gold-only fractional votes, gold-only binary votes at k10 or k3. Cosine similarity squared, native weight .3. Apply to all reason scores or Other only.",
             initial="Focused comparison predeclared on all eight frozen repeats in BOTH pools. First three retained separately only for audit; no screening-based deletion.",
             shortlist="All sixteen fixed recipes advance to five additional saved repeats regardless of initial score. This is the policy's predeclared all-eight-repeat option.",
             metadata_diagnostic="Diagnostics are centered neighbor votes, not metadata-only probabilities. No clinical metadata or Part 2 input is introduced.",
             control_reuse="Native Ani refit must match original saved scores on every fold. Native three-way remains the parent for ensemble changes. Report all8 paired gains and wins versus both controls.",
             limitations="Focused representation test on reused development labels. Binary votes can increase false positives; per-label and persistent-error diagnostics remain required.")
    runner.save(OUT / "protocol.json", p)
    print("Frozen four neighbor representations x two scopes x two parents, all eight repeats.", flush=True)


def parity():
    guard()
    # A multi-label neighbor contributes one vote to each positive reason.
    y = np.array([[0, 0, 0, 1, 1]])
    x = np.array([[1., 0.]])
    independent = vote(x, y, x, 1, True)
    fractional = vote(x, y, x, 1, False)
    np.testing.assert_allclose(independent[:, 3:], 1, atol=1e-8)
    np.testing.assert_allclose(fractional[:, 3:], .5, atol=1e-8)
    frame = runner.frame_for("pooled")
    p, pos, groups, masked = runner.plan("pooled3", frame)
    fit, query, held = runner.fold_inputs(frame, pos, groups, masked, p["folds"][0])
    packet = f.train_native(fit)
    original = f.native_prediction(packet, query)
    np.testing.assert_allclose(original, runner.controls("pooled3", frame.iloc[pos])["ani"][0, held], atol=1e-10, rtol=0)
    zq = packet["scaler"].transform(query[f.EMB].to_numpy())
    old_knn = vote(packet["z"][packet["rows"]], packet["targets"], zq, 10, False)
    model = packet["base"][0]
    lr = np.zeros_like(old_knn)
    lr[:, model.classes_] = model.predict_proba(zq)
    np.testing.assert_allclose(f.centered(.7*lr + .3*old_knn), original, atol=1e-10, rtol=0)
    for name in BLOCKS:
        p1, _, _ = fit_addition(packet, fit, query, name)
        p2, _, _ = fit_addition(packet, fit, query.iloc[:1], name)
        np.testing.assert_allclose(p1[:1], p2, atol=1e-10, rtol=0)
        if name.endswith("_other"):
            np.testing.assert_array_equal(p1[:, :4], original[:, :4])
    runner.save(OUT / "parity.json", dict(passed=True, native_control_parity=True,
                 binary_label_votes_verified=True, query_batch_invariance=True, other_scope_boundary=True))


def shortlist():
    guard()
    rows = runner.comparisons()
    runner.save(OUT / "screen_comparison.json", rows)
    runner.pd.DataFrame(rows).to_csv(OUT / "screen_comparison.csv", index=False)
    names = [r["name"] for r in runner.read(OUT / "registry.json")]
    runner.save(OUT / "shortlist.json", dict(names=names, diagnostic_only=False,
                 selection_rule="All recipes, declared before scores; no screening-based pruning"))


def report():
    OLD_REPORT()
    path = OUT / "report.html"
    body = path.read_text().replace("Part 1 feature gap experiments", "Independent multi-label neighbor votes")
    body = body.replace("Part 1: new feature structure and earlier-feature retries", "Verified neighbors and independent reason votes")
    start, stop = body.index("<p>Nine blocks"), body.index("</p>", body.index("<p>Nine blocks")) + 4
    body = body[:start] + '<p>Four final-neighbor representations, applied to all reasons or Other only, evaluated with both original parents. Every recipe was predeclared for all eight repeats. Native pseudo-label generation and final logistic regression remain unchanged; the comparison changes only final neighbor information. Only Part 1 data are used.</p>' + body[stop:]
    body = body.replace("Metadata-only diagnostics", "Neighbor-only diagnostics")
    path.write_text(body)
    (OUT / "metadata_only.csv").rename(OUT / "neighbor_diagnostics.csv")


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
