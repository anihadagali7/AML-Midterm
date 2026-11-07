"""Test metadata as neighbor-retrieval features, keeping the logistic head fixed."""
import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(variable, "1")
from pathlib import Path
import sys
import datetime
import hashlib
import json
import importlib.util
from types import SimpleNamespace
import numpy as np
from sklearn.preprocessing import normalize

OUT = Path(__file__).resolve().parent
PRIOR = OUT.parent / "feature_gap_cv_v1"
sys.path.insert(0, str(PRIOR))
import run as runner
import features as f
CANONICAL = {name: getattr(runner, name) for name in ["OUT", "m", "prepare", "guard", "parity", "run_stage", "comparisons", "report"]}


def load_helpers(name, folder):
    spec = importlib.util.spec_from_file_location(name, OUT.parent / folder / "study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


text_features = load_helpers("selected_text_helpers", "selected_text_cv_v2")
sys.path.insert(0, str(OUT.parent / "biochemical_features_cv_v1"))
bio_features = load_helpers("biochemical_helpers", "biochemical_features_cv_v1")
for name, value in CANONICAL.items():
    setattr(runner, name, value)
BASE = runner.read(PRIOR / "protocol.json")
ORIGINAL_REPORT, ORIGINAL_COMPARE, ORIGINAL_STAGE = runner.report, runner.comparisons, runner.run_stage
BLOCKS = {mode+"_"+view: f"{mode} nearest neighbors with {view}"
          for mode in ["augmented", "goldbinary"]
          for view in ["none", "criteria_words", "drug_chars", "morgan", "drug_targets"]}


def fit_addition(packet, fit, query, name):
    mode, view = name.split("_", 1)
    zq = packet["scaler"].transform(query[f.EMB].to_numpy())
    details = {}
    if view == "none":
        x, q, multiplier = packet["z"], zq, 0.
    else:
        if view == "morgan":
            a, b, saved = bio_features.representation(packet, fit, query, view)
        elif view == "drug_targets":
            if "retrieval_targets" not in packet:
                block = f.Block(view).fit(fit)
                packet["retrieval_targets"] = (block, block.transform(fit))
            block, a = packet["retrieval_targets"]
            b = block.transform(query)
            saved = dict(names=block.feature_names, fitting_ids=block.fitting_ids,
                         selection_ids=[], vocabulary_size=len(block.feature_names))
        else:
            a, b, saved = text_features.representation(packet, fit, query, view)
        multiplier = .3*np.sqrt(np.mean(np.sum(packet["z"]**2, axis=1)))/max(np.sqrt(np.mean(np.sum(a*a, axis=1))), 1e-12)
        x, q = np.column_stack([packet["z"], a*multiplier]), np.column_stack([zq, b*multiplier])
        details = dict(feature_names_sha256=hashlib.sha256(json.dumps(saved["names"]).encode()).hexdigest(),
            vocabulary_size=saved["vocabulary_size"], fitting_ids=saved["fitting_ids"],
            selection_gold_ids=saved["selection_ids"])
    rows = packet["rows"] if mode == "augmented" else packet["gold"]
    targets = packet["targets"].astype(float) if mode == "augmented" else packet["y"].astype(float)
    if mode == "augmented":
        targets = targets/targets.sum(axis=1, keepdims=True)
    similarity = normalize(q) @ normalize(x[rows]).T
    nearest = np.argsort(-similarity, axis=1)[:, :10]
    weights = np.take_along_axis(similarity, nearest, axis=1).clip(0)**2
    knn = (weights[:, :, None]*targets[nearest]).sum(axis=1)/(weights.sum(axis=1, keepdims=True)+1e-9)
    model = packet["base"][0]
    lr = np.zeros((len(query), 5))
    lr[:, model.classes_] = model.predict_proba(zq)
    new = f.centered(.7*lr+.3*knn)
    native_neighbors = np.argsort(-(normalize(zq) @ normalize(packet["z"][rows]).T), axis=1)[:, :10]
    changed = [len(set(a)-set(b)) for a, b in zip(nearest, native_neighbors)]
    return new, f.centered(knn), dict(block=name, dimensions=x.shape[1], multiplier=float(multiplier),
        neighbor_pool=mode, original_logistic_head_unchanged=True, mean_replaced_neighbors=float(np.mean(changed)),
        retrieved_fit_ids=[fit.iloc[rows[a]].ID.tolist() for a in nearest], **details)


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
    dependencies = [Path(__file__), PRIOR / "protocol.json"]
    for folder in ["selected_text_cv_v2", "biochemical_features_cv_v1"]:
        dependencies.extend(OUT.parent / folder / name for name in ["study.py", "protocol.json", "confirmation.json"])
    dependencies.append(OUT.parent / "biochemical_features_cv_v1/representations.py")
    for path in dependencies:
        sources[str(path.relative_to(runner.PART1))] = runner.sha(path)
    rows = [dict(name=parent+"__"+block, parent=parent, block=block, description=description)
            for block, description in BLOCKS.items() for parent in runner.CONTROL_NAMES]
    runner.save(OUT / "registry.json", rows)
    p = dict(BASE)
    p.update(version="metadata-neighbors-cv-v1", frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sources, registry_sha256=runner.sha(OUT / "registry.json"), candidates=len(rows), hypotheses=BLOCKS,
        fixed_recipe="Keep original LR, pseudo-label generation and decoder. Add one metadata block at .3 embedding RMS only to cosine neighbor retrieval; retain k10 and squared-positive-cosine weights. Compare augmented fractional targets and gold-only binary targets, each with no-feature matched control. Native final .7LR+.3kNN and original ensemble combination unchanged.",
        feature_details="Reuse frozen fold-local criteria word selection, raw-drug character selection, molecular fingerprint selection, and target indicator encodings from their linked studies. All transformations refit inside the permitted outer fitting pool; no saved outer scores used as features.",
        hypothesis_detail="Metadata may change which labeled examples are retrieved even when the logistic head does not use it effectively. Evaluate this distinct integration route with matched embedding-only neighbor controls.",
        metadata_diagnostic="Diagnostic is new neighbor-vote score centered at .3, not a metadata-only classifier.",
        controls="Original parents and both matched no-feature retrieval variants always run. Feature-specific confirmation also requires >=5/8 paired wins and positive mean in both pools versus matching retrieval control.")
    runner.save(OUT / "protocol.json", p)
    print("Frozen four retrieval features x two neighbor pools x two parents plus four controls.", flush=True)


def run_stage(stage, blocks):
    return ORIGINAL_STAGE(stage, list(dict.fromkeys([*blocks, "augmented_none", "goldbinary_none"])))


def comparisons(extra=False):
    rows = ORIGINAL_COMPARE(extra)
    for row in rows:
        mode, view = row["block"].split("_", 1)
        control = row["parent"]+"__"+mode+"_none"
        for pool in ["train", "pooled"]:
            values = []
            for stage in [pool+"3", pool+"5"] if extra else [pool+"3"]:
                entry = next(r for r in runner.read(OUT / stage / "index.json") if r["candidate"] == control)
                values.extend(r["sample_f1"] for r in entry["repeat_scores"])
            delta = np.array(row[pool+"_repeats"])-values
            row[pool+"_vs_matched_delta"] = float(delta.mean())
            row[pool+"_vs_matched_wins"] = int((delta>1e-12).sum())
        row["matched_control"] = control
        row["metadata_contribution_confirmed"] = bool(view != "none" and all(
            row[pool+"_vs_matched_delta"]>1e-12 and row[pool+"_vs_matched_wins"] >= (5 if extra else 2)
            for pool in ["train", "pooled"]))
    return rows


def parity():
    guard()
    frame = runner.frame_for("pooled")
    p, positions, groups, masked = runner.plan("pooled3", frame)
    fit, query, h = runner.fold_inputs(frame, positions, groups, masked, p["folds"][0])
    packet = f.train_native(fit)
    old = f.native_prediction(packet, query)
    np.testing.assert_allclose(old, runner.controls("pooled3", frame.iloc[positions])["ani"][0, h], atol=1e-10, rtol=0)
    for name in BLOCKS:
        full, _, audit = fit_addition(packet, fit, query, name)
        first, _, single = fit_addition(packet, fit, query.iloc[:1], name)
        np.testing.assert_allclose(full[:1], first, atol=1e-10, rtol=0)
        assert audit["retrieved_fit_ids"][:1] == single["retrieved_fit_ids"]
        assert not set(query.ID) & {s for row in audit["retrieved_fit_ids"] for s in row}
        if name == "augmented_none":
            np.testing.assert_allclose(full, old, atol=1e-10, rtol=0)
        assert audit["original_logistic_head_unchanged"]
    runner.save(OUT / "parity.json", dict(passed=True, native_control_parity=True,
        no_feature_retrieval_parity=True, query_batch_invariance=True, retrieved_rows_fit_only=True,
        original_logistic_head_unchanged=True))


def report():
    ORIGINAL_REPORT()
    path = OUT / "report.html"
    body = path.read_text().replace("Part 1 feature gap experiments", "Metadata-assisted neighbor retrieval")
    body = body.replace("Part 1: new feature structure and earlier-feature retries", "Testing feature contributions through neighbor retrieval")
    start = body.index("<p>Nine blocks")
    stop = body.index("</p>", start)+4
    body = body[:start]+'<p>Criteria terms, raw-drug fragments, molecular fingerprints and protein targets each modify cosine neighbor retrieval. The logistic classifier and pseudo-labels stay fixed. Augmented fractional and gold-only binary neighbor pools each have matched embedding-only controls. Only supplied Part 1 inputs are used.</p>'+body[stop:]
    body = body.replace("Metadata-only diagnostics", "Neighbor-vote diagnostics")
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
runner.run_stage, runner.comparisons, runner.report = run_stage, comparisons, report
if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "independent_verify":
        independent_verify()
    else:
        runner.main()
