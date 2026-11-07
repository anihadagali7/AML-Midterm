"""Fold-local biochemical representations from supplied Part 1 fields only."""
import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(variable, "1")
from pathlib import Path
import sys
import datetime
import hashlib
import json
import ast
from types import SimpleNamespace
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.preprocessing import normalize

OUT = Path(__file__).resolve().parent
PRIOR = OUT.parent / "feature_gap_cv_v1"
sys.path.insert(0, str(PRIOR))
import run as runner
import features as f
BASE = runner.read(PRIOR / "protocol.json")
ORIGINAL_REPORT, ORIGINAL_COMPARE, ORIGINAL_STAGE = runner.report, runner.comparisons, runner.run_stage
from representations import BioBlock, catalog_from, fingerprint, values
CATALOG = catalog_from(runner.PART1 / "data/pathway_catalog.csv")
VIEWS = ["none", "actions", "pathway_all", "pathway_specific", "morgan"]
BLOCKS = {mode + "_" + view: f"{mode} final LR with fold-selected biochemical {view}"
          for mode in ["augmented", "goldhalf"] for view in VIEWS}


def select_columns(x, y):
    selected = set()
    for j in range(y.shape[1]):
        if len(np.unique(y[:, j])) < 2:
            continue
        values = np.nan_to_num(chi2(x, y[:, j])[0], nan=0., posinf=0., neginf=0.)
        order = np.argsort(-values, kind="stable")
        selected.update(int(k) for k in order[:60] if values[k] > 0)
    return np.array(sorted(selected), dtype=int)


def representation(packet, fit, query, view):
    cache = packet.setdefault("selected_biochemistry", {})
    if view not in cache:
        transform = BioBlock(view, CATALOG).fit(fit)
        x, missing, invalid = transform.transform(fit)
        selected = select_columns(x[packet["gold"]], packet["y"])
        a = np.column_stack([normalize(x[:, selected]).toarray(), missing])
        cache[view] = dict(transform=transform, selected=selected, a=a,
            names=[transform.names[k] for k in selected]+["__missing__"],
            vocabulary_size=len(transform.names), selection_ids=fit.iloc[packet["gold"]].ID.tolist(),
            fitting_ids=fit.ID.tolist(), invalid_fitting_rows=invalid)
    saved = cache[view]
    x, missing, _ = saved["transform"].transform(query)
    q = np.column_stack([normalize(x[:, saved["selected"]]).toarray(), missing])
    return saved["a"], q, saved


def fit_addition(packet, fit, query, name):
    mode, view = name.split("_", 1)
    zq = packet["scaler"].transform(query[f.EMB].to_numpy())
    multiplier, details = 0., {}
    if view == "none":
        x, q = packet["z"], zq
    else:
        a, b, saved = representation(packet, fit, query, view)
        multiplier = .3*np.sqrt(np.mean(np.sum(packet["z"]**2, axis=1)))/max(np.sqrt(np.mean(np.sum(a*a, axis=1))), 1e-12)
        x, q = np.column_stack([packet["z"], a*multiplier]), np.column_stack([zq, b*multiplier])
        details = dict(feature_names_sha256=hashlib.sha256(json.dumps(saved["names"]).encode()).hexdigest(),
            vocabulary_size=saved["vocabulary_size"], selected_features=len(saved["selected"]),
            selection_gold_ids=saved["selection_ids"], vocabulary_fit_ids=saved["fitting_ids"])
    rows = packet["rows"] if mode == "augmented" else packet["gold"]
    targets = packet["targets"] if mode == "augmented" else packet["y"]
    model = f.native.fit_ani_base(x[rows], targets)[0]
    lr = np.zeros((len(query), 5))
    lr[:, model.classes_] = model.predict_proba(q)
    native_model = packet["base"][0]
    original_lr = np.zeros_like(lr)
    original_lr[:, native_model.classes_] = native_model.predict_proba(zq)
    raw = f.native.ani_base_predict(packet["base"], zq)
    blend = 1. if mode == "augmented" else .5
    result = f.centered(np.clip(raw + .7*blend*(lr-original_lr), 0, 1))
    return result, f.centered(lr), dict(block=name, dimensions=x.shape[1], multiplier=float(multiplier),
        integration=mode, replacement_fraction=blend, selection_uses_gold_only=True,
        model_iterations=int(model.n_iter_.max()), **details)


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
    for path in [Path(__file__), PRIOR / "protocol.json", OUT / "representations.py", OUT.parent / "gold_correction_cv_v1/confirmation.json"]:
        sources[str(path.relative_to(runner.PART1))] = runner.sha(path)
    rows = [dict(name=parent+"__"+block, parent=parent, block=block, description=description)
            for block, description in BLOCKS.items() for parent in runner.CONTROL_NAMES]
    runner.save(OUT / "registry.json", rows)
    p = dict(BASE)
    p.update(version="biochemical-features-cv-v1", frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sources, registry_sha256=runner.sha(OUT / "registry.json"), candidates=len(rows), hypotheses=BLOCKS,
        fixed_recipe="Freeze native pseudo-labels and kNN. Add one selected text block at .3 embedding RMS to final LR: augmented C.3 head or half gold-only C.3 head, with native class weights. Include matched no-feature controls. Original ensemble correction remains .7 times source change.",
        feature_details="Actions and pathway ID multi-hot blocks retain fitting count>=3. Compare all supplied pathways with a distinct block retaining catalog levels>=3; no hierarchy edges inferred. Morgan radius2 chirality-aware2048-bit fingerprints aggregate mean and any over valid distinct supplied SMILES; append log molecule count and invalid flag. For each block, select union of top60 positive chi-square features per reason using fitting gold only, L2 normalize and append missingness. RDKit parsing is deterministic; invalid structures retained as indicators.",
        hypothesis_detail="Retry action/pathway indicators and test finer pathway granularity and molecular substructure fingerprints, which the previous descriptor-only chemistry screen omitted. One field per variant; no additional records or chemical database used.",
        metadata_diagnostic="Diagnostic is the final multinomial LR head, including embeddings, centered at native .3 cutoff. It is not metadata-only.",
        controls="Original parents plus no-feature heads for each integration. Both no-feature controls always run. Report matched feature contribution separately and require corresponding paired wins before claiming metadata-specific confirmation.")
    runner.save(OUT / "protocol.json", p)
    print("Frozen four biochemical representations x two integrations x two parents, plus four matched controls.", flush=True)


def run_stage(stage, blocks):
    return ORIGINAL_STAGE(stage, list(dict.fromkeys([*blocks, "augmented_none", "goldhalf_none"])))


def comparisons(extra=False):
    rows = ORIGINAL_COMPARE(extra)
    for row in rows:
        mode, view = row["block"].split("_", 1)
        matched = row["parent"]+"__"+mode+"_none"
        for pool in ["train", "pooled"]:
            values = []
            for stage in [pool+"3", pool+"5"] if extra else [pool+"3"]:
                index = runner.read(OUT / stage / "index.json")
                entry = next(r for r in index if r["candidate"] == matched)
                values.extend(r["sample_f1"] for r in entry["repeat_scores"])
            delta = np.array(row[pool+"_repeats"])-values
            row[pool+"_vs_matched_delta"] = float(delta.mean())
            row[pool+"_vs_matched_wins"] = int((delta>1e-12).sum())
        row["matched_control"] = matched
        row["metadata_contribution_confirmed"] = bool(view != "none" and all(
            row[pool+"_vs_matched_delta"]>1e-12 and row[pool+"_vs_matched_wins"] >= (5 if extra else 2)
            for pool in ["train", "pooled"]))
    return rows


def parity():
    guard()
    import pandas as pd
    np.testing.assert_array_equal(fingerprint("CCO"), fingerprint("OCC"))
    assert fingerprint("not-a-valid-molecule") is None
    assert values("inhibitor;agonist;inhibitor") == ["agonist", "inhibitor"]
    mock = BioBlock("pathway_specific", {"a": 2, "b": 3})
    assert mock.tokens("a;b;unknown") == ["b"]
    from scipy.sparse import csr_matrix
    np.testing.assert_array_equal(select_columns(csr_matrix([[1, 0], [1, 1], [1, 0], [1, 1]]),
        np.array([[0], [1], [0], [1]])), [1])
    frame = runner.frame_for("pooled")
    p, positions, groups, masked = runner.plan("pooled3", frame)
    fit, query, h = runner.fold_inputs(frame, positions, groups, masked, p["folds"][0])
    packet = f.train_native(fit)
    old = f.native_prediction(packet, query)
    np.testing.assert_allclose(old, runner.controls("pooled3", frame.iloc[positions])["ani"][0, h], atol=1e-10, rtol=0)
    for name in BLOCKS:
        full, _, audit = fit_addition(packet, fit, query, name)
        first, _, audit1 = fit_addition(packet, fit, query.iloc[:1], name)
        np.testing.assert_allclose(full[:1], first, atol=1e-10, rtol=0)
        if name == "augmented_none":
            np.testing.assert_allclose(full, old, atol=1e-10, rtol=0)
        if not name.endswith("_none"):
            assert set(audit["selection_gold_ids"]) == set(fit.iloc[packet["gold"]].ID)
            assert not set(query.ID) & set(audit["vocabulary_fit_ids"])
            assert audit["feature_names_sha256"] == audit1["feature_names_sha256"]
    for view in VIEWS[1:]:
        saved = packet["selected_biochemistry"][view]
        original = saved["a"].copy()
        changed = query.copy()
        for column in ["drug_smiles", "drug_actions", "drug_pathways"]:
            changed[column] = ""
        representation(packet, fit, changed, view)
        np.testing.assert_array_equal(original, saved["a"])
    runner.save(OUT / "parity.json", dict(passed=True, native_control_parity=True,
        augmented_no_feature_parity=True, query_batch_invariance=True, supervised_selection_gold_only=True,
        synthetic_chi2_ranking_verified=True, query_transform_does_not_change_fitted_vocabulary=True, molecule_and_catalog_checks=True))


def report():
    ORIGINAL_REPORT()
    path = OUT / "report.html"
    body = path.read_text().replace("Part 1 feature gap experiments", "Biochemical feature retries")
    body = body.replace("Part 1: new feature structure and earlier-feature retries", "Testing supplied drug actions, pathways and molecular substructures")
    start = body.index("<p>Nine blocks")
    stop = body.index("</p>", start)+4
    body = body[:start]+'<p>Action indicators, all/specific pathway indicators and Morgan molecular fingerprints are selected using fitting gold labels only. Each block modifies the augmented or half gold-only final classifier, with matched no-feature controls. All preprocessing and selection are fold-local. Only supplied Part 1 inputs are used.</p>'+body[stop:]
    body = body.replace("Metadata-only diagnostics", "Final classifier diagnostics (embeddings included)")
    path.write_text(body)
    (OUT / "metadata_only.csv").rename(OUT / "head_diagnostics.csv")


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
