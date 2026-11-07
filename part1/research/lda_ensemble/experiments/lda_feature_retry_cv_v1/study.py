"""Retry promising metadata with the latest LDA source."""
import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(variable, "1")
from pathlib import Path
import sys
import datetime
import importlib.util
from types import SimpleNamespace
import numpy as np

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
import heads
BLOCKS = {name: str(settings) for name, settings in heads.SETTINGS.items()}


def fit_addition(packet, fit, query, name):
    if name == "revision_full":
        return source.fit_addition(packet, fit, query, name)
    return heads.fit_predict(packet, fit, query, name, source, f)


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
    for path in [Path(__file__), OUT/"heads.py", REVISION/"protocol.json", REVISION/"confirmation.json", REVISION/"summary.json", OUT.parent/"biochemical_features_cv_v1/representations.py"]:
        sources[str(path.relative_to(runner.PART1))] = runner.sha(path)
    registry = [dict(name=parent+"__"+name, parent=parent, block=name, description=description)
                for name, description in BLOCKS.items() for parent in runner.CONTROL_NAMES]
    runner.save(OUT/"registry.json", registry)
    p = dict(BASE)
    p.update(version="lda-feature-retry-cv-v1", frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sources, registry_sha256=runner.sha(OUT/"registry.json"), candidates=len(registry), hypotheses=BLOCKS,
        parent="Latest inductive Ani LDA revision and its ensemble. Original Ani/ensemble retained as additional references.",
        fixed_recipe="Freeze latest revision LDA, neighbors and pseudo-labels. Retest protein-target indicators and Morgan fingerprints in the final C.1 logistic head. Compare augmented fitting against half-replacement by gold-only fitting; include no-feature controls for both. The original source logistic weight is .5 and ensemble source weight is .7. Metadata RMS weight.3, same previous parsers and fingerprint settings. Fingerprint columns use union of top60 positive chi-square values per label on fitting gold only; normalization and missingness match the preceding biochemical study.",
        hypothesis_detail="Protein targets and molecular fingerprints were the most promising metadata corrections on the older source. Their contribution may change with LDA, stronger regularization and revised pseudo-labels. Isolate feature effects from gold-only fitting using matching no-feature heads.",
        initial="All twelve candidates run all eight repeats, predeclared. Matched revision_full controls always run in both pools.",
        shortlist="All twelve candidates; no pruning after initial scores.",
        confirmation="Positive paired means and >=5/8 wins in BOTH pools against corresponding latest revision control, not merely the older original parent.",
        supplementary="At most three confirmed candidates, or at most two diagnostics if none confirm, selected after all-eight-repeat comparison against latest revision.",
        metadata_diagnostic="Final logistic-head scores including embeddings; diagnostic only, not metadata-only.")
    runner.save(OUT/"protocol.json", p)
    print("Frozen twelve metadata retries and matched controls against latest LDA revision.", flush=True)


def run_stage(stage, blocks):
    return ORIGINAL_STAGE(stage, list(dict.fromkeys([*blocks, "revision_full", "augmented_none", "goldhalf_none"])))


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
    for row in rows:
        mode, view = row["block"].split("_", 1)
        matched = row["parent"]+"__"+mode+"_none"
        row["matched_feature_control"] = matched
        for pool in ["train", "pooled"]:
            values = []
            for stage in [pool+"3", pool+"5"] if extra else [pool+"3"]:
                entry = next(r for r in runner.read(OUT/stage/"index.json") if r["candidate"]==matched)
                values.extend(r["sample_f1"] for r in entry["repeat_scores"])
            delta = np.array(row[pool+"_repeats"])-values
            row[pool+"_vs_matched_feature_delta"] = float(delta.mean())
            row[pool+"_vs_matched_feature_wins"] = int((delta>1e-12).sum())
        row["metadata_contribution_confirmed"] = bool(view!="none" and all(row[pool+"_vs_matched_feature_delta"]>1e-12 and row[pool+"_vs_matched_feature_wins"] >= (5 if extra else 2) for pool in ["train", "pooled"]))
    return sorted(rows, key=lambda r: (-r["minimum_parent_gain"], r["candidate"]))


def parity():
    guard()
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
        expected = packet["gold"] if name.startswith("goldhalf_") else source.fitted_variant(packet, source.SETTINGS["revision_full"])["rows"]
        assert set(audit["fitting_ids"]) == set(fit.iloc[expected].ID)
        if name == "augmented_none":
            np.testing.assert_allclose(full, revised, atol=1e-10, rtol=0)
        if not name.endswith("_none"):
            assert set(audit["vocabulary_fit_ids"]) == set(fit.ID)
            assert not set(audit["vocabulary_fit_ids"]) & set(query.ID)
            if name.endswith("_morgan"):
                assert set(audit["feature_selection_gold_ids"]) == set(fit.iloc[packet["gold"]].ID)
    runner.save(OUT/"parity.json", dict(passed=True, native_control_parity=True, latest_revision_control_parity=True,
        query_batch_invariance=True, augmented_no_feature_control_matches_revision=True,
        fitting_boundaries_verified=True, feature_selection_fitting_gold_only=True))


def shortlist():
    rows = comparisons()
    runner.save(OUT/"screen_comparison.json", rows)
    runner.pd.DataFrame(rows).to_csv(OUT/"screen_comparison.csv", index=False)
    runner.save(OUT/"shortlist.json", dict(names=[r["name"] for r in runner.read(OUT/"registry.json")],
        diagnostic_only=False, selection_rule="All candidates, predeclared before scoring"))


def report():
    ORIGINAL_REPORT()
    path = OUT/"report.html"
    body = path.read_text().replace("Part 1 feature gap experiments", "Promising feature retries with latest LDA")
    body = body.replace("Part 1: new feature structure and earlier-feature retries", "Retesting targets and molecular fingerprints with the stronger source")
    start = body.index("<p>Nine blocks")
    stop = body.index("</p>", start)+4
    body = body[:start]+'<p>Protein-target and molecular-fingerprint corrections with latest C.1 source classifier and latest pseudo-labels. Retest augmented fitting and half gold-only replacement, with matching no-feature controls. LDA and neighbors remain fixed. Confirmation is against the latest LDA revision; metadata-specific effects use the matching integration control. See final_comparison.csv for both paired contrasts. Only supplied Part 1 fields are used.</p>'+body[stop:]
    body = body.replace("Metadata-only diagnostics", "Final logistic-head diagnostics")
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
