"""Recruitment-conditional Other specialists versus global Other heads."""
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
    for path in [Path(__file__), OUT/"heads.py", REVISION/"protocol.json", REVISION/"confirmation.json", REVISION/"summary.json"]:
        sources[str(path.relative_to(runner.PART1))] = runner.sha(path)
    registry = [dict(name=parent+"__"+name, parent=parent, block=name, description=description)
                for name, description in BLOCKS.items() for parent in runner.CONTROL_NAMES]
    runner.save(OUT/"registry.json", registry)
    p = dict(BASE)
    p.update(version="conditional-other-cv-v1", frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sources, registry_sha256=runner.sha(OUT/"registry.json"), candidates=len(registry), hypotheses=BLOCKS,
        parent="Latest inductive Ani LDA revision and its ensemble. Original Ani/ensemble retained as additional references.",
        fixed_recipe="Freeze latest LDA source and ensemble. Global binary shrinkage LDA predicts Other from gold only, with empirical or uniform priors; average its probability halfway into source Other decision score. Conditional LDA and C.1 LR fit only gold Recruitment rows; compare empirical and balanced priors. On source-predicted Recruitment only, raise Other to conditional score if larger; empirical models use cutoff.5, balanced models use conservative cutoff.8. First four source scores stay identical. Zero/one-class training fallback is the constant observed label (zero for empty); no pseudo-labels train a head.",
        hypothesis_detail="Existing eight-repeat pooled predictions detect Other in 71.15% of 26 Other-only rows but 4.17% of nine Recruitment+Other rows. Conditional training separates this rare second-reason task from the dominant standalone-Other cases. Cohort definitions are retrospective diagnostics; no held-out truth selects query gates.",
        initial="All twelve candidates run all eight repeats, predeclared. Matched revision_full controls always run in both pools.",
        shortlist="All twelve candidates; no pruning after initial scores.",
        confirmation="Positive paired means and >=5/8 wins in BOTH pools against corresponding latest revision control, not merely the older original parent.",
        supplementary="At most three confirmed candidates, or at most two diagnostics if none confirm, selected after all-eight comparison against latest revision.",
        metadata_diagnostic="Original first-four scores with independent head probability in Other; diagnostic only, not metadata-only.")
    runner.save(OUT/"protocol.json", p)
    print("Frozen twelve global/conditional Other-head comparisons against latest LDA controls.", flush=True)


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
        expected = packet["gold"][packet["y"][:, 3] == 1] if name.startswith("conditional") else packet["gold"]
        assert set(audit["fitting_ids"]) == set(fit.iloc[expected].ID)
        np.testing.assert_array_equal(full[:, :4], revised[:, :4])
        if name.startswith("conditional"):
            assert (full[:, 4] >= revised[:, 4]).all()
            np.testing.assert_array_equal(full[revised[:, 3] < .5], revised[revised[:, 3] < .5])
    runner.save(OUT/"parity.json", dict(passed=True, native_control_parity=True, latest_revision_control_parity=True,
        query_batch_invariance=True, conditional_gold_training_verified=True, query_gate_uses_predictions_only=True, head_training_boundaries_verified=True))


def shortlist():
    rows = comparisons()
    runner.save(OUT/"screen_comparison.json", rows)
    runner.pd.DataFrame(rows).to_csv(OUT/"screen_comparison.csv", index=False)
    runner.save(OUT/"shortlist.json", dict(names=[r["name"] for r in runner.read(OUT/"registry.json")],
        diagnostic_only=False, selection_rule="All candidates, predeclared before scoring"))


def report():
    ORIGINAL_REPORT()
    path = OUT/"report.html"
    body = path.read_text().replace("Part 1 feature gap experiments", "Conditional Other specialists")
    body = body.replace("Part 1: new feature structure and earlier-feature retries", "Recovering Other alongside Recruitment")
    start = body.index("<p>Nine blocks")
    stop = body.index("</p>", start)+4
    body = body[:start]+'<p>Compare global binary Other LDA with LDA/LR specialists trained only on gold Recruitment cases. Conditional heads can only raise Other on source-predicted Recruitment, using fixed empirical/balanced cutoffs. The first four source scores stay fixed. Confirmation is measured against the latest LDA revision, whose matched control runs in every fold. Original Ani/ensemble scores are additional references. Only supplied Part 1 inputs are used.</p>'+body[stop:]
    body = body.replace("Metadata-only diagnostics", "Binary Other-head diagnostics")
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
