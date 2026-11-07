"""Nested decision learning for the submitted LDA ensemble."""
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
import decision
BLOCKS = decision.SETTINGS


def fit_addition(packet, fit, query, name):
    if name == "revision_full":
        return source.fit_addition(packet, fit, query, name)
    return decision.fit_predict(packet, fit, query, name, source, f)


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
    sources.update(runner.read(OUT.parent/"ani_direct_comparison_v1/protocol.json")["source_sha256"])
    for path in [Path(__file__), OUT/"decision.py", REVISION/"protocol.json", REVISION/"confirmation.json", REVISION/"summary.json", OUT.parent/"final_lda_ensemble_trainval235_v1/kaggle_result.json"]:
        sources[str(path.relative_to(runner.PART1))] = runner.sha(path)
    registry = [dict(name=parent+"__"+name, parent=parent, block=name, description=description)
                for name, description in BLOCKS.items() for parent in runner.CONTROL_NAMES]
    runner.save(OUT/"registry.json", registry)
    p = dict(BASE)
    p.update(version="nested-decision-lda-cv-v1", frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sources, registry_sha256=runner.sha(OUT/"registry.json"), candidates=len(registry), hypotheses=BLOCKS,
        parent="Latest inductive Ani LDA revision and its ensemble. Original Ani/ensemble retained as additional references.",
        fixed_recipe="Keep all three submitted model components unchanged; learn the source centering cutoff from three-fold grouped inner OOF predictions (seed 20261106), refitting the full inductive LDA pseudo-label pipeline and both ensemble partners in each inner fit with inner held-group masked purges. Four recipes: shared cutoff or five label cutoffs, each with equal or five-year-half-life gold recency weighting (floor .1). Maximize inner ensemble sample-F1 minus .01 times squared cutoff displacement/.1 per free parameter. Fixed candidate grid .3,.25,.35,.2,.4; shared one pass, label-specific two ordered coordinate passes. Change only on strict objective improvement, preserving incumbent on ties. Gold targets and all final model fits unchanged. No actual test inputs.",
        hypothesis_detail="The temporal feature/expert studies did not improve later-year decisions. This bounded retry of decision learning uses the new LDA ensemble and inner-only supervision, unlike historical outer-validation threshold tuning. Source-only results are diagnostic because cutoffs optimize the ensemble inner objective.",
        initial="All eight parent-specific candidates run all eight repeats, predeclared; matched revision_full controls run every stage.",
        shortlist="All candidates; no pruning.",
        confirmation="Report ordinary CV confirmation separately: positive paired means and >=5/8 wins in both pools against latest LDA parent.",
        supplementary="After all-eight CV, evaluate ALL FOUR ensemble candidates plus current LDA control on validation and all three fixed chronological windows. This is a predeclared temporal diagnostic exception to the normal shortlist; source-only candidates do not receive supplementary fits.",
        temporal_eligibility="Ensemble only: chronological gain >=.005; strictly improve at least two of three windows; validation delta >=-.005; train and pooled CV deltas each >=-.003. This permits an explicit temporal tradeoff and does not relabel it as a CV win. No new settings chosen from these windows.",
        metadata_diagnostic="Duplicate source scores; not metadata-only.")
    runner.save(OUT/"protocol.json", p)
    print("Frozen four nested decision recipes and matched controls, with all-eight CV and predeclared later-year diagnostics.", flush=True)


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
        assert set(audit["fitting_ids"]) == set(fit.ID)
        assert np.isin(audit["thresholds"],decision.GRID).all()
        assert set(audit["inner_gold_ids"])==set(fit.iloc[packet["gold"]].ID)
        seen = []
        indexed = fit.set_index("ID")
        for inner in audit["inner_folds"]:
            assert set(inner["fit_ids"]) <= set(fit.ID)
            assert not set(inner["fit_ids"]) & set(inner["query_ids"])
            assert not set(inner["fit_ids"]) & set(query.ID)
            a = runner.pd.util.hash_pandas_object(indexed.loc[inner["fit_ids"],f.EMB].round(10),index=False)
            b = runner.pd.util.hash_pandas_object(indexed.loc[inner["query_ids"],f.EMB].round(10),index=False)
            assert not set(a)&set(b)
            seen.extend(inner["query_ids"])
        assert sorted(seen)==sorted(audit["inner_gold_ids"])
        np.testing.assert_allclose(decision.centered(np.linspace(.01,.99,50),.3),f.centered(np.linspace(.01,.99,50)),atol=1e-12,rtol=0)
    runner.save(OUT/"parity.json", dict(passed=True,native_control_parity=True,latest_revision_control_parity=True,
        query_batch_invariance=True,inner_only_cutoff_learning=True,inner_group_separation=True,
        complete_inner_oof_coverage=True,native_cutoff_transform_parity=True))


def shortlist():
    rows = comparisons()
    runner.save(OUT/"screen_comparison.json", rows)
    runner.pd.DataFrame(rows).to_csv(OUT/"screen_comparison.csv", index=False)
    runner.save(OUT/"shortlist.json", dict(names=[r["name"] for r in runner.read(OUT/"registry.json")],
        diagnostic_only=False, selection_rule="All candidates, predeclared before scoring"))


def confirm():
    for stage in ["train5", "pooled5"]:
        run_stage(stage, list(BLOCKS))
    rows = comparisons(extra=True)
    runner.save(OUT/"confirmation.json", dict(comparison=rows,
        supplementary_names=["ensemble__revision_full"]+["ensemble__"+block for block in BLOCKS],
        diagnostic_only=True, predeclared_all_ensemble_temporal_diagnostics=True))
    runner.pd.DataFrame(rows).to_csv(OUT/"confirmation.csv", index=False)


def report():
    ORIGINAL_REPORT()
    summary = runner.read(OUT/"summary.json")
    records = runner.read(OUT/"supplementary/results.json")
    control = {r["window"]:r["sample_f1"] for r in records if r["candidate"]=="ensemble__revision_full"}
    arrays = [np.load(OUT/"supplementary"/f"window{wi}_ensemble__revision_full.npz") for wi in [1,2,3]]
    baseline_chrono = runner.metrics(np.concatenate([a["truth"] for a in arrays]),np.concatenate([a["scores"] for a in arrays]))["sample_f1"]
    for row in summary["final_comparison"]:
        row["temporal_eligible"] = False
        if row["parent"]=="ensemble":
            windows = {r["window"]:r["sample_f1"] for r in records if r["candidate"]==row["candidate"]}
            row["chronological_delta"] = row["chronological"]-baseline_chrono
            row["validation_delta"] = row["validation"]-control[0]
            row["chronological_window_deltas"] = [windows[wi]-control[wi] for wi in [1,2,3]]
            row["chronological_window_wins"] = sum(v>1e-12 for v in row["chronological_window_deltas"])
            row["temporal_eligible"] = bool(row["chronological_delta"]>=.005-1e-12 and row["chronological_window_wins"]>=2 and row["validation_delta"]>=-.005-1e-12 and all(row[pool+"_vs_revision_delta"]>=-.003-1e-12 for pool in ["train","pooled"]))
    summary["temporal_eligible_recipes"] = [row["candidate"] for row in summary["final_comparison"] if row["temporal_eligible"]]
    summary["current_ensemble_validation"] = control[0]
    summary["current_ensemble_chronological"] = baseline_chrono
    runner.save(OUT/"summary.json",summary)
    frame = runner.pd.DataFrame(summary["final_comparison"])
    frame.to_csv(OUT/"final_comparison.csv",index=False)
    columns = ["candidate","train_mean","pooled_mean","train_vs_revision_delta","pooled_vs_revision_delta","train_vs_revision_wins","pooled_vs_revision_wins","qualifies","validation","chronological","chronological_window_wins","temporal_eligible"]
    body = '<!doctype html><meta charset="utf-8"><title>Nested decision LDA comparisons</title><style>body{font:16px/1.5 system-ui;margin:35px}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #ddd}</style><h1>Inner-fold decision learning for the submitted LDA ensemble</h1><p>Four frozen variants, all eight grouped repeats in both pools. Both original controls and latest LDA controls retained. Every ensemble candidate receives the fixed chronological and validation checks after CV, as predeclared. No actual test inputs are loaded.</p>'
    body += frame[columns].to_html(index=False,float_format=lambda v:f"{v:.6f}")
    body += '<p>CV qualification and temporal eligibility are distinct. Temporal eligibility requires chronology gain at least .005, gains in at least two windows, validation loss no more than .005 and CV loss no more than .003 in either pool. These reused labels and windows are development evidence; passing does not establish Kaggle improvement.</p><p><a href="protocol.json">Protocol</a> · <a href="summary.json">All results</a> · <a href="independent_verification.json">Independent verification</a></p>'
    (OUT/"report.html").write_text(body)
    (OUT/"metadata_only.csv").rename(OUT/"source_diagnostics.csv")
    print(frame[columns].to_string(index=False),flush=True)


def independent_verify():
    import verify_saved
    verify_saved.HERE = OUT
    verify_saved.main()


runner.OUT = OUT
runner.m = SimpleNamespace(BLOCKS=BLOCKS, LABELS=f.LABELS, EMB=f.EMB,
    train_native=f.train_native, native_prediction=f.native_prediction, fit_addition=fit_addition)
runner.prepare, runner.guard, runner.parity = prepare, guard, parity
runner.run_stage, runner.comparisons = run_stage, comparisons
runner.shortlist, runner.report, runner.confirm = shortlist, report, confirm
if __name__ == "__main__":
    if len(sys.argv)==2 and sys.argv[1]=="independent_verify":
        independent_verify()
    else:
        runner.main()
