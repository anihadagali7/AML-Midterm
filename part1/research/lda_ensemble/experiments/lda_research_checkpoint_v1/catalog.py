"""Document this research branch without fitting or rewriting any experiment."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
import sys

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
PART1 = EXPERIMENTS.parent
PURPOSES = {
    "feature_gap_cv_v1": "Nine supplied-data feature additions and retries against original Ani and its ensemble.",
    "gold_correction_cv_v1": "Gold-only embedding/metadata corrections with matched no-feature controls.",
    "other_transfer_cv_v1": "Transfer gold-only metadata scores for Other.",
    "neighbor_labels_cv_v1": "Independent multilabel neighbor votes.",
    "other_residual_cv_v1": "Inner-cross-fitted residual corrections for Other.",
    "other_utility_cv_v1": "Failed preflight: zero sample-F1 utility was incorrectly forbidden; preserved.",
    "other_utility_cv_v2": "Confirmation stopped on optimizer failure; superseded by a complete v3 rerun.",
    "other_utility_cv_v3": "Sample-F1-weighted Other correction, with convergence checks.",
    "multilabel_teacher_cv_v1": "Multilabel pseudo-label teaching alternatives.",
    "selected_text_cv_v1": "Failed preflight: bracketed isotope drug names broke recursive parsing; preserved.",
    "selected_text_cv_v2": "Finer eligibility/drug text representations with corrected parsing.",
    "biochemical_features_cv_v1": "Protein targets and drug fingerprints from supplied Part 1 fields.",
    "metadata_neighbors_cv_v1": "Metadata-assisted neighbor retrieval.",
    "ani_lda_revision_cv_v1": "Later Ani LDA revision, regularization controls and teacher/final-head ablations.",
    "lda_covariance_cv_v1": "Gold-only and class-specific covariance heads against the LDA revision.",
    "joint_utility_cv_v1": "Direct sample-F1 utility and independent-label regression heads.",
    "conditional_other_cv_v1": "Binary and conditional Other heads against the LDA revision.",
    "lda_neighbor_geometry_cv_v1": "Alternative neighbor geometry against the LDA revision.",
    "lda_subset_cv_v1": "Grouped gold-subset averaging against the LDA revision.",
    "recruitment_novelty_cv_v1": "Reference-distance features for remaining reason confusion.",
    "lda_ensemble_mix_cv_v1": "Five fixed mixtures of LDA, tri-training and SVM partners.",
    "joint_lda_reason_cv_v1": "Joint reason-set LDA and expected sample-F1 decoding.",
    "lda_feature_retry_cv_v1": "Protein-target and fingerprint retries on the stronger LDA source.",
    "ani_direct_comparison_v1": "Fresh original-Ani versus LDA comparison under matched inductive and target-input access.",
    "temporal_lda_cv_v1": "Year drift, recency-weighted LR and time-aware neighbors.",
    "recent_era_lda_cv_v1": "Recent-gold experts and fixed blends with the full-data source.",
    "nested_decision_lda_cv_v1": "Inner-only cutoff learning, with equal or recent-weighted objectives.",
}
SUPPORT = ["feature_reassessment_v1", "feature_followup_review_v1", "lda_revision_review_v1",
    "lda_followup_review_v1", "temporal_generalization_review_v1", "final_lda_ensemble_trainval235_v1"]
HASHES = {}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    path = path.resolve()
    if path not in HASHES:
        h = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024*1024), b""):
                h.update(block)
        HASHES[path] = h.hexdigest()
    return HASHES[path]


def save(name, value):
    (HERE/name).write_text(json.dumps(value, indent=2)+"\n")


def write_csv(name, rows, fields):
    with (HERE/name).open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def study_record(name):
    folder = EXPERIMENTS/name
    protocol = read(folder/"protocol.json")
    registry = read(folder/"registry.json")
    failure = read(folder/"failure.json") if (folder/"failure.json").exists() else None
    summary = read(folder/"summary.json") if (folder/"summary.json").exists() else None
    status = "completed" if summary else "failed/partial; superseded"
    checks = {}
    for filename in ["verification.json", "independent_verification.json"]:
        if (folder/filename).exists():
            checks[filename] = read(folder/filename)
            assert checks[filename]["passed"], (name, filename)
    if summary:
        assert len(checks)==2, name
        if "protocol_sha256" in summary:
            assert summary["protocol_sha256"]==digest(folder/"protocol.json"), name
    assert protocol["registry_sha256"]==digest(folder/"registry.json"), name
    for path, expected in protocol["source_sha256"].items():
        assert digest(PART1/path)==expected, (name, path)
    results = []
    if summary:
        for phase, key in [("screen", "screening"), ("final", "final_comparison")]:
            results.extend(dict(study=name, phase=phase, result=row) for row in summary.get(key, []))
        if "comparison" in summary and "final_comparison" not in summary:
            results.extend(dict(study=name, phase="direct comparison", result=row) for row in summary["comparison"])
    elif (folder/"screen_comparison.json").exists():
        results.extend(dict(study=name, phase="incomplete study screen", result=row)
            for row in read(folder/"screen_comparison.json"))
    artifacts = {key:str((folder/key).relative_to(PART1)) for key in ["report.html", "protocol.json",
        "registry.json", "summary.json", "final_comparison.csv", "comparison.csv", "failure.json",
        "verification.json", "independent_verification.json", "per_trial_changes.csv", "per_label_metrics.csv"]
        if (folder/key).exists()}
    entry = dict(study=name, purpose=PURPOSES[name], status=status, registered_recipes=len(registry),
        parent=protocol.get("parent", protocol.get("requested_comparison", "See frozen protocol and registry")),
        recipe=protocol.get("fixed_recipe"), selection={k:protocol[k] for k in
            ["initial", "shortlist", "confirmation", "supplementary", "temporal_eligibility"] if k in protocol},
        confirmed_vs_own_parent=summary.get("confirmed_recipes", []) if summary else [],
        failure=failure, verifications=checks, artifacts=artifacts, registry=registry,
        protocol_sha256=digest(folder/"protocol.json"), protected_sources_checked=len(protocol["source_sha256"]))
    return entry, results


def build():
    if (HERE/"catalog.json").exists():
        raise SystemExit("Checkpoint exists. Use verify, or create a new version for changed evidence.")
    studies, results = [], []
    for name in PURPOSES:
        study, rows = study_record(name)
        studies.append(study)
        results.extend(rows)
    anchored_record = PART1/"../../AML-Midterm/part1/experiments/paired_improvement/candidate/kaggle_result.json"
    anchored = read(anchored_record)
    anchored_csv = PART1/"../../AML-Midterm"/anchored["submission_file"]
    anchored_archive = PART1/"../../AML-Midterm/part1/submissions/archive/submitted/anchored_reason_trainval235__kaggle_0.89636.csv"
    assert digest(anchored_csv)==digest(anchored_archive)==anchored["submission_sha256"]
    lda_dir = EXPERIMENTS/"final_lda_ensemble_trainval235_v1"
    lda_csv = lda_dir/"part1_lda_ensemble_trainval235.csv"
    assert digest(lda_csv)=="871bd7faff1380c13b4e099f725372399b40bf3954c70854efe9f3bad7d3f7a2"
    context = dict(lda_reported_kaggle=.89401, lda_submission_sha256=digest(lda_csv),
        lda_result_record=str((lda_dir/"kaggle_result.json").relative_to(PART1)),
        separate_anchored_result=anchored, anchored_record_path=str(anchored_record.relative_to(PART1)),
        anchored_record_sha256=digest(anchored_record), anchored_candidate_and_archive_bytes_match=True,
        attribution="Anchored result is from the sibling project. It is not a score of any new LDA follow-up in this checkpoint.",
        leaderboard_independently_queried=False, historical_preparation_and_result_records_unchanged=True,
        quota_note="User reported one daily slot remained, then submitted the different model; no remaining slot is assumed.")
    catalog = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), studies=studies,
        completed_studies=sum(s["status"]=="completed" for s in studies),
        failed_or_partial_studies=sum(s["status"]!="completed" for s in studies), support_directories=SUPPORT,
        submission_context=context, fitting_performed=False, submissions_created=False,
        scope="This integration branch's 27 versioned studies and six supporting reviews/export directories. Earlier imported research remains indexed by research/independent/README.md and its immutable import manifest.",
        interpretation="Registries include controls and related variants. Confirmation is relative to the study's declared parent. Incomplete runs are not completed negative experiments. These results do not establish whether the same changes help the newer anchored classifier.")
    save("catalog.json", catalog)
    save("all_recorded_comparisons.json", results)
    save("submission_context.json", context)
    write_csv("studies.csv", studies, ["study", "purpose", "status", "registered_recipes", "protocol_sha256"])
    flat = [dict(study=r["study"], phase=r["phase"], **r["result"]) for r in results]
    write_csv("comparisons.csv", flat, ["study", "phase", "candidate", "parent", "train_mean", "pooled_mean",
        "train_vs_revision_delta", "pooled_vs_revision_delta", "train_vs_revision_wins", "pooled_vs_revision_wins",
        "validation", "chronological", "qualifies", "temporal_eligible"])
    body = '<!doctype html><html><head><meta charset="utf-8"><title>LDA research checkpoint</title><style>body{font:16px/1.5 system-ui;max-width:1400px;margin:36px auto;padding:0 24px;color:#172132}table{border-collapse:collapse;font-size:13px}td,th{padding:9px;border:1px solid #d8dde5;text-align:left;vertical-align:top}th{background:#edf2f8}a{color:#145c9e}code{font-size:13px}</style></head><body>'
    body += '<h1>LDA research checkpoint</h1><p>This branch’s LDA ensemble improved grouped CV to .931794 train / .921631 pooled, but scored .89401 on Kaggle, tying the previous ensemble. Fourteen later temporal and cutoff variants produced no confirmed ensemble upgrade. The separate anchored classifier subsequently scored a user-reported <strong>.89636</strong>; its local candidate and scored archive match the recorded hash.</p>'
    body += '<p>This checkpoint documents completed comparisons, negative findings and failed versions. It does not retrain, upload, merge the newer model, or reinterpret old results against a new parent. CV and validation labels were repeatedly used during development. The original validation score does not validate the final235 refit.</p>'
    body += '<p><a href="../../RESEARCH_HANDOFF.md">Central research handoff</a> · <a href="../../TESTING_POLICY.md">Testing policy</a> · <a href="catalog.json">Full catalog and selection procedures</a> · <a href="submission_context.json">Exact-file submission provenance</a></p>'
    body += '<h2>Model and evaluation</h2><p>The submitted ensemble uses 70% centered LDA-revision source, 15% classical tri-training and 15% SVM votes. The source combines .5 logistic regression (C=.1), .3 cosine neighbors and .2 shrinkage LDA, with two pseudo-label rounds and a native .3 cutoff. Fits are inductive by default; the direct-comparison target-input arm is explicitly separate. Gold labels remain unchanged, with held-group masked duplicates purged in grouped CV and chronology.</p>'
    body += '<p>Saved grouped five-fold assignments use eight repeats in train157 and pooled235. Primary comparisons require positive means and at least five paired wins in both pools. Original validation78 and chronology98 are separate supplementary checks. Every learned step belongs to its fitting partition. No Part 2 data or external trial records were used.</p>'
    body += '<h2>Study index</h2><table><thead><tr><th>Study / purpose</th><th>Status</th><th>Registered recipes</th><th>Evidence</th></tr></thead><tbody>'
    for study in studies:
        name=study["study"]
        links=' · '.join(f'<a href="../{name}/{key}">{html.escape(key)}</a>' for key in
            ["report.html","protocol.json","registry.json","summary.json","failure.json","independent_verification.json"] if key in study["artifacts"])
        body+=f'<tr><td><strong>{html.escape(name)}</strong><br>{html.escape(study["purpose"])}</td><td>{study["status"]}</td><td>{study["registered_recipes"]}</td><td>{links}</td></tr>'
    body+='</tbody></table><p>Counts include matched controls and related variants. “Completed” does not imply improvement. See each study’s own comparator and confirmation results; do not combine different parents or access regimes into one ranking. Source-only wins in earlier studies did not necessarily improve the ensemble.</p>'
    body+='<h2>Reviews and final artifact</h2><ul>'
    for name in SUPPORT:
        folder=EXPERIMENTS/name
        target=next(p for p in ["report.html","manifest.json","findings.json"] if (folder/p).exists())
        body+=f'<li><a href="../{name}/{target}">{name}</a></li>'
    body+='</ul><h2>Preservation and replay</h2><p>Original code, frozen recipes, protocols, complete-OOF predictions, per-fold audits, supplementary results, failures and logs remain in their original directories. The manifest records every non-bytecode file in the cataloged directories; no artifacts were moved. Existing imported research is linked through the central handoff.</p>'
    body+='<p>From the integration repository, verify this documentation snapshot with <code>python3 part1/experiments/lda_research_checkpoint_v1/catalog.py verify</code>. Verification hashes saved files and checks the frozen source guards; it does not fit models. Study-specific prediction rescoring commands are in their saved runners and reports. Read the testing policy before replaying a fitting command; never overwrite a scored CSV or modify frozen study code.</p>'
    body+='<p><a href="studies.csv">Study CSV</a> · <a href="comparisons.csv">Compact results CSV</a> · <a href="all_recorded_comparisons.json">Unabridged recorded comparison rows</a> · <a href="artifact_manifest.json">Artifact hashes</a> · <a href="verification.json">Checkpoint verification</a></p></body></html>'
    (HERE/"index.html").write_text(body)
    files=[]
    for name in [*PURPOSES, *SUPPORT]:
        for path in sorted((EXPERIMENTS/name).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix!=".pyc":
                files.append(dict(path=str(path.relative_to(PART1)), bytes=path.stat().st_size, sha256=digest(path)))
    own=[p for p in sorted(HERE.iterdir()) if p.is_file() and p.name not in ["artifact_manifest.json","verification.json"]]
    files.extend(dict(path=str(p.relative_to(PART1)), bytes=p.stat().st_size, sha256=digest(p)) for p in own)
    save("artifact_manifest.json", dict(files=files, file_count=len(files), bytes=sum(f["bytes"] for f in files),
        excludes=["bytecode caches", "the manifest itself", "verification output", "mutable central handoff"],
        external_score_reference=dict(path=context["anchored_record_path"],sha256=context["anchored_record_sha256"])))
    verify()


def verify():
    manifest=read(HERE/"artifact_manifest.json")
    for row in manifest["files"]:
        path=PART1/row["path"]
        assert path.stat().st_size==row["bytes"] and digest(path)==row["sha256"], row["path"]
    catalog=read(HERE/"catalog.json")
    protected=0
    for study in catalog["studies"]:
        p=read(EXPERIMENTS/study["study"]/"protocol.json")
        for name,expected in p["source_sha256"].items():
            assert digest(PART1/name)==expected,name
            protected+=1
    import re
    links=re.findall(r'href="([^"]+)"',(HERE/"index.html").read_text())
    for link in links:
        if link!="verification.json":
            assert (HERE/link).exists(),link
    result=dict(passed=True,checked_at_utc=datetime.now(timezone.utc).isoformat(),
        files_hashed=len(manifest["files"]),bytes_hashed=manifest["bytes"],
        protected_source_references_checked=protected,completed_studies=catalog["completed_studies"],
        failed_or_partial_studies=catalog["failed_or_partial_studies"],links_checked=len(links),
        model_fitting_performed=False,new_predictions_created=False,source_artifacts_rewritten=False,
        metric_verification="Existing passing independent prediction-rescore records are preserved and hashed; this documentation check does not rerun model scoring.")
    save("verification.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    assert len(sys.argv)==2 and sys.argv[1] in ["build","verify"]
    globals()[sys.argv[1]]()
