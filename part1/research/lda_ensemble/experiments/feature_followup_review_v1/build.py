"""Consolidate completed Part 1 experiments without fitting or selecting models."""
from pathlib import Path
import hashlib
import json
import html
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
STUDIES = {
    "gold_correction_cv_v1": "Gold-only corrective heads and matched embedding controls",
    "other_transfer_cv_v1": "Transfer only Other corrections into the ensemble",
    "neighbor_labels_cv_v1": "Preserve multiple labels in nearest-neighbor votes",
    "other_residual_cv_v1": "Cross-fitted Other corrections with context and metadata",
    "other_utility_cv_v3": "Weight corrections by inner sample-F1 benefit or harm",
    "multilabel_teacher_cv_v1": "Allow multiple reasons during pseudo-label teaching",
    "selected_text_cv_v2": "Finer criteria, raw drug and disease text features",
    "biochemical_features_cv_v1": "Actions, pathway granularity and molecular fingerprints",
    "metadata_neighbors_cv_v1": "Use metadata to retrieve neighbors with matched embedding controls",
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources, studies, screening, confirmed, supplementary = {}, [], [], [], []
    manual_repeats = fold_arrays = supplementary_checks = fitting_contexts = 0
    for folder, title in STUDIES.items():
        p = ROOT / folder
        protocol, registry = read(p / "protocol.json"), read(p / "registry.json")
        verification, independent = read(p / "verification.json"), read(p / "independent_verification.json")
        assert verification["passed"] and independent["passed"]
        assert verification["part1_only"] and not verification["test_inputs_loaded"]
        assert verification["native_control_parity_all_folds"] and independent["source_hashes_unchanged"]
        for path, digest in protocol["source_sha256"].items():
            assert sha(ROOT.parent / path) == digest, path
        assert protocol["registry_sha256"] == sha(p / "registry.json")
        for name in ["protocol.json", "registry.json", "verification.json", "independent_verification.json",
                     "screen_comparison.json", "confirmation.json", "summary.json", "report.html"]:
            sources[f"{folder}/{name}"] = sha(p / name)
        initial = read(p / "screen_comparison.json")
        final = read(p / "confirmation.json")["comparison"]
        summary = read(p / "summary.json")
        assert len(initial) == len(registry)
        screening.extend(dict(study=folder, **r) for r in initial)
        confirmed.extend(dict(study=folder, **r) for r in final)
        supplementary.extend(dict(study=folder, **r) for r in summary["final_comparison"]
                             if "validation" in r or "chronological" in r)
        passed = [r for r in final if r["qualifies"]]
        studies.append(dict(study=folder, hypothesis=title, screened=len(initial),
            eight_repeat_recipes=len(final), confirmed_vs_parent=len(passed),
            ensemble_confirmations=sum(r["parent"] == "ensemble" for r in passed),
            independent_verification=True))
        manual_repeats += independent["manually_rescored_repeats"]
        fold_arrays += independent["fold_arrays_checked"]
        supplementary_checks += independent["supplementary_scores_checked"]
        fitting_contexts += verification["folds_verified"]
    candidates = pd.DataFrame(confirmed)
    frame = pd.DataFrame(studies)
    pd.DataFrame(screening).to_csv(OUT / "all_screening.csv", index=False)
    candidates.to_csv(OUT / "all_confirmation.csv", index=False)
    pd.DataFrame(supplementary).to_csv(OUT / "all_supplementary.csv", index=False)
    frame.to_csv(OUT / "studies.csv", index=False)
    failures = {}
    for name in ["other_utility_cv_v1", "other_utility_cv_v2", "selected_text_cv_v1"]:
        path = ROOT / name / "failure.json"
        failures[name] = read(path)
        sources[f"{name}/failure.json"] = sha(path)
    summary = dict(studies=len(studies), screened_parent_recipes=len(screening),
        eight_repeat_parent_recipes=len(confirmed),
        confirmed_vs_declared_parent=candidates.loc[candidates.qualifies, ["study", "candidate"]].to_dict("records"),
        confirmed_ensemble_upgrades=int(frame.ensemble_confirmations.sum()),
        manually_rescored_repeat_records=manual_repeats, fold_prediction_arrays_checked=fold_arrays,
        supplementary_records_checked=supplementary_checks, audited_outer_fold_contexts=fitting_contexts,
        part1_only=True, test_inputs_loaded=False, submissions_created=False, full235_refit=False,
        outcome="No confirmed upgrade to the three-way ensemble. Improvements against weaker Ani do not establish improvement over the stronger reference.",
        failures_retained=failures, source_sha256=sources, builder_sha256=sha(Path(__file__)))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    table = frame.copy()
    table["study"] = [f'<a href="../{s}/report.html">{s}</a>' for s in table.study]
    key = []
    for study, name in [("gold_correction_cv_v1", "ani__half_drug_targets"),
                        ("neighbor_labels_cv_v1", "ani__augmented_binary10_all"),
                        ("neighbor_labels_cv_v1", "ensemble__augmented_binary10_all"),
                        ("other_transfer_cv_v1", "ensemble__gold_targets_full"),
                        ("selected_text_cv_v2", "ensemble__augmented_disease_words"),
                        ("biochemical_features_cv_v1", "ani__goldhalf_morgan")]:
        row = next(r for r in confirmed if r["study"] == study and r["candidate"] == name)
        extra = next((r for r in supplementary if r["study"] == study and r["candidate"] == name), {})
        parent = row["parent"]
        key.append(dict(candidate=name, train=row["train_mean"], pooled=row["pooled_mean"],
                        train_wins=row[f"train_vs_{parent}_wins"], pooled_wins=row[f"pooled_vs_{parent}_wins"],
                        confirmed_vs_parent=row["qualifies"], validation=extra.get("validation"),
                        chronology=extra.get("chronological")))
    key = pd.DataFrame(key)
    key.to_csv(OUT / "key_results.csv", index=False)
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Part 1 feature follow-up results</title>
    <style>body{{font:16px/1.55 system-ui;max-width:1400px;margin:40px auto;padding:0 24px;color:#172333;background:#fafbfc}}table{{border-collapse:collapse;font-size:13px;width:100%;background:white}}th,td{{padding:9px;border:1px solid #dce2e8;text-align:left}}th{{background:#eaf0f7}}a{{color:#075c9b}}code{{font-size:.9em}}.note{{background:#fff2d6;padding:18px;border-left:4px solid #b06c00}}</style>
    <h1>Part 1 feature follow-up results</h1>
    <p class="note"><strong>No confirmed upgrade to the three-way ensemble.</strong> {len(screening)} parent-specific recipes across {len(studies)} completed studies were screened; {len(confirmed)} received all eight repeats in both pools. This count includes matched controls and related variants, not independent scientific hypotheses.</p>
    <p>The existing ensemble scores train157 <strong>.925690</strong>, pooled235 <strong>.916489</strong>, original validation <strong>.910256</strong>, and chronology <strong>.877551</strong>. The subset-averaged 75/25 model has higher grouped CV (.928875/.918794) but worse supplementary scores (.880342/.870748). The follow-ups establish no replacement for either reference. No Kaggle improvement has been measured.</p>
    <h2>Decision-relevant results</h2>{key.to_html(index=False, float_format=lambda x:f"{x:.6f}", na_rep="Not evaluated")}
    <p>Wins are paired repeat wins against the declared parent. Confirmation requires positive means and at least five of eight wins in both pools. Protein-target correction and binary neighbor votes passed against Ani only; both remain below the stronger ensemble. Fingerprints won six train repeats but only four pooled repeats. Their paired effect beyond the matched gold-only embedding correction was only +.000796 train / +.000089 pooled. Disease phrases lost their initial-screen gain.</p>
    <h2>Completed studies</h2>{table.to_html(index=False, escape=False)}
    <h2>What this changes</h2>
    <p>Retesting previously rejected fields under grouped CV was worthwhile: several initially promising effects disappeared on confirmation, and matched controls showed that much of the gold-head gain came from the training change rather than the added feature. Raw-drug subwords, supervised criteria terms, pathway granularity and molecular fingerprints now have explicit evidence under the stronger models. These bounded encodings do not exhaust all uses of those fields.</p>
    <p>The native pseudo-label teacher cannot generate multiple positives with its probability normalization, confidence .8 and label cutoff .3. Removing that restriction did not improve the ensemble reliably; it is not established as the cause of the performance gap. Other still concentrates the error budget. Context, nearest-label geometry, metadata and sample-F1-weighted corrections did not solve it in these tests.</p>
    <p>The final targeted study also tested molecular fingerprints, criteria terms, raw-drug fragments and protein targets as neighbor-retrieval features, using matched embedding-only controls. Its full results are linked above. Further work should be justified by a new representation or measurable complementary signal. These results do not support another threshold or blend search on the same predictions, or combining several unconfirmed features and calling the result a discovery.</p>
    <h2>Verification and retained failures</h2>
    <p>Independent rescoring checked {manual_repeats:,} complete-OOF repeat records, {fold_arrays:,} fold prediction arrays and {supplementary_checks:,} supplementary records. {fitting_contexts} outer-fold contexts checked row coverage, group purges, original-control parity and protected hashes. These are development repeats over reused data, not independent datasets.</p>
    <p>Three failed versions remain preserved: the initial utility test rejected valid zero-utility examples; its next version stopped on optimizer line-search failure; the first selected-text version mishandled bracketed isotope notation in raw drug names. Corrected versions reran their tests; no failed fold was dropped. All completed versions passed verification.</p>
    <p>Only supplied Part 1 files were used. No Part 2 phase/outcome data, external records, test fitting, test predictions, final235 refit, submission or model promotion.</p>
    <p><a href="all_screening.csv">All screens</a> · <a href="all_confirmation.csv">All confirmations</a> · <a href="all_supplementary.csv">Supplementary results</a> · <a href="summary.json">Provenance and verification totals</a></p></html>'''
    (OUT / "report.html").write_text(body)
    print(json.dumps({k:v for k,v in summary.items() if k not in ["source_sha256", "failures_retained"]}, indent=2))


if __name__ == "__main__":
    main()
