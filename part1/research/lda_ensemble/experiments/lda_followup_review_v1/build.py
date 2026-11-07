"""Read-only consolidation of the follow-up round against the new LDA controls."""
from pathlib import Path
import hashlib
import json
import html
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
LABELS = ["Safety", "Funding", "Efficacy", "Recruitment", "Other"]
STUDIES = {
    "conditional_other_cv_v1": "Binary and Recruitment-conditional Other heads",
    "lda_neighbor_geometry_cv_v1": "PCA, partial whitening and common-component removal",
    "lda_subset_cv_v1": "Fully refitted grouped subset averages",
    "recruitment_novelty_cv_v1": "Gold-reference distance features for Recruitment+Other",
    "lda_ensemble_mix_cv_v1": "Five fixed ensemble partner mixes",
    "joint_lda_reason_cv_v1": "Joint-label-set shrinkage LDA",
    "lda_feature_retry_cv_v1": "Protein-target and molecular-fingerprint retries",
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnostic():
    directory = ROOT/"ani_lda_revision_cv_v1"
    parts = [np.load(directory/stage/"ensemble__revision_full.npz") for stage in ["pooled3", "pooled5"]]
    np.testing.assert_array_equal(parts[0]["ids"], parts[1]["ids"])
    np.testing.assert_array_equal(parts[0]["truth"], parts[1]["truth"])
    truth, ids = parts[0]["truth"], parts[0]["ids"]
    pred = np.concatenate([p["predictions"] for p in parts])
    score = np.concatenate([p["scores"] for p in parts])
    f1 = 2*(pred*truth[None]).sum(2)/(pred.sum(2)+truth.sum(1)[None])
    rows = []
    for pattern in np.unique(truth, axis=0):
        mask = (truth==pattern).all(1)
        rows.append(dict(actual_labels=" + ".join(label for label, value in zip(LABELS, pattern) if value),
            trials=int(mask.sum()), mean_sample_f1=float(f1[:, mask].mean()),
            exact_set_accuracy=float((pred[:, mask]==truth[None, mask]).all(2).mean()),
            other_prediction_rate=float(pred[:, mask, 4].mean()),
            mean_other_decision_score=float(score[:, mask, 4].mean()),
            contribution_to_total_f1_loss=float((1-f1[:, mask]).sum(1).mean()/len(truth))))
    trials = pd.DataFrame(dict(ID=ids, actual_labels=[" + ".join(label for label, value in zip(LABELS, row) if value) for row in truth],
        mean_sample_f1=f1.mean(0), exact_set_errors=(pred!=truth[None]).any(2).sum(0),
        other_positive_repeats=pred[:, :, 4].sum(0)))
    cohorts = pd.DataFrame(rows).sort_values("contribution_to_total_f1_loss", ascending=False)
    trials.sort_values(["mean_sample_f1", "ID"]).to_csv(OUT/"baseline_trial_errors.csv", index=False)
    cohorts.to_csv(OUT/"baseline_labelset_errors.csv", index=False)
    return cohorts


def table(frame):
    return frame.to_html(index=False, escape=False, border=0, float_format=lambda value:f"{value:.6f}")


def main():
    sources, all_rows, outcomes = {}, [], []
    totals = dict(repeat_records=0, fold_arrays=0, supplementary_records=0)
    for study, description in STUDIES.items():
        p = ROOT/study
        protocol = read(p/"protocol.json")
        for path, digest in protocol["source_sha256"].items():
            assert sha(ROOT.parent/path)==digest, path
        assert sha(p/"registry.json")==protocol["registry_sha256"]
        independent = read(p/"independent_verification.json")
        assert independent["passed"] and read(p/"verification.json")["passed"]
        totals["repeat_records"] += independent["manually_rescored_repeats"]
        totals["fold_arrays"] += independent["fold_arrays_checked"]
        totals["supplementary_records"] += independent["supplementary_scores_checked"]
        results = read(p/"summary.json")["final_comparison"]
        all_rows.extend(dict(study=study, **row) for row in results)
        outcomes.append(dict(study=study, question=description, recipes=len(results),
            confirmed_against_latest_parent=sum(bool(row["qualifies"]) for row in results)))
        for filename in ["protocol.json", "registry.json", "confirmation.json", "summary.json", "verification.json", "independent_verification.json", "report.html"]:
            sources[study+"/"+filename] = sha(p/filename)
    results = pd.DataFrame(all_rows)
    results.to_csv(OUT/"all_results.csv", index=False)
    cohorts = diagnostic()
    for stage in ["pooled3", "pooled5"]:
        path = ROOT/"ani_lda_revision_cv_v1"/stage/"ensemble__revision_full.npz"
        sources[str(path.relative_to(ROOT))] = sha(path)
    controls = [dict(model="Current LDA 70/15/15 ensemble", train=.9317940552016986, pooled=.9216312056737589,
                     validation=.9059829059829061, chronology=.8775510204081632)]
    # These examples are descriptive, selected after observing all comparisons.
    highlights = [("lda_ensemble_mix_cv_v1", "ensemble__tri25"),
                  ("lda_subset_cv_v1", "ensemble__half_all"),
                  ("recruitment_novelty_cv_v1", "ensemble__lr_empirical")]
    for study, name in highlights:
        r = next(r for r in all_rows if r["study"]==study and r["candidate"]==name)
        controls.append(dict(model=study+" / "+name, train=r["train_mean"], pooled=r["pooled_mean"],
            validation=r.get("validation"), chronology=r.get("chronological")))
    comparison = pd.DataFrame(controls)
    comparison.to_csv(OUT/"comparison.csv", index=False)
    confirmed = [dict(study=r["study"], candidate=r["candidate"], parent=r["parent"]) for r in all_rows if r["qualifies"]]
    metadata = [r for r in all_rows if r["study"]=="lda_feature_retry_cv_v1"]
    metadata_columns = ["candidate", "train_mean", "pooled_mean", "train_vs_revision_delta", "pooled_vs_revision_delta",
        "train_vs_matched_feature_delta", "pooled_vs_matched_feature_delta", "metadata_contribution_confirmed", "qualifies"]
    metadata_table = pd.DataFrame(metadata)[metadata_columns]
    metadata_table.to_csv(OUT/"metadata_attribution.csv", index=False)
    summary = dict(studies=outcomes, recipes=len(all_rows), confirmed=confirmed, comparator="Matching latest LDA source or 70/15/15 ensemble",
        verification_totals=totals, labelset_diagnostic=cohorts.to_dict("records"), comparison=controls,
        source_sha256=sources, builder_sha256=sha(Path(__file__)), part1_only=True,
        test_inputs_loaded=False, full235_refit=False, submissions_created=False, model_promoted=False,
        interpretation="Development-CV comparisons using reused folds and labels; descriptive error cohorts and selected examples are not independent evidence.")
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    outcome = "No further recipe met the two-pool confirmation rule against its latest LDA parent." if not confirmed else f"{len(confirmed)} recipes met the two-pool confirmation rule; supplementary tradeoffs are retained below."
    study_table = pd.DataFrame(outcomes)
    study_table["study"] = [f'<a href="../{name}/report.html">{name}</a>' for name in study_table.study]
    compact = results[["study", "candidate", "train_mean", "pooled_mean", "train_vs_revision_delta", "train_vs_revision_wins",
        "pooled_vs_revision_delta", "pooled_vs_revision_wins", "qualifies", "validation", "chronological"]]
    other = cohorts[cohorts.actual_labels=="Other"].iloc[0]
    both = cohorts[cohorts.actual_labels=="Recruitment + Other"].iloc[0]
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>LDA follow-up experiments</title>
    <style>body{{font:16px/1.5 system-ui;max-width:1400px;margin:40px auto;padding:0 24px;color:#182739}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:8px;border:1px solid #d9e0e7;text-align:left}}th{{background:#edf2f7}}a{{color:#075c9b}}.callout{{background:#edf2f7;padding:20px;border-left:5px solid #4b6b89}}.scroll{{overflow-x:auto}}</style>
    <h1>Further tests against the stronger LDA model</h1>
    <p class="callout">{html.escape(outcome)} {len(all_rows)} parent-specific recipes, including controls and related variants, received eight fixed grouped repeats in both pools. These are not {len(all_rows)} independent hypotheses.</p>
    <p>The current reference remains .931794 train CV / .921631 pooled CV, .905983 original validation and .877551 chronology. No result here is a Kaggle score. The highest reported Kaggle result is still the original Ani artifact's .89423.</p>
    <h2>What was tested</h2>{table(study_table)}
    <p>Confirmation requires positive paired means and at least five of eight wins in both pools against the matching latest LDA parent. Original Ani/ensemble references are retained in each study, but beating the older source does not qualify as an additional improvement. Supplementary examples were selected only after eight-repeat comparisons.</p>
    <h2>Why a higher train score was insufficient</h2><div class="scroll">{table(comparison)}</div>
    <p>The 75/25 mixture raised train CV but reduced pooled CV and validation. The subset average mostly tied the current ensemble and did not meet the paired-win requirement. All results, including negative supplementary results, remain available.</p>
    <h2>The remaining Other blind spot</h2>
    <p>Across pooled OOF predictions, Other is detected on {other.other_prediction_rate:.1%} of the 26 Other-only trials, but only {both.other_prediction_rate:.1%} of the nine Recruitment+Other trials. These rates average over eight repeated predictions per trial. The first specialist studies targeted this gap, but their added Other predictions did not produce reliable overall gains.</p>
    <p>Reference-distance features excluded each training example's entire embedding group. Joint-pattern LDA learned combined labels as distinct outcomes. Neither recipe family confirmed. This does not prove the cases are unlearnable; there are only nine distinct gold examples of that combination, and the labels/folds have already been used in development.</p>
    <div class="scroll">{table(cohorts)}</div>
    <h2>Feature retries with matched controls</h2>
    <p>Protein targets and molecular fingerprints were retried with the latest C.1 logistic head and revised pseudo-labels. Matching heads without metadata distinguish feature effects from using gold-only training. Final LDA and neighbors were held fixed. Fingerprint selection used fitting gold only; no external chemistry data was fetched.</p>
    <div class="scroll">{table(metadata_table)}</div>
    <h2>All paired comparisons against the latest LDA reference</h2><div class="scroll">{table(compact)}</div>
    <h2>Verification</h2>
    <p>Independent rescoring checked {totals['repeat_records']:,} complete-OOF repeat records, {totals['fold_arrays']:,} fold arrays and {totals['supplementary_records']:,} supplementary records. All protected hashes, saved split coverage, grouped purges and native control checks passed. Ensemble reweighting reused exact verified source predictions and reconstructed both controls; it required no additional fits.</p>
    <p>Only supplied Part 1 inputs were used. No Part 2 fields, external trial records, test fitting or prediction, final235 refit, submission, or model promotion. These experiments identify what failed under a particular recipe; they cannot rule out every encoding or guarantee performance on Kaggle.</p>
    <p><a href="all_results.csv">All results</a> · <a href="metadata_attribution.csv">Feature attribution</a> · <a href="baseline_trial_errors.csv">Per-trial error diagnostic</a> · <a href="summary.json">Sources and verification</a> · <a href="../../RESEARCH_HANDOFF.md">Research handoff</a></p></html>'''
    (OUT/"report.html").write_text(body)
    print(json.dumps(dict(recipes=len(all_rows), confirmed=confirmed, verification_totals=totals), indent=2))


if __name__=="__main__":
    main()
