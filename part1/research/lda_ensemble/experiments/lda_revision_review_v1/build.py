"""Read-only evidence consolidation for the latest Ani revision and follow-ups."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
STUDIES = ["ani_lda_revision_cv_v1", "lda_covariance_cv_v1", "joint_utility_cv_v1"]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources, all_rows, outcomes = {}, [], []
    repeats = folds = supplementary = 0
    for study in STUDIES:
        p = ROOT/study
        protocol = read(p/"protocol.json")
        for path, digest in protocol["source_sha256"].items():
            assert sha(ROOT.parent/path) == digest, path
        assert sha(p/"registry.json") == protocol["registry_sha256"]
        independent = read(p/"independent_verification.json")
        assert independent["passed"]
        repeats += independent["manually_rescored_repeats"]
        folds += independent["fold_arrays_checked"]
        supplementary += independent["supplementary_scores_checked"]
        results = read(p/"summary.json")["final_comparison"]
        all_rows.extend(dict(study=study, **row) for row in results)
        outcomes.append(dict(study=study, recipes=len(results),
            passed_against_declared_control=sum(row["qualifies"] for row in results),
            comparator="Original Ani/ensemble" if study==STUDIES[0] else "Latest LDA Ani/ensemble"))
        for filename in ["protocol.json", "registry.json", "confirmation.json", "summary.json", "independent_verification.json", "report.html"]:
            sources[study+"/"+filename] = sha(p/filename)
    reference_rows = read(ROOT/STUDIES[0]/"confirmation.json")["comparison"]
    contrasts = []
    for parent in ["ani", "ensemble"]:
        revised = next(r for r in reference_rows if r["candidate"]==parent+"__revision_full")
        regularization = next(r for r in reference_rows if r["candidate"]==parent+"__regularization_only")
        record = dict(parent=parent, contrast="LDA revision minus C.1 without LDA")
        for pool in ["train", "pooled"]:
            delta = np.array(revised[pool+"_repeats"])-regularization[pool+"_repeats"]
            record[pool+"_mean_delta"] = float(delta.mean())
            record[pool+"_wins"] = int((delta>1e-12).sum())
            record[pool+"_ties"] = int((abs(delta)<=1e-12).sum())
            record[pool+"_paired_deltas"] = delta.tolist()
        contrasts.append(record)
    winner = next(r for r in all_rows if r["study"]==STUDIES[0] and r["candidate"]=="ensemble__revision_full")
    comparison = pd.DataFrame([
        dict(model="Original 70/15/15 ensemble", train=.9256900212314225, pooled=.9164893617021276,
             validation=.9102564102564102, chronology=.8775510204081632),
        dict(model="LDA-revision 70/15/15 ensemble", train=winner["train_mean"], pooled=winner["pooled_mean"],
             validation=winner["validation"], chronology=winner["chronological"])])
    pd.DataFrame(all_rows).to_csv(OUT/"all_results.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT/"lda_vs_regularization.csv", index=False)
    comparison.to_csv(OUT/"comparison.csv", index=False)
    summary = dict(outcome="Confirmed grouped-CV improvement from latest Ani LDA revision; original validation slightly worse and chronology tied. No established Kaggle improvement.",
        candidate="ensemble__revision_full", source_study=STUDIES[0], studies=outcomes,
        paired_train_delta=winner["train_vs_ensemble_delta"], paired_pooled_delta=winner["pooled_vs_ensemble_delta"],
        paired_train_wins=winner["train_vs_ensemble_wins"], paired_pooled_wins=winner["pooled_vs_ensemble_wins"],
        comparison=comparison.to_dict("records"), lda_vs_regularization=contrasts,
        independent_repeat_records=repeats, independent_fold_arrays=folds,
        independent_supplementary_records=supplementary, source_sha256=sources,
        part1_only=True, test_inputs_loaded=False, submissions_created=False, full235_refit=False,
        builder_sha256=sha(Path(__file__)))
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    study_table = pd.DataFrame(outcomes)
    study_table["study"] = [f'<a href="../{s}/report.html">{s}</a>' for s in study_table.study]
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Latest Ani revision: confirmed CV gain</title>
    <style>body{{font:16px/1.55 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#182739}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;border:1px solid #d9e0e7;text-align:left}}th{{background:#e9f0f7}}a{{color:#075c9b}}.callout{{padding:20px;background:#ecf5ef;border-left:5px solid #337a4e}}</style>
    <h1>A stronger grouped-CV candidate from the latest Ani revision</h1>
    <p class="callout">Replacing the original Ani component with the previously unbenchmarked LDA revision improves the 70/15/15 ensemble by <strong>+.006104 train CV and +.005142 pooled CV</strong>, winning <strong>7/8 repeats in both pools</strong>. Original validation is lower by .004274 and chronology ties. This is a confirmed development-CV improvement, not a measured Kaggle improvement.</p>
    {comparison.to_html(index=False,float_format=lambda x:f"{x:.6f}")}
    <h2>What was missing</h2>
    <p>The preserved 8a8c3fc Ani notebook changed the source model to .5 multinomial logistic regression, .3 cosine neighbors and .2 shrinkage LDA, and changed logistic C from .3 to .1. It also used that LDA mixture during pseudo-label teaching. The shared grouped benchmark had only evaluated the earlier source generation. This study adds an inductive adapter, with direct parity against the notebook's function definitions on identical permitted fitting data. Notebook input-loading, transductive fitting and export cells were not executed.</p>
    <p>The notebook itself uses a different input-access regime and validation protocol. These measurements belong to this inductive adapter and cannot be assigned to its existing submission CSV. The highest reported Kaggle score remains .89423 for the older transductive artifact; the user reports another team near .93.</p>
    <h2>Matched ablations and further tests</h2>
    {study_table.to_html(index=False,escape=False)}
    <p>The new full ensemble also beats its C.1/no-LDA control by +.004777 train and +.003901 pooled. The separate paired repeat counts are in <a href="lda_vs_regularization.csv">the contrast table</a>; the gain is not solely the regularization change. Gold-only covariance heads and class-specific covariance models failed to improve the new revision consistently. A direct conditional sample-F1 utility expert, compared with matched independent-label regression, also failed the two-pool confirmation rule.</p>
    <p>The utility experiment used the conditional expected-F1 decision principle described by <a href="https://www.jmlr.org/papers/v15/waegeman14a.html">Waegeman et al.</a>, implemented here as kernel regression of all 31 nonempty label-set utilities. Its finite-sample estimates have no Bayes-optimality guarantee. It used existing Part 1 labels and embeddings only.</p>
    <h2>Where the remaining gap is</h2>
    <p>Pooled Funding F1 improves from .769493 to .821196. Other precision rises from .707766 to .758236, but Other recall slips from .550000 to .539286. The strongest persistent Other misses are therefore still unresolved. The .931794 training-CV number must not be presented as .93 Kaggle performance, or as evidence that the leaderboard gap has been closed.</p>
    <h2>Verification and status</h2>
    <p>Independent rescoring verified {repeats:,} complete-OOF repeat records, {folds:,} fold arrays and {supplementary:,} supplementary records. Protected source/data hashes, fold coverage, grouped purges, original controls, notebook-function parity and query-batch invariance passed. All recipes were frozen before scoring; follow-ups used the new LDA revision as their comparator.</p>
    <p>No Part 2 data, external records, test fitting/predictions, final235 refit, upload, retagging or model promotion. This candidate is ready for review; no submission has been spent.</p>
    <p><a href="all_results.csv">All candidate results</a> · <a href="summary.json">Sources and verification</a> · <a href="../../RESEARCH_HANDOFF.md">Research handoff</a></p></html>'''
    (OUT/"report.html").write_text(body)
    print(json.dumps({k:v for k,v in summary.items() if k not in ["source_sha256", "lda_vs_regularization"]}, indent=2))


if __name__=="__main__":
    main()
