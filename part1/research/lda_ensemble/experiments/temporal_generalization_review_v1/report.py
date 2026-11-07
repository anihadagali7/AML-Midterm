"""Consolidate the three completed, predeclared Part 1 comparisons."""
from pathlib import Path
import hashlib
import html
import json
import numpy as np
import pandas as pd
from audit import HERE,PART1,LABELS,STUDIES,read


def arrays(directory,pool,name):
    result = {}
    for stage in [pool+"3",pool+"5"]:
        with np.load(directory/stage/(name+".npz")) as a:
            for key in ["ids","truth"]:
                if key in result:
                    np.testing.assert_array_equal(result[key],a[key])
                result[key]=a[key].copy()
            for key in ["scores","predictions"]:
                result[key]=np.concatenate([result[key],a[key]]) if key in result else a[key].copy()
    return result


def f1(a):
    pred,truth=a["predictions"],a["truth"]
    return 2*(pred*truth).sum(2)/(pred.sum(2)+truth.sum(1))


def decode(score):
    predicted=(score>=.5).astype(int)
    empty=predicted.sum(1)==0
    predicted[np.flatnonzero(empty),score[empty].argmax(1)]=1
    return predicted


def main():
    verification=read(HERE/"verification.json")
    assert verification["passed"]
    revision=HERE.parent/"ani_lda_revision_cv_v1"
    base={pool:arrays(revision,pool,"ensemble__revision_full") for pool in ["train","pooled"]}
    baseline=dict(candidate="Submitted LDA ensemble",study="ani_lda_revision_cv_v1",
        train_mean=float(f1(base["train"]).mean()),pooled_mean=float(f1(base["pooled"]).mean()),
        validation=.9059829059829061,chronological=.8775510204081632,
        train_vs_revision_delta=0.,pooled_vs_revision_delta=0.,
        train_vs_revision_wins=0,pooled_vs_revision_wins=0,qualifies=False,temporal_eligible=False)
    combined=[];changes=[];chronology_changes=[]
    for name in STUDIES:
        directory=HERE.parent/name
        summary=read(directory/"summary.json")
        for pool in ["train","pooled"]:
            matched=arrays(directory,pool,"ensemble__revision_full")
            np.testing.assert_array_equal(matched["ids"],base[pool]["ids"])
            np.testing.assert_allclose(matched["scores"],base[pool]["scores"],atol=1e-10,rtol=0)
        for row in summary["final_comparison"]:
            combined.append(dict(row,study=name))
            if row["parent"]!="ensemble":
                continue
            for pool in ["train","pooled"]:
                a=arrays(directory,pool,row["candidate"])
                np.testing.assert_array_equal(a["ids"],base[pool]["ids"])
                delta=f1(a)-f1(base[pool])
                changes.append(dict(study=name,candidate=row["candidate"],pool=pool,
                    mean_changed_label_sets=float(np.any(a["predictions"]!=base[pool]["predictions"],axis=2).sum(1).mean()),
                    mean_trials_improved=float((delta>1e-12).sum(1).mean()),
                    mean_trials_worsened=float((delta<-1e-12).sum(1).mean()),
                    sample_f1_delta=float(delta.mean())))
            for window in [1,2,3]:
                with np.load(directory/"supplementary"/f"window{window}_ensemble__revision_full.npz") as old_window, np.load(directory/"supplementary"/f"window{window}_{row['candidate']}.npz") as new_window:
                    np.testing.assert_array_equal(old_window["ids"],new_window["ids"])
                    a,b=decode(old_window["scores"]),decode(new_window["scores"])
                    truth=old_window["truth"]
                    delta=2*(b*truth).sum(1)/(b.sum(1)+truth.sum(1))-2*(a*truth).sum(1)/(a.sum(1)+truth.sum(1))
                    for i in np.flatnonzero(np.any(a!=b,axis=1)):
                        labels=lambda v:" ".join(label for label,flag in zip(LABELS,v) if flag)
                        chronology_changes.append(dict(study=name,candidate=row["candidate"],window=window,
                            ID=str(old_window["ids"][i]),truth=labels(truth[i]),baseline=labels(a[i]),
                            prediction=labels(b[i]),sample_f1_delta=float(delta[i])))
    data=pd.concat([pd.read_csv(PART1/"data/part1_train.csv"),pd.read_csv(PART1/"data/part1_val.csv")]).set_index("ID")
    years=data.loc[base["pooled"]["ids"],"start_year"].to_numpy()
    old=arrays(revision,"pooled","ensemble")
    cohorts=[]
    for label,mask in [("Through 2004",years<=2004),("2005 onward",years>=2005),("2008 onward",years>=2008)]:
        a,b=f1(old)[:,mask],f1(base["pooled"])[:,mask]
        delta=b.mean(1)-a.mean(1)
        cohorts.append(dict(cohort=label,trials=int(mask.sum()),previous_ensemble=float(a.mean()),
            submitted_lda_ensemble=float(b.mean()),delta=float(delta.mean()),wins=int((delta>1e-12).sum()),
            note="Retrospective subset of pooled CV, not a new temporal holdout; cohorts overlap."))
    table=pd.DataFrame(combined)
    table.to_csv(HERE/"comparison.csv",index=False)
    pd.DataFrame(changes).to_csv(HERE/"prediction_changes.csv",index=False)
    pd.DataFrame(chronology_changes).to_csv(HERE/"chronological_prediction_changes.csv",index=False)
    pd.DataFrame(cohorts).to_csv(HERE/"historical_year_cohorts.csv",index=False)
    qualified=[dict(study=r["study"],candidate=r["candidate"]) for r in combined if r["parent"]=="ensemble" and r["qualifies"]]
    temporal=[dict(study=r["study"],candidate=r["candidate"]) for r in combined if r["parent"]=="ensemble" and r["temporal_eligible"]]
    outcome=dict(baseline=baseline,ensemble_cv_qualified=qualified,ensemble_temporally_eligible=temporal,
        frozen_mechanisms=14,parent_specific_candidates=len(combined),studies=STUDIES,
        comparison=combined,prediction_changes=changes,chronological_prediction_changes=chronology_changes,historical_year_cohorts=cohorts,
        independent_repeat_records=sum(s["independent_rescore"]["manually_rescored_repeats"] for s in verification["studies"]),
        independent_fold_arrays=sum(s["independent_rescore"]["fold_arrays_checked"] for s in verification["studies"]),
        independent_supplementary_records=sum(s["independent_rescore"]["supplementary_scores_checked"] for s in verification["studies"]),
        part1_only=True,final235_refit=False,test_predictions_created=False,submissions_created=False,
        all_latest_lda_control_scores_match_saved_baseline=True,
        interpretation="These are related variants tested on reused labels. CV qualification and predeclared temporal eligibility are different criteria; neither establishes Kaggle improvement.",
        report_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/"summary.json").write_text(json.dumps(outcome,indent=2)+"\n")
    cols=["candidate","train_mean","pooled_mean","train_vs_revision_wins","pooled_vs_revision_wins","validation","chronological","qualifies","temporal_eligible"]
    view=pd.concat([pd.DataFrame([baseline]),table[table.parent=="ensemble"]],ignore_index=True)
    text='<!doctype html><html><head><meta charset="utf-8"><title>Part 1 generalization follow-up</title><style>body{font:16px/1.5 system-ui;max-width:1500px;margin:36px auto;padding:0 24px;color:#172132}table{border-collapse:collapse;font-size:13px}th,td{border:1px solid #d8dde5;padding:8px;text-align:left}th{background:#edf2f8}a{color:#145c9e}.note{padding:18px;background:#f4f6f8;border-left:4px solid #637d9f}</style></head><body><h1>Part 1: testing the Kaggle generalization gap</h1>'
    text+='<p>The submitted LDA ensemble scored .89401 on Kaggle, equal to the previous ensemble despite stronger local CV. The two local submission files differ on 13 of 312 label sets. No hidden labels or external records were used to explain the tie.</p>'
    text+=f'<p class="note"><strong>New results:</strong> {len(qualified)} ensemble recipes met ordinary CV confirmation; {len(temporal)} met the separately predeclared temporal eligibility rule. No final235 model or submission was created by this follow-up.</p>'
    text+='<p>Fourteen frozen mechanisms (28 parent-specific comparisons) across three studies received the same eight grouped repeats in both pools. Each ensemble variant also received the fixed validation and chronological checks under a predeclared diagnostic exception. The experiments tested linear time drift, recency-weighted classifiers, time-aware neighbors, recent-gold experts and inner-fold decision cutoffs. All fitting used only allowed Part 1 inputs; actual test inputs were not loaded.</p>'
    text+=view[cols].to_html(index=False,float_format=lambda v:f"{v:.6f}")
    text+='<p>Wins are paired wins out of eight repeats against the submitted LDA parent. Ordinary qualification requires positive means and at least five wins in both pools. Temporal eligibility requires chronology gain ≥.005, improvement in at least two of three windows, validation loss ≤.005 and CV loss ≤.003 in either pool. The two criteria remain separate.</p>'
    cutoff_changes=[r for r in chronology_changes if r["study"]=="nested_decision_lda_cv_v1" and r["candidate"]=="ensemble__global_recent"]
    text+='<p>The recent-weighted shared cutoff changes four chronological label sets: two improve and two worsen, all in the final window. The net chronological gain is .003401. This is a weak lead rather than a confirmed improvement.</p>'
    text+=pd.DataFrame(cutoff_changes)[["ID","truth","baseline","prediction","sample_f1_delta"]].to_html(index=False,float_format=lambda v:f"{v:.6f}")
    text+='<h2>Where the earlier local improvement occurred</h2><p>The submitted LDA gain was not confined to old trials. These are descriptive, overlapping subsets of the existing pooled CV; they are not new holdouts and were not used to select the new recipes.</p>'
    text+=pd.DataFrame(cohorts).drop(columns="note").to_html(index=False,float_format=lambda v:f"{v:.6f}")
    text+='<h2>Audit and reproducibility</h2><p>Saved predictions were independently rescored, protected source hashes checked, and inner-fold cutoff choices reconstructed from saved inner OOF predictions. Recent-expert checks verified that excluded gold was omitted and its matching groups purged from masked rows.</p><ul>'
    for name in STUDIES:
        text+=f'<li><a href="../{name}/report.html">{html.escape(name)}</a> — <a href="../{name}/protocol.json">frozen protocol</a></li>'
    text+='</ul><p><a href="verification.json">Detailed verification</a> · <a href="comparison.csv">All source and ensemble comparisons</a> · <a href="prediction_changes.csv">CV corrections and regressions</a> · <a href="chronological_prediction_changes.csv">Every changed chronological decision</a> · <a href="summary.json">Full machine-readable review</a></p><p>All labels and evaluation windows have influenced prior development. Repeat wins are descriptive, not independent significance tests. A new local gain, if present, is not a measured Kaggle gain.</p></body></html>'
    (HERE/"report.html").write_text(text)
    print(json.dumps({k:v for k,v in outcome.items() if k not in ["comparison","prediction_changes","chronological_prediction_changes","historical_year_cohorts"]},indent=2))


if __name__=="__main__":
    main()
