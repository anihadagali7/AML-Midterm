"""Audit sprint results and report comparisons against the scored file 10."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark_protocol import initialize, resolve, digest
from experiments import ROOT
from improvement_report import audit_track, paired_interval
from sprint_benchmark import current, DECISION, CONTROLS
from sprint_models import RECIPES


def table(df):
    def cell(x):
        return f"{x:.6f}" if isinstance(x,(float,np.floating)) else str(x)
    return "\n".join(["| "+" | ".join(df.columns)+" |", "| "+" | ".join(["---"]*len(df.columns))+" |"]+
        ["| "+" | ".join(map(cell,row))+" |" for row in df.itertuples(index=False,name=None)])


def run():
    out=current();universe,protocol=initialize();gaps=json.loads((DECISION/"gaps/protocol.json").read_text())
    manifest=json.loads((out/"manifest.json").read_text())
    for file,sha in manifest["source_sha256"].items():assert digest(ROOT/file)==sha
    assert digest(ROOT/"SPRINT_PROTOCOL.md")==manifest["policy_sha256"]
    names=CONTROLS+list(RECIPES);checked=0;intervals=[];all_scores=[]
    selected=json.loads((out/"selection.json").read_text())
    compare=[selected["first_candidate"],selected["reserve_orthogonal"],selected["reserve_if_first_improves"]]
    for track in ["strict","published"]:
        for is_gap,spec in [(False,protocol),(True,gaps)]:
            folder=out/"gaps"/track if is_gap else out/track
            _,scores,frames=audit_track(folder,universe,spec,names=names)
            checked+=len(scores);all_scores.append(scores.assign(track=track,suite="gaps" if is_gap else "main"))
            for split in spec["splits"]:
                train,ev=resolve(universe,split)
                a=json.loads((folder/"audits"/f"{split['name']}.json").read_text())
                assert a["train_ids"]==train.ID.tolist() and a["eval_ids"]==ev.ID.tolist()
                for field in ["old_fold_audit","extra_fold_audit"]:
                    for f in a[field]:
                        hold=f.get("heldout_ids",f.get("eval_ids"));fit=f["fit_ids"]
                        indexed=train.set_index("ID")
                        assert not set(indexed.loc[fit,"trial_id"])&set(indexed.loc[hold,"trial_id"])
                        assert not set(fit)&set(ev.ID)
                        if track=="strict":assert set(f["preprocessing_fit_ids"])==set(fit)
            for name in compare:
                for kind in (["temporal"] if is_gap else ["temporal","group_cv"]):
                    intervals.append(dict(track=track,suite="gaps" if is_gap else "main",**paired_interval(frames,name,"file10",kind)))
    pd.concat(all_scores,ignore_index=True).to_csv(out/"all_scores.csv",index=False)
    pairs=pd.DataFrame(intervals);pairs.to_csv(out/"paired_comparisons.csv",index=False)
    exports=[]
    for path in sorted((out/"exports").glob("*.json")):
        item=json.loads(path.read_text());p=ROOT/"submissions"/item["file"]
        assert digest(p)==item["sha256"] and item["native_replay_max_error"]<1e-12
        d=pd.read_csv(p);sample=pd.read_csv(ROOT/"part2_sample_submission.csv")
        assert d.ID.equals(sample.ID) and d.ID.is_unique and len(d)==1500
        assert np.isfinite(d.label).all() and d.label.between(0,1,inclusive="neither").all()
        exports.append(item)
    verification=dict(recipes=len(RECIPES),controls=len(CONTROLS),evaluations=checked,
        same_fixed_splits=True,grouped_inner_boundaries_verified=True,source_hashes_match=True,
        exports_verified=len(exports),baseline_kaggle_score=.58609)
    (out/"verification.json").write_text(json.dumps(verification,indent=2)+"\n")
    ledger=json.loads((ROOT/"submissions/leaderboard_results.json").read_text())
    result=ledger["submissions"].get("11_file10_legacy20.csv")
    score_note=(f"File 11 scored **{result['kaggle_score']:.5f}**, with user-reported rank {result['reported_rank']}. "
        "Its exact hash is recorded in the current ledger; the export receipt retains its pre-submission state. "
        "The live queue and ledger track subsequent uploads, including file 13's later score."
        if result else "No new file has a known score until the user returns a Kaggle result.")
    sections=["# Iteration after file 10 scored 0.58609",
        "File 10 is this round's frozen baseline; these are adaptive local experiments. "+score_note,
        "## First candidate\n\nFile 11 uses **80% file 10 + 20% the original balanced ensemble**, with every final component trained on all 9,050 labeled rows. The 30% mixture has the best local means, but 20% limits regression on the latest forward window and worst gap. Use the first Kaggle result before increasing this weight. File 12 is a distinct reserve: 50% file 10, 25% sharper neighbors (file 09), and 25% Ani's Phase III trend control.",
        "The first batch freezes 32 proposals: five fast-neighbor blend weights, complementary mixtures, and ten small meta-model changes. Every recipe uses the same ten original splits plus three longer-gap checks. Strict preprocessing is the primary comparison; the published-recipe sensitivity track preserves each component's original feature policy. Equal-phase log loss is lower-is-better."]
    for track in ["strict","published"]:
        summary=pd.read_csv(out/f"summary_{track}.csv").set_index("model")
        view=summary.loc[["file10"]+compare,["official","temporal","gap_mean","gap_worst","cv"]].reset_index()
        sections.append("## "+track.capitalize()+"\n\n"+table(view))
    sections += ["## Paired uncertainty\n\nThe historical evaluation labels have been reused. Intervals are descriptive and do not correct for adaptive model selection. No local point improvement guarantees a better Kaggle score.\n\n"+table(pairs[pairs.track.eq("strict")][["model","suite","kind","delta","lower","upper"]]),
        "## Artifacts\n\n- [All strict results](summary_strict.csv)\n- [All published-recipe results](summary_published.csv)\n- [Every period and phase](all_scores.csv)\n- [Frozen selection](selection.json)\n- [Verification](verification.json)\n- [Recipe/source manifest](manifest.json)",
        "## Reproduce\n\n```sh\n.venv/bin/python part2/sprint_benchmark.py --track strict\n.venv/bin/python part2/sprint_benchmark.py --track published\n.venv/bin/python part2/sprint_benchmark.py --track strict --summarize\n.venv/bin/python part2/sprint_report.py\n```\n\nNative caches are ignored by Git. Export receipts record exact CSV hashes, recipes, training populations and native replay errors. Original submissions remain unchanged."]
    (out/"RESULTS.md").write_text("\n\n".join(sections)+"\n")
    print(json.dumps(verification,indent=2),flush=True)


if __name__=="__main__":run()
