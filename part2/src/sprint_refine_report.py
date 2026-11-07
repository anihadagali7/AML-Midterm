"""Audit every refinement and compare shortlisted mixtures to scored winners."""
import json
import numpy as np
import pandas as pd
from benchmark_protocol import initialize,resolve,digest,DIRECTORY
from experiments import ROOT
from improvement_report import audit_track,paired_interval
from sprint_refine import current,NAMES,CONTROLS,RECIPES
from sprint_benchmark import DECISION
from sprint_report import table


def run():
    out=current();universe,protocol=initialize()
    gaps=json.loads((DECISION/"gaps/protocol.json").read_text())
    manifest=json.loads((out/"manifest.json").read_text())
    for file,sha in manifest["source_sha256"].items():assert digest(ROOT/file)==sha,file
    assert digest(ROOT/"REFINE_PROTOCOL.md")==manifest["policy_sha256"]
    assert digest(DIRECTORY/"protocol.json")==manifest["protocol_sha256"]
    assert digest(DECISION/"gaps/protocol.json")==manifest["gaps_sha256"]
    for item in manifest["parents"].values():assert digest(ROOT/item["path"]/"manifest.json")==item["manifest_sha256"]
    pairs=[];all_scores=[];count=0
    selected=["legacy0.2_tree0.1_trend0.1","legacy0.3_tree0.05_trend0.1","legacy0.3_sharp0.2"]
    for track in ["strict","published"]:
        for is_gap,spec in [(False,protocol),(True,gaps)]:
            folder=out/"gaps"/track if is_gap else out/track
            _,scores,frames=audit_track(folder,universe,spec,names=NAMES)
            count+=len(scores);all_scores.append(scores.assign(track=track,suite="gaps" if is_gap else "main"))
            for split in spec["splits"]:
                train,ev=resolve(universe,split)
                a=json.loads((folder/"audits"/f"{split['name']}.json").read_text())
                assert a["train_ids"]==train.ID.tolist() and a["eval_ids"]==ev.ID.tolist()
                assert not set(train.trial_id)&set(ev.trial_id)
                components={}
                aliases={"file10":"file10","legacy01":"legacy","sharp":"sharp","trend":"trend","recent_tree":"tree"}
                for file,sha in a["component_sha256"].items():
                    path=ROOT/file;assert digest(path)==sha
                    d=pd.read_csv(path)
                    for col in ["ID","trial_id","phase","label"]:assert d[col].tolist()==ev[col].tolist()
                    components[aliases[path.stem]]=d.probability.to_numpy()
                for name,recipe in (CONTROLS|RECIPES).items():
                    expected=components["file10"].copy()*(1-sum(recipe.values()))
                    for component,weight in recipe.items():expected+=weight*components[component]
                    actual=frames[split["name"]][name].probability.to_numpy()
                    assert np.max(np.abs(expected-actual))<4e-15,name
            for name in ["file13"]+selected:
                for reference in ["file11","file13"]:
                    if name==reference:continue
                    for kind in (["temporal"] if is_gap else ["temporal","group_cv"]):
                        pairs.append(dict(track=track,suite="gaps" if is_gap else "main",**paired_interval(frames,name,reference,kind)))
    pd.concat(all_scores,ignore_index=True).to_csv(out/"all_scores.csv",index=False)
    pair=pd.DataFrame(pairs);pair.to_csv(out/"paired_comparisons.csv",index=False)
    exported=[]
    for path in sorted((out/"exports").glob("*.json")):
        item=json.loads(path.read_text());file=ROOT/"submissions"/item["file"]
        assert digest(file)==item["sha256"] and item["native_replay_max_error"]<1e-12
        assert item["source_sha256"]==manifest["source_sha256"]
        assert item["benchmark_manifest_sha256"]==digest(out/"manifest.json")
        for dependency,sha in item["dependencies_sha256"].items():assert digest(ROOT/dependency)==sha
        data=pd.read_csv(file);sample=pd.read_csv(ROOT/"part2_sample_submission.csv")
        assert data.ID.equals(sample.ID) and data.ID.is_unique and len(data)==1500
        assert np.isfinite(data.label).all() and data.label.between(0,1,inclusive="neither").all()
        exported.append(item["file"])
    verification=dict(recipes=len(RECIPES),controls=len(CONTROLS),evaluations=count,
        same_fixed_splits=True,source_and_component_hashes_verified=True,
        independent_blend_replay=True,exports_verified=exported)
    (out/"verification.json").write_text(json.dumps(verification,indent=2)+"\n")
    ledger=json.loads((ROOT/"submissions/leaderboard_results.json").read_text())
    sections=["# Refinement after file 11 reached 0.58474",
        "The next reported score is **file 13: 0.58465**, a further 0.00009 improvement. The 19 new mixtures were declared before their evaluation. The recipes reuse the exact 13 fixed splits and both original preprocessing policies; no additional base models are fitted.",
        "The stronger 30% legacy weight gave only a small Kaggle gain over 20%. **File 15** retained 20% legacy ensemble and added 10% recent-weighted tree and 10% Ani phase trend; 60% remained file 10. It had the best combined chronological mean among candidates satisfying the declared adverse-gap limit under both policies, but scored **0.58492**, worse than file 13 by 0.00027. The improved historical averages did not transfer on this upload.",
        "The prepared **file 16** keeps the successful 30% legacy weight and replaces 20% of file 10 with the sharper-neighbor file 09, leaving 50% file 10. It improves both chronological means and the latest forward window versus file 13 in both tracks, with small regressions in grouped CV, official loss and the worst gap. It adds neither the recent-tree nor phase-trend endpoint tested by file 15. The separate nine-recipe final-slot batch also evaluated smaller interpolations and nearby tree/trend weights.",
        "**Final decision: hold file 16; do not spend the last slot on current evidence.** The user confirmed first place with nine submissions. The small chronological gains and intervals spanning zero do not give a strong case for this upload, and a single returned score would leave no follow-up slot. The earlier exploratory upload recommendation was withdrawn; the frozen local selection remains as historical evidence, with the later decision in submission_decision.json. All local labels have been consulted repeatedly; bootstrap intervals do not correct for adaptive selection. File 13 at 0.58465 remains the best scored submission."]
    for track in ["strict","published"]:
        d=pd.read_csv(out/f"summary_{track}.csv").set_index("model")
        sections.append("## "+track.capitalize()+"\n\n"+table(d.loc[["file10","file11","file13","file14"]+selected,["official","temporal","gap_mean","gap_worst","cv","latest_delta"]].reset_index()))
    sections.append("## Paired differences from file 13\n\n"+table(pair[pair.track.eq("strict")&pair.reference.eq("file13")][["model","suite","kind","delta","lower","upper"]]))
    score_lines=[]
    for filename in ["11_file10_legacy20.csv","13_file10_legacy30.csv"]+exported:
        record=ledger["submissions"].get(filename)
        score_lines.append(f"- `{filename}`: "+(f"{record['kaggle_score']:.5f} (user-reported)" if record else "unknown Kaggle score"))
    sections.append("## Submission evidence\n\n"+"\n".join(score_lines))
    sections.append("[Every recipe, strict](summary_strict.csv), [published](summary_published.csv), [every period and phase](all_scores.csv), [paired uncertainty](paired_comparisons.csv), [verification](verification.json), and [frozen recipes/source hashes](manifest.json) retain the complete record. Receipts preserve native dependency hashes and their pre-submission score state. See the live queue and ledger for later observations.")
    sections.append("## Reproduce\n\n```sh\n.venv/bin/python part2/sprint_refine.py benchmark\n.venv/bin/python part2/sprint_refine.py export --model legacy0.2_tree0.1_trend0.1 --filename 15_file10_legacy20_tree10_trend10.csv\n.venv/bin/python part2/sprint_refine_report.py\n```\n\nExisting native component caches are required for export; the earlier benchmark workflows reproduce them. All supervised final components use 9,050 labeled rows. No new Kaggle upload is performed by these commands.")
    (out/"RESULTS.md").write_text("\n\n".join(sections)+"\n")
    print(json.dumps(verification,indent=2),flush=True)


if __name__=="__main__":run()
