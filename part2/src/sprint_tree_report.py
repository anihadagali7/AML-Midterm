"""Independent audit and paired comparison of the fixed recency-tree batch."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark_protocol import initialize,resolve,digest
from experiments import ROOT
from improvement_report import audit_track,paired_interval
from sprint_tree import NAMES,RECIPES
from sprint_benchmark import OUT,OLD_ALL,DECISION
from sprint_tree_export import current
from sprint_report import table


def run():
    out=current();universe,protocol=initialize();gaps=json.loads((DECISION/"gaps/protocol.json").read_text())
    manifest=json.loads((out/"manifest.json").read_text())
    for file,sha in manifest["source_sha256"].items():assert digest(ROOT/file)==sha
    assert digest(ROOT/"SPRINT_TREE_PROTOCOL.md")==manifest["policy_sha256"]
    scores_all=[];pairs=[];count=0
    for track in ["strict","published"]:
        for is_gap,spec in [(False,protocol),(True,gaps)]:
            folder=out/"gaps"/track if is_gap else out/track
            _,scores,frames=audit_track(folder,universe,spec,names=NAMES)
            scores_all.append(scores.assign(track=track,suite="gaps" if is_gap else "main"));count+=len(scores)
            for split in spec["splits"]:
                train,ev=resolve(universe,split);a=json.loads((folder.parent/"audits"/f"{split['name']}.json").read_text())
                assert a["train_ids"]==a["preprocessing_fit_ids"]==train.ID.tolist()
                assert a["eval_ids"]==ev.ID.tolist()
                assert not set(train.trial_id)&set(ev.trial_id)
                cache=OUT/"cache"/out.name/f"{split['name']}.joblib"
                assert digest(cache)==a["tree_model_sha256"]
                parent=OLD_ALL/"gaps"/track if is_gap else OLD_ALL/track
                legacy={}
                for number in ["01_balanced_ensemble","02_text_heavy"]:
                    d=pd.read_csv(parent/"predictions"/split["name"]/f"{number}.csv")
                    assert d.ID.tolist()==ev.ID.tolist();legacy[number]=d.probability.to_numpy()
                linear=2*legacy["02_text_heavy"]-legacy["01_balanced_ensemble"]
                p={name:frames[split["name"]][name].probability.to_numpy() for name in NAMES}
                assert np.allclose(p["file11"],.8*p["file10"]+.2*legacy["01_balanced_ensemble"],rtol=0,atol=4e-15)
                for name,r in RECIPES.items():
                    expected=(1-r["tree"]-r["linear"])*p["file10"]+r["tree"]*p["recent_tree"]+r["linear"]*linear
                    assert np.allclose(expected,p[name],rtol=0,atol=4e-15),name
            for ref in ["file10","file11"]:
                for kind in (["temporal"] if is_gap else ["temporal","group_cv"]):
                    pairs.append(dict(track=track,suite="gaps" if is_gap else "main",**paired_interval(frames,"tree20",ref,kind)))
    pd.concat(scores_all,ignore_index=True).to_csv(out/"all_scores.csv",index=False)
    paired=pd.DataFrame(pairs);paired.to_csv(out/"paired_comparisons.csv",index=False)
    exports=[]
    for path in sorted((out/"exports").glob("*.json")):
        item=json.loads(path.read_text());file=ROOT/"submissions"/item["file"]
        assert digest(file)==item["sha256"] and item["native_replay_max_error"]<1e-12
        data=pd.read_csv(file);sample=pd.read_csv(ROOT/"part2_sample_submission.csv")
        assert data.ID.equals(sample.ID) and data.ID.is_unique and len(data)==1500
        assert np.isfinite(data.label).all() and data.label.between(0,1,inclusive="neither").all()
        exports.append(item["file"])
    result=dict(new_tree_settings=1,blend_recipes=len(RECIPES),controls=3,evaluations=count,
        identical_fixed_splits=True,training_only_preprocessing_verified=True,source_hashes_match=True,
        independent_blend_replay=True,exports_verified=exports)
    (out/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    text=["# Recent-weighted tree follow-up",
        "One fixed recency-weighted CatBoost model is fitted per original split. Four prespecified probability blends are compared to files 10 and 11. No settings were adjusted inside this batch. This is adaptive follow-up research using previously consulted labels.",
        "File 14 (`tree20`) is 80% file 10 plus 20% the new depth-6 tree. Its stronger standard forward mean and latest-period performance come with a regression on the worst individual longer-gap window. Compare both the average and that adverse period before interpreting the result. The Kaggle score is unknown until reported."]
    for track in ["strict","published"]:
        d=pd.read_csv(out/f"summary_{track}.csv")
        text.append("## "+track.capitalize()+"\n\n"+table(d[["model","official","temporal","gap_mean","gap_worst","cv"]]))
    text.append("## Paired uncertainty\n\nIntervals do not correct for the adaptive research history.\n\n"+table(paired[paired.track.eq("strict")][["reference","suite","kind","delta","lower","upper"]]))
    text.append("[Every period](all_scores.csv), [verification](verification.json), and [frozen settings](manifest.json) retain the complete record. Final exports use all 9,050 labeled rows and preserve the 0.58609 baseline file.")
    (out/"RESULTS.md").write_text("\n\n".join(text)+"\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":run()
