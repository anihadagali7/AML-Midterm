"""Verify and retain all final-slot alternatives, including rejected directions."""
import json
import numpy as np
import pandas as pd
from benchmark_protocol import initialize,resolve,digest,DIRECTORY
from experiments import ROOT
from improvement_report import audit_track
from sprint_finish import current,NAMES,CONTROLS,RECIPES
from sprint_benchmark import DECISION
from sprint_report import table


def run():
    out=current();universe,protocol=initialize();gaps=json.loads((DECISION/"gaps/protocol.json").read_text())
    manifest=json.loads((out/"manifest.json").read_text())
    for file,sha in manifest["source_sha256"].items():assert digest(ROOT/file)==sha
    assert digest(ROOT/"FINISH_PROTOCOL.md")==manifest["policy_sha256"]
    assert digest(DIRECTORY/"protocol.json")==manifest["protocol_sha256"]
    assert digest(DECISION/"gaps/protocol.json")==manifest["gaps_sha256"]
    assert digest(ROOT/manifest["parent"]/"manifest.json")==manifest["parent_manifest_sha256"]
    scores_all=[];count=0
    aliases={"file10":"file10","legacy01":"legacy","sharp":"sharp","trend":"trend","recent_tree":"tree"}
    for track in ["strict","published"]:
        summaries=[]
        for is_gap,spec in [(False,protocol),(True,gaps)]:
            folder=out/"gaps"/track if is_gap else out/track
            summary,scores,frames=audit_track(folder,universe,spec,names=NAMES)
            summaries.append(summary);count+=len(scores)
            scores_all.append(scores.assign(track=track,suite="gaps" if is_gap else "main"))
            for split in spec["splits"]:
                train,ev=resolve(universe,split)
                a=json.loads((folder/"audits"/f"{split['name']}.json").read_text())
                assert a["train_ids"]==train.ID.tolist() and a["eval_ids"]==ev.ID.tolist()
                assert not set(train.trial_id)&set(ev.trial_id)
                components={}
                for file,sha in a["component_sha256"].items():
                    path=ROOT/file;assert digest(path)==sha
                    d=pd.read_csv(path)
                    for col in ["ID","trial_id","phase","label"]:assert d[col].tolist()==ev[col].tolist()
                    components[aliases[path.stem]]=d.probability.to_numpy()
                for name,recipe in (CONTROLS|RECIPES).items():
                    expected=(1-sum(recipe.values()))*components["file10"].copy()
                    for key,w in recipe.items():expected+=w*components[key]
                    assert np.max(np.abs(expected-frames[split["name"]][name].probability))<4e-15
        combined=summaries[0].join(summaries[1][["temporal","worst"]].rename(columns=dict(temporal="gap_mean",worst="gap_worst")))
        saved=pd.read_csv(out/f"summary_{track}.csv").set_index("model")
        assert np.allclose(combined.sort_index(),saved.loc[combined.index,combined.columns].sort_index(),rtol=0,atol=1e-12,equal_nan=True)
    pd.concat(scores_all,ignore_index=True).to_csv(out/"all_scores.csv",index=False)
    verification=dict(recipes=len(RECIPES),controls=len(CONTROLS),evaluations=count,
        same_fixed_splits=True,source_and_component_hashes_verified=True,independent_blend_replay=True,
        selected_for_export=None,reason="File 15 lost on Kaggle. A separate sharper-neighbor candidate was prepared, then held because evidence did not justify the final submission slot.")
    (out/"verification.json").write_text(json.dumps(verification,indent=2)+"\n")
    sections=["# Final-slot alternatives",
        "Nine fixed variations were declared while file 15's score was pending. Its **0.58492** result arrived before these candidates were evaluated and is explicitly recorded in the manifest. File 13 remains better at **0.58465**. No base models were refitted.",
        "Local loss continues to favor stronger tree/trend mixtures, even though the tested direction worsened Kaggle. The independent sharper-neighbor file 16 was prepared from the preceding refinement batch, but the final decision is to hold the remaining slot because the evidence is weak. None of this batch's candidates was exported or submitted.",
        "The smaller interpolations between files 13 and 15 improve local averages versus file 13, but their local averaging benefit relative to the weighted endpoint losses is only about 0.00003–0.00005 on the chronological means. This is evidence for the decision to explore a different direction after file 15's 0.00027 Kaggle loss; it is not a prediction of the unknown interpolation scores. All candidate results remain in the tables below."]
    for track in ["strict","published"]:
        d=pd.read_csv(out/f"summary_{track}.csv")
        sections.append("## "+track.capitalize()+"\n\n"+table(d[["model","official","temporal","gap_mean","gap_worst","cv","latest"]]))
    sections.append("[Every period](all_scores.csv), [interpolation diagnostics](interpolation_diagnostics.csv), [verification](verification.json) and [frozen manifest](manifest.json). The same historical labels were reused; these are adaptive comparisons, not independent validation.")
    sections.append("## Reproduce\n\n```sh\n.venv/bin/python part2/sprint_finish.py benchmark\n.venv/bin/python part2/sprint_finish_report.py\n```\n\nThis batch only evaluates local predictions. It neither exports nor uploads a Kaggle submission.")
    (out/"RESULTS.md").write_text("\n\n".join(sections)+"\n")
    print(json.dumps(verification,indent=2),flush=True)


if __name__=="__main__":run()
