"""Render the live submission record without changing frozen experiment evidence."""
import json
from pathlib import Path
import pandas as pd
from benchmark_protocol import digest
from experiments import ROOT
from sprint_benchmark import OUT,current as initial_current
from sprint_tree_export import current as tree_current
from sprint_refine import current as refine_current
from sprint_finish import current as finish_current


def run():
    ledger=json.loads((ROOT/"submissions/leaderboard_results.json").read_text())
    queue=json.loads((ROOT/"submissions/sprint_queue.json").read_text())
    for file,r in ledger["submissions"].items():assert digest(ROOT/"submissions"/file)==r["sha256"]
    best=ledger["current_best"]
    mapping=[(10,"10_ani_fast2_neighbors_half.csv",initial_current(),"file10"),
        (11,"11_file10_legacy20.csv",initial_current(),"mix_legacy01_0.2"),
        (12,"12_file10_sharp_trend.csv",initial_current(),"mix_sharp_trend"),
        (13,"13_file10_legacy30.csv",initial_current(),"mix_legacy01_0.3"),
        (14,"14_file10_recent_tree20.csv",tree_current(),"tree20"),
        (15,"15_file10_legacy20_tree10_trend10.csv",refine_current(),"legacy0.2_tree0.1_trend0.1"),
        (16,"16_file10_legacy30_sharp20.csv",refine_current(),"legacy0.3_sharp0.2")]
    rows=[]
    for number,file,path,model in mapping:
        r=pd.read_csv(path/"summary_strict.csv").set_index("model").loc[model]
        score=ledger["submissions"].get(file,{}).get("kaggle_score")
        rows.append(f"| {number} | {r.temporal:.6f} | {r.gap_mean:.6f} | {r.gap_worst:.6f} | {r.cv:.6f} | "+(f"{score:.5f}" if score is not None else "Unknown")+" |")
    sections=["# Part Two submission sprint",
        f"**Best scored submission: `{best['submission']}`, {best['kaggle_score']:.5f}.** Scores are user-reported and tied to exact CSV hashes in the [ledger](../../submissions/leaderboard_results.json). File 11 reached reported first place at 0.58474; file 13 subsequently improved to 0.58465.",
        f"{queue['confirmed_additional_uploads']} of the original four additional uploads were used; the remaining {queue['remaining_from_original_budget']} submission slot was deliberately left unused. The [live queue](../../submissions/sprint_queue.json) records the final hold decision. The user revised the active deadline to 2026-11-07 00:06:49 UTC.",
        "## Direct comparison\n\nAll entries use the same ten original splits plus three longer-gap checks, equal-phase log loss, and pooled repeated grouped CV. Lower is better. These are the primary strict-preprocessing results; both preprocessing tracks and every period remain in the linked reports.\n\n| File | Forward mean | Longer-gap mean | Worst gap | Grouped CV | Kaggle |\n| --- | ---: | ---: | ---: | ---: | ---: |\n"+"\n".join(rows),
        "## What the feedback changed\n\nFile 11's 20% original-ensemble contribution improved file 10 by 0.00135 on Kaggle. Increasing it to 30% in file 13 gained another 0.00009, showing a much smaller incremental benefit. File 15's recent-tree/phase-trend mixture improved local averages but scored 0.58492, losing 0.00027 to file 13. That result argues against increasing those additions solely because historical validation favors them.",
        "**Hold the final remaining submission.** The user confirmed first place with nine total uploads. The prepared file 16 retains 30% original ensemble, replaces 20% of file 10 with sharper-neighbor file 09, and leaves 50% file 10. Both chronological averages improve versus file 13, but the gains are small, intervals span zero, and grouped CV worsens slightly. Its initial exploratory upload recommendation was withdrawn after reviewing the value of the final slot. File 16 remains an unsubmitted reserve; file 13's exact CSV remains intact.",
        "## Experiments retained\n\n"+"\n".join([
            f"- [Initial 32-recipe batch]({initial_current().relative_to(OUT)}/RESULTS.md): 1,014 evaluations including controls.",
            f"- [Four recent-tree mixtures]({tree_current().relative_to(OUT)}/RESULTS.md): 182 evaluations including controls.",
            f"- [19 refinements around files 11 and 13]({refine_current().relative_to(OUT)}/RESULTS.md): 598 evaluations including controls and the final submission choice.",
            f"- [Nine final-slot alternatives]({finish_current().relative_to(OUT)}/RESULTS.md): 312 evaluations including controls; no candidate from this batch was submitted."]),
        "All local labels were repeatedly consulted. Paired bootstrap intervals share trial weights across phases, but do not correct for adaptive model selection. Later Kaggle results are separate observations; they do not retroactively make local comparisons independent.",
        "## Verification and reproduction\n\nExports preserve sample order, 1,500 unique IDs, valid probabilities and exact native inference after evaluation labels/IDs are replaced. All supervised final components use 9,050 labeled rows. Native caches are ignored by Git; source hashes, held-out predictions, component hashes and export receipts are retained. Frozen export receipts keep their pre-submission score state; the live ledger records later scores.\n\n```sh\nmake -C part2 sprint\nmake -C part2 sprint-final\nmake -C part2 sprint-check\nmake -C part2 sprint-refine\nmake -C part2 sprint-refine-check\n```\n\nA fresh checkout must first reproduce the earlier all-submission and final-neighbor native caches. None of these commands uploads to Kaggle."]
    (OUT/"RESULTS.md").write_text("\n\n".join(sections)+"\n")
    print("Updated live sprint overview:",best["submission"],best["kaggle_score"])


if __name__=="__main__":run()
