"""Write the extension findings and an executed-notebook source bundle."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
from experiments import ROOT
from extension_common import OUT,REFERENCES,current
from extension_followthrough import DECISION,selection
from extension_meta import RECIPES as META_RECIPES
from extension_neighbors import RECIPES as NN_RECIPES


def table(df):
    headers=list(df.columns)
    rows=[[f"{v:.6f}" if isinstance(v,(float,np.floating)) else str(v) for v in row] for row in df.to_numpy()]
    return "\n".join(["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |"]+
                     ["| "+" | ".join(row)+" |" for row in rows])


def build():
    decision=selection();names=REFERENCES+[x["model"] for x in decision["shortlist"]]
    manifest=json.loads((DECISION/"final_manifest.json").read_text())
    verification=json.loads((DECISION/"verification.json").read_text())
    ledger=json.loads((ROOT/"submissions/leaderboard_results.json").read_text())
    score_labels={}
    for item in manifest["submissions"]:
        record=ledger["submissions"].get(item["file"])
        if record: assert record["sha256"]==item["sha256"]
        score_labels[item["file"]]=f"{record['kaggle_score']:.5f} (user-reported)" if record else "unknown"
    score_note=("**File 10 Kaggle result: "+score_labels["10_ani_fast2_neighbors_half.csv"]+
        ".** File 09's Kaggle score is "+score_labels["09_ani_replace_knn5.csv"]+
        ". The current ledger records results reported after the frozen export manifests.")
    primary="fast2_neighbors_half";sharp="replace_knn5"
    sources={t:pd.read_csv(DECISION/t/"summary.csv").set_index("model") for t in ["strict","published"]}
    gaps={t:pd.read_csv(DECISION/"gaps"/t/"summary.csv").set_index("model") for t in sources}
    paired=pd.read_csv(DECISION/"paired_comparisons.csv")
    figures=DECISION/"figures";figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    for ax,track in zip(axes,sources):
        normal=pd.read_csv(DECISION/track/"fold_scores.csv")
        normal=normal[normal.kind.eq("temporal")].pivot(index="split",columns="model",values="phase_mean_logloss")
        gap=pd.read_csv(DECISION/"gaps"/track/"fold_scores.csv").pivot(index="split",columns="model",values="phase_mean_logloss")
        d=pd.concat([normal,gap])
        for name,color in [(primary,"#12775d"),(sharp,"#7851a9")]:
            ax.plot(np.arange(len(d)),d[name]-d.ani_stack,marker="o",label=name,color=color)
        ax.axhline(0,color="#666666",linewidth=1)
        ax.axvline(2.5,color="#cccccc",linestyle="--")
        ax.set_xticks(np.arange(6),["08–09","10–11","12–13","Gap 08–09","Gap 10–11","Gap 12–13"],rotation=20)
        ax.set(title=track.capitalize()+" preprocessing",ylabel="Log-loss difference from Ani (negative is better)")
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    fig.savefig(figures/"chronological_differences.png",dpi=180);plt.close(fig)
    sections=["# New Part Two candidates: recent, sharper neighbors",
        score_note,
        "File 10 improves all six chronological windows under both preprocessing policies, with a small grouped-CV improvement over Ani's 0.58688 model. Its reported Kaggle score of 0.58609 improves that reference by 0.00079. The user reported second place behind 0.58526, a remaining gap of 0.00083. Use file 10 as the next experiment's reference. File 09 has the strongest improvement on the original three forward windows, but a small grouped-CV regression and no reported Kaggle score.",
        "## What the earlier search missed\n\nThe first round tested slower recency and many calibration changes while retaining Ani's nearest-neighbor estimator. That estimator ignores recency, uses linear text similarities, and shrinks toward a fixed 50% success rate. The previous promotion rule also demanded near-best random CV from the unweighted model, despite its weaker chronological and reported Kaggle results. This round preserves that history and uses a separately declared rule that prioritizes later-cohort prediction. No old result was relabeled as new evidence.",
        "## Changes that helped\n\n**File 09** retains Ani's seven other base estimators and replaces only the kNN input before refitting his logistic stack. For the 50 most similar trials in the same phase, it squares similarity relative to the closest match, weights recent trials more strongly (exponential tau=3), and shrinks toward a smoothed training-only success prior for that phase. All neighbor labels, priors, vocabularies and IDF values are rebuilt inside each grouped inner fold.\n\n**File 10** averages Ani's original stack with a tau=2 stack that also receives eight neighborhood estimates: several neighborhood sizes, recent weighting, sharper text similarity, and combined criteria/disease/drug/ICD similarity. The probability blend is fixed at 50/50. Its OOF meta-model is regularized with C=0.3.",
        "For file 09, neighbor weights are `(similarity / best_similarity)^2 × recency_weight`. Recency weights have mean one within the fitting phase. The phase prior is `(weighted_successes + 1) / (total_weight + 2)`, and the neighbor probability is `(weighted_neighbor_successes + 2 × prior) / (neighbor_weight + 2)`. With no overlapping text, prediction falls back to the fitted phase prior.",
        "## Identical-split comparison\n\nLower log loss is better. All tables score the exact same held-out trials, with equal phase weighting. Random grouped CV is pooled within each repeat. `official` is the supplied validation set; `temporal` averages the three original forward windows; `worst` is the worst forward window."]
    for track in sources:
        sections.append("### "+track.capitalize()+"\n\n"+table(sources[track].loc[names,["official","temporal","worst","cv"]].reset_index()))
    sections.append("Both proposals improve the original forward windows against the recency winner in all three periods. The supplied validation score worsens, and file 09's CV worsens. Those trade-offs are retained in the table rather than hidden. Ani's unweighted winner still has lower grouped-CV loss than these candidates; its chronological loss is higher.")
    sections.append("## Frozen-shortlist checks at larger time gaps\n\nThe two recipes were frozen before scoring their predictions on the three larger-gap tests: train through 2005/evaluate 2008–2009, through 2007/evaluate 2010–2011, and through 2009/evaluate 2012–2013. These labels were used in earlier research; the checks are robustness evidence, not an untouched independent test.")
    for track in gaps:
        sections.append("### "+track.capitalize()+"\n\n"+table(gaps[track].loc[names,["temporal","worst"]].reset_index()))
    sections.append("File 10 wins all three additional windows in both tracks. File 09 improves their average but slightly worsens the first gap. Ani's unsubmitted Phase III trend control remains better on the *average of these longer-gap tests*; the new candidates are better on the original forward windows. This prevents claiming dominance over every alternative.")
    sections.append("![Window-by-window differences](decision/figures/chronological_differences.png)")
    sections.append("## Uncertainty\n\nPaired trial bootstraps share weights across phase rows and average repeated-CV losses before resampling. The chronological confidence intervals versus Ani include zero. The local intervals alone do not establish a Kaggle improvement; the later reported score above is separate evidence. Selection across this and earlier research is adaptive, and the intervals do not correct for it.\n\n"+table(paired[(paired.track=="strict")&(paired.reference=="ani_stack")][["model","kind","delta","lower","upper"]]))
    sections.append("All paired comparisons, including against the earlier 0.58810 winner, are in [paired_comparisons.csv](decision/paired_comparisons.csv). Larger-gap intervals are stored separately under each track's `gaps` directory.")
    phases=pd.read_csv(DECISION/"phase_metrics.csv")
    sections.append("## Required per-phase and stopping-reason diagnostics\n\nSupplied-validation metrics for the recommended candidate and Ani's recency model, under strict preprocessing:\n\n"+
        table(phases[(phases.track=="strict")&phases.model.isin(["ani_stack",primary])].drop(columns="track")))
    reasons=pd.read_csv(DECISION/"reason_errors.csv")
    a=reasons[(reasons.track=="strict")&(reasons.model=="ani_stack")]
    b=reasons[(reasons.track=="strict")&(reasons.model==primary)]
    keys=["phase","reason","n","true_reason_rows","predicted_reason_rows"]
    combined=a[keys+["false_success"]].merge(b[keys+["false_success"]],on=keys,suffixes=("_ani","_candidate"))
    sections.append("Counts of unsuccessful validation trials incorrectly predicted successful at cutoff 0.5:\n\n"+table(combined))
    sections.append("Reason tags are joined only after predicting. Visible true tags and frozen Part One predictions are distinguished; categories overlap and small groups are uncertain. Full rates and Wilson intervals are in [reason_errors.csv](decision/reason_errors.csv). No identifiers, stopping-reason labels or reason embeddings enter a predictor.")
    total=len(META_RECIPES)+len(NN_RECIPES)
    sections.append(f"## Reproduction and complete record\n\nThis round tested **{total} extension recipes**, each on ten frozen splits under two preprocessing policies: **{verification['new_recipe_evaluations']} new recipe evaluations**, plus unchanged controls. All proposals, including failures, are in [all_comparisons.csv](decision/all_comparisons.csv). The shallow nonlinear stacks did not improve temporal loss; faster recency alone also produced mixed results. The neighbor changes supply the useful gain.\n\n[Selection and source manifests](decision/selection.json) preserve recipe definitions, hashes and source runs. [Verification](decision/verification.json) checks fitting boundaries, all saved scores, and unchanged controls. Native model replay has maximum probability difference {verification['native_replay_max_error']:.2g}; changing evaluation labels/IDs does not alter inference. All 16 unit checks passed.")
    sections.append("Final fits use all 9,050 labeled rows and the same feature-only transductive preprocessing as Ani. Each CSV has exactly 1,500 unique IDs in sample-submission order, finite probabilities, and columns `ID,label`. The original Ani files remain intact.\n\n"+
        "\n".join(f"- [{r['file']}](../../submissions/{r['file']}): Kaggle score {score_labels[r['file']]}; SHA-256 `{r['sha256']}`." for r in manifest["submissions"]))
    commands="""```sh
# From repository root, with part2/requirements-benchmark.txt installed in .venv:
.venv/bin/python part2/extension_meta.py --track strict
.venv/bin/python part2/extension_meta.py --track published
.venv/bin/python part2/extension_neighbors.py --track strict
.venv/bin/python part2/extension_neighbors.py --track published
.venv/bin/python part2/extension_followthrough.py select
.venv/bin/python part2/extension_followthrough.py compare
.venv/bin/python part2/extension_followthrough.py gaps --track strict
.venv/bin/python part2/extension_followthrough.py gaps --track published
.venv/bin/python part2/extension_followthrough.py final
.venv/bin/python part2/extension_verify.py
.venv/bin/python -m unittest discover -s part2 -p 'test_*.py'
.venv/bin/python part2/extension_predict.py --input part2/outcome_test.csv --output /tmp/replayed.csv
```"""
    sections.append(commands+"\n\nNative caches are regenerable and ignored by Git. Existing OOF arrays and fitting IDs permit replay without retraining the original base models. If the old final native cache is missing, final fitting rebuilds it. [Notebook with embedded implementation source](../../ani_extensions.ipynb).")
    (OUT/"RESULTS.md").write_text("\n\n".join(sections)+"\n")
    notebook=nbformat.v4.new_notebook()
    notebook.metadata.kernelspec=dict(name="part2",display_name="Python (Part Two)",language="python")
    notebook.cells=[nbformat.v4.new_markdown_cell("# Extending Ani's outcome stack\n\n"+score_note+" Local chronological intervals include zero. Training code is embedded below; default execution reads the saved results."),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\nroot = Path.cwd()\nif (root / 'part2').exists(): root /= 'part2'\nout = root / 'experiments/ani_extensions'\ndecision = out / 'decision'\ndisplay(json.loads((decision / 'selection.json').read_text()))"),
        nbformat.v4.new_code_cell("for track in ['strict','published']:\n    print(track)\n    display(pd.read_csv(decision / track / 'summary.csv').round(6))\n    print('Larger time gaps')\n    display(pd.read_csv(decision / 'gaps' / track / 'summary.csv').round(6))\ndisplay(Image(filename=str(decision / 'figures/chronological_differences.png')))"),
        nbformat.v4.new_code_cell("pairs=pd.read_csv(decision / 'paired_comparisons.csv')\ndisplay(pairs[(pairs.track=='strict') & (pairs.reference=='ani_stack')].round(6))\ndisplay(pd.read_csv(decision / 'phase_metrics.csv').round(6))\ndisplay(pd.read_csv(decision / 'reason_errors.csv').round(4))\ndisplay(json.loads((decision / 'verification.json').read_text()))\ndisplay(json.loads((decision / 'final_manifest.json').read_text()))"),
        nbformat.v4.new_markdown_cell("## Full implementation source\n\nThe source bundle below includes the estimators, feature extraction, benchmark orchestration, inference and verification code. It is stored as strings so normal execution does not launch training. The repository supplies CSVs, frozen splits and source-checksummed previous OOF arrays. No external labeled data is retrieved."),
        nbformat.v4.new_code_cell("SOURCES = {}")]
    files=["ani_models.py","experiments.py","benchmark_protocol.py","benchmark_models.py","local_benchmark.py","improvement_models.py","improvement_benchmark.py",
           "improvement_report.py","error_analysis.py","extension_common.py","extension_meta.py","extension_neighbors.py",
           "extension_followthrough.py","extension_predict.py","extension_verify.py","test_extensions.py","extension_report.py"]
    for filename in files:
        source=(ROOT/filename).read_text()
        if "'''" not in source:
            code=f"# {filename}\nSOURCES[{filename!r}] = r'''"+source+"'''"
        else:
            code=f"# {filename}\nSOURCES[{filename!r}] = (\n"+"\n".join(json.dumps(line,ensure_ascii=False) for line in source.splitlines(keepends=True))+"\n)"
        notebook.cells.append(nbformat.v4.new_code_cell(code))
    notebook.cells.append(nbformat.v4.new_code_cell("for filename, source in SOURCES.items():\n    assert source == (root / filename).read_text(), filename\n    compile(source, filename, 'exec')\nprint('Embedded source files verified:', len(SOURCES))"))
    notebook.cells.append(nbformat.v4.new_markdown_cell("## Re-run experiments\n\n"+commands+"\n\nThe original comparison rule is preserved in `IMPROVEMENT_PROTOCOL.md`; this round's decision policy is separately recorded in `EXTENSION_PROTOCOL.md`. The changed policy is not evidence of better performance by itself; compare the actual fixed-split and gapped predictions."))
    nbformat.write(notebook,ROOT/"ani_extensions.ipynb")
    print("Wrote",OUT/"RESULTS.md",flush=True)


if __name__=="__main__":build()
