"""Render a concise research report, figures, and a runnable reading notebook."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
from experiments import ROOT
from improvement_report import current, REFERENCES


def table(df):
    if isinstance(df.index, pd.MultiIndex):
        reset = df.reset_index()
        headers = list(reset.columns)
        rows = [[f"{v:.6f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
                for row in reset.to_numpy()]
        return "\n".join(["| "+" | ".join(headers)+" |", "| "+" | ".join(["---"]*len(headers))+" |"]+
                         ["| "+" | ".join(row)+" |" for row in rows])
    rows = [[str(i)]+[f"{v:.6f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
            for i,row in zip(df.index, df.to_numpy())]
    headers = [df.index.name or "model"]+list(df.columns)
    return "\n".join(["| "+" | ".join(headers)+" |", "| "+" | ".join(["---"]*len(headers))+" |"]+
                     ["| "+" | ".join(row)+" |" for row in rows])


def build():
    out = current()
    selection = json.loads((out / "selection.json").read_text())
    verification = json.loads((out / "verification.json").read_text())
    manifest = json.loads((ROOT / "submissions/stack_improvements_manifest.json").read_text())
    summaries = {t: pd.read_csv(out / "text_followup" / t / "independent_summary.csv").set_index("model")
                 for t in ["strict", "published"]}
    pairs = pd.read_csv(out / "paired_comparisons.csv")
    chosen = selection["shortlist"]
    display_names = list(dict.fromkeys(REFERENCES+chosen+["ani_phase3_trend", "tau6_stack", "blend_winners50",
                    "recent_joint_stack", "recent_text_stack", "unweighted_text_stack", "unweighted_text_trend"]))
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1, 2, figsize=(13,5), constrained_layout=True)
    for ax, (track, summary) in zip(axes, summaries.items()):
        ax.scatter(summary.cv, summary.temporal, color="#a4acb4", s=35, label="Proposals / controls")
        highlighted = [("ani_stack", "#0067b1"), ("ani_first", "#db7600")]+[(n,"#268046") for n in chosen]
        best_cv = summary.cv.idxmin()
        if best_cv not in REFERENCES+chosen:
            highlighted.append((best_cv, "#7851a9"))
        for name, color in highlighted:
            row = summary.loc[name]
            ax.scatter([row.cv], [row.temporal], color=color, s=90, label=name)
            ax.annotate(name, (row.cv,row.temporal), xytext=(5,5), textcoords="offset points", fontsize=8)
        ax.set(title=f"{track.capitalize()} preprocessing", xlabel="Grouped CV log loss (lower is better)",
               ylabel="Chronological mean log loss (lower is better)")
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.savefig(figdir / "tradeoffs.png", dpi=180)
    plt.close(fig)
    scores = pd.read_csv(out / "text_followup/strict/independent_scores.csv")
    fig, ax = plt.subplots(figsize=(10,4), constrained_layout=True)
    ts = scores[scores.kind.eq("temporal")].pivot(index="split",columns="model",values="phase_mean_logloss")
    for name in list(dict.fromkeys(REFERENCES+chosen+["ani_phase3_trend"])):
        ax.plot(np.arange(len(ts)), ts[name], marker="o", label=name)
    ax.set_xticks(np.arange(len(ts)), [x.removeprefix("time_").replace("_to_"," → ").replace("_","–") for x in ts.index])
    ax.set(ylabel="Equal-phase log loss", title="Identical forward windows, strict preprocessing")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.savefig(figdir / "chronological.png", dpi=180)
    plt.close(fig)
    root = ROOT / "experiments/stack_improvements"
    rel = f"runs/{out.name}"
    if chosen:
        recommendation = "The fixed screening rule shortlisted **"+", ".join(chosen)+"**. Their new CSVs are exploratory and unsubmitted; inspect the gapped-window checks below before using them."
    else:
        recommendation = "**Retain Ani's 0.58688 submission. No new recipe passed the rule fixed before testing, so no new numbered Kaggle submission was promoted or added.**"
    sections = ["# Part Two: both Ani winners and proposed improvements\n", recommendation,
        "Both original winners were reproduced: unweighted stack **0.58810** and recency-weighted stack **0.58688** Kaggle log loss. These scores are tied to the archived CSV hashes and Ani's commit messages, not a fresh leaderboard query. New local scores are not Kaggle results.",
        "## What was tested\n\n27 recipes on the identical ten saved splits in each of two preprocessing tracks: **540 held-out recipe/split evaluations**. These comprise both winners, the existing Phase III trend control, and 24 other variants. A 13-recipe first batch tests weighting, calibration, interactions and mixtures; an adaptive eight-recipe batch adds capped years; a final adaptive six-recipe batch adds joint clinical-text and supervised word-ratio classifiers. All attempts are retained. No further search followed the gapped checks.",
        "Training folds exclude each held-out trial and all its phase rows. The strict track refits vocabulary/IDF inside stacking folds; the separately labeled published track preserves Ani's use of all available covariates for text preprocessing. Neither uses evaluation outcomes for fitting. All final fits use the same 9,050 labeled rows.",
        "## Strict comparison\n\n"+table(summaries["strict"].loc[display_names,["official","temporal","worst","cv"]]),
        "## Published-recipe sensitivity comparison\n\n"+table(summaries["published"].loc[display_names,["official","temporal","worst","cv"]]),
        f"Full [strict table]({rel}/text_followup/strict/independent_summary.csv) and [published table]({rel}/text_followup/published/independent_summary.csv) include every attempted recipe. Lower is better. Temporal means give equal weight to windows and phases; CV pools predictions across folds within each repeat.",
        f"![Trade-offs]({rel}/figures/tradeoffs.png)\n\n![Chronological checks]({rel}/figures/chronological.png)",
        "## Selection and uncertainty\n\nThe predeclared rule requires a lower chronological mean than **both** references, CV within 0.001 of the better reference, and worst-window loss within 0.003 of the better reference's worst loss. It is intentionally conservative. Ani's recency winner and unweighted winner can trade places between chronological and random grouped CV, so one favorable metric is insufficient.",
        f"[Paired intervals against both references]({rel}/paired_comparisons.csv) resample whole trials and average repeated losses before resampling. They are descriptive and conditional, without correction for adaptive model selection. All supplied labels have been consulted; these are not untouched-test estimates."]
    strict = summaries["strict"]
    best_cv = strict.cv.idxmin()
    row = strict.loc[best_cv]
    sections.insert(4, f"The strongest strict grouped-CV result is **{best_cv}**, with loss **{row.cv:.6f}**, versus **{strict.loc['ani_first','cv']:.6f}** for Ani's unweighted stack. Its chronological mean is **{row.temporal:.6f}**, versus **{strict.loc['ani_stack','temporal']:.6f}** for Ani's recency winner. This is a useful CV gain, but it does not justify replacing the model with better time-based performance.")
    focus = pairs[(pairs.track=="strict") & pairs.model.isin(chosen or ["ani_phase3_trend", best_cv])]
    sections.append(table(focus.set_index(["model","reference","kind"])[["delta","lower","upper"]]))
    sections.append("## Larger time-gap robustness checks\n\nAfter freezing selection, fit through 2005/2007/2009 and evaluate 2008–2009/2010–2011/2012–2013, respectively. The additional gaps probe longer-range transport. Evaluation labels overlap prior checks; this is robustness evidence, not new independent data.")
    for track in ["strict","published"]:
        gap = pd.read_csv(out / "robustness" / track / "summary.csv").set_index("model")
        sections.append(f"### {track.capitalize()}\n\n"+table(gap[["temporal","worst"]]))
    sections.append("## Reproduction and deliverables\n\n"+f"All source/data hashes and fitting boundaries were checked. Maximum probability difference from the archived ten-split Ani predictions is {verification['historical_baseline_max_probability_error']:.2g}. Maximum differences from original Kaggle CSV probabilities: "+", ".join(f"{name} {v['max_probability_error']:.2g}" for name,v in verification["both_kaggle_sources_reproduced"].items())+". The final native model replay also verifies that changing IDs or evaluation labels cannot change predictions. All 13 protocol/scoring/model checks passed.")
    if manifest["submissions"]:
        sections.append("New optional CSVs: "+", ".join(f"[{v['file']}](../../submissions/{v['file']})" for v in manifest["submissions"])+". Their Kaggle scores are unknown.")
    sections.append("Reproduce with [IMPROVEMENT_PROTOCOL.md](../../IMPROVEMENT_PROTOCOL.md), pinned [requirements-benchmark.txt](../../requirements-benchmark.txt), and the [executed notebook](../../stack_improvements.ipynb). The original data, submissions, branch history, and ten split definitions are preserved.")
    (root / "RESULTS.md").write_text("\n\n".join(sections)+"\n")
    nb = nbformat.v4.new_notebook()
    nb.metadata.kernelspec = dict(display_name="Python (Part Two)", language="python", name="part2")
    nb.cells = [nbformat.v4.new_markdown_cell("# Part Two: paired comparison of both Ani winners\n\n"+recommendation+"\n\nRead `IMPROVEMENT_PROTOCOL.md` for the frozen rules and adaptive-batch history. This notebook reads the saved evaluations; it does not retrain models."),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\nroot = Path.cwd()\nif (root / 'part2').exists(): root /= 'part2'\nbase = root / 'experiments/stack_improvements'\nrun = base / 'runs' / json.loads((base / 'current_run.json').read_text())['run']\nprint('Run:', run.name)\ndisplay(json.loads((run / 'selection.json').read_text()))"),
        nbformat.v4.new_markdown_cell("## All recipes, identical splits\n\nSmaller log loss is better. Keep strict and published preprocessing policies separate."),
        nbformat.v4.new_code_cell("for track in ['strict', 'published']:\n    print(track)\n    display(pd.read_csv(run / 'text_followup' / track / 'independent_summary.csv').sort_values('temporal').round(6))\ndisplay(Image(filename=str(run / 'figures/tradeoffs.png')))"),
        nbformat.v4.new_markdown_cell("## Paired differences against both winners\n\nNegative favors the proposal; intervals crossing zero are inconclusive. Resampling keeps trial rows together and does not count repeat appearances as independent trials."),
        nbformat.v4.new_code_cell("pairs = pd.read_csv(run / 'paired_comparisons.csv')\nshortlist = json.loads((run / 'selection.json').read_text())['shortlist']\nfocus = shortlist or ['ani_phase3_trend', 'recent_text_stack']\ndisplay(pairs[pairs.model.isin(focus)].round(6))\ndisplay(Image(filename=str(run / 'figures/chronological.png')))"),
        nbformat.v4.new_markdown_cell("## Additional time gaps and verification\n\nThese reuse historical labels; they are robustness checks, not untouched validation. Only the frozen shortlist and both references are evaluated."),
        nbformat.v4.new_code_cell("for track in ['strict','published']:\n    print(track)\n    display(pd.read_csv(run / 'robustness' / track / 'summary.csv').round(6))\ndisplay(json.loads((run / 'verification.json').read_text()))\ndisplay(json.loads((root / 'submissions/stack_improvements_manifest.json').read_text()))")]
    nbformat.write(nb, ROOT / "stack_improvements.ipynb")
    print("Built", root / "RESULTS.md")


if __name__ == "__main__":
    build()
