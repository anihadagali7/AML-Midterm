"""Build directly comparable tables, paired intervals, figures and a notebook."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from all_submissions_benchmark import current, NAMES, EXTENSION, SOURCES
from benchmark_protocol import initialize, digest
from experiments import ROOT
from improvement_report import audit_track, paired_interval

TARGET = "10_ani_fast2_neighbors_half"


def markdown_table(data):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return "Unknown" if np.isnan(value) else f"{value:.6f}"
        return str(value)
    return "\n".join(["| " + " | ".join(data.columns) + " |",
                      "| " + " | ".join(["---"]*len(data.columns)) + " |"] +
                     ["| " + " | ".join(cell(x) for x in row) + " |" for row in data.itertuples(index=False, name=None)])


def report():
    out = current(); universe, protocol = initialize()
    gaps = json.loads((EXTENSION / "gaps/protocol.json").read_text())
    ledger = json.loads((ROOT / "submissions/leaderboard_results.json").read_text())
    kaggle = {k: v["kaggle_score"] for k, v in ledger["ani_winners"].items()}
    kaggle.update({Path(k).stem: v["kaggle_score"] for k, v in ledger["submissions"].items()})
    for filename, entry in ledger["submissions"].items():
        assert digest(ROOT / "submissions" / filename) == entry["sha256"]
    best = ledger.get("current_best")
    rank_note = (f"The user reported rank {best['reported_rank']}. "
                 if best and best.get("reported_rank") is not None
                 else "Leaderboard rank was not refreshed for this score. ")
    score_update = (f"**Later Kaggle result:** `{best['submission']}` scored **{best['kaggle_score']:.5f}**, "
        f"improving Ani's 0.58688 by **{best['improvement_over_ani']:.5f}**. "
        + rank_note +
        "This result was reported after the frozen comparison. Local predictions and selection evidence are unchanged. "
        "The current ledger supplies the Kaggle column; its original input snapshot is preserved with the run."
        if best else "Kaggle results are recorded separately from local validation.")
    tables = {}; windows = []; pairs = []; wins = []
    for track in ["strict", "published"]:
        main, scores, frames = audit_track(out / track, universe, protocol, names=NAMES)
        gap, gap_scores, gap_frames = audit_track(out / "gaps" / track, universe, gaps, names=NAMES)
        table = main.join(gap[["temporal", "worst"]].rename(columns={"temporal": "gap_mean", "worst": "gap_worst"}))
        table["kaggle"] = table.index.map(kaggle)
        forward = scores[scores.kind.eq("temporal")].pivot(index="split", columns="model", values="phase_mean_logloss")
        gaps_wide = gap_scores.pivot(index="split", columns="model", values="phase_mean_logloss")
        table["forward_wins_vs_ani"] = forward.lt(forward.ani_stack, axis=0).sum()
        table["gap_wins_vs_ani"] = gaps_wide.lt(gaps_wide.ani_stack, axis=0).sum()
        table["temporal_delta_vs_02"] = table.temporal - table.loc["02_text_heavy", "temporal"]
        table["cv_delta_vs_02"] = table.cv - table.loc["02_text_heavy", "cv"]
        table.to_csv(out / f"comparison_{track}.csv")
        for metric in ["official", "temporal", "cv", "gap_mean"]:
            values = table[metric].to_numpy()
            matrix = pd.DataFrame(values[:, None] - values[None, :],
                                  index=table.index, columns=table.index)
            matrix.to_csv(out / f"pairwise_{track}_{metric}.csv")
        tables[track] = table
        windows += [scores.assign(track=track, suite="main"), gap_scores.assign(track=track, suite="gaps")]
        for reference in NAMES:
            if reference == TARGET:
                continue
            wins.append(dict(track=track, reference=reference,
                forward_wins=int((forward[TARGET] < forward[reference]).sum()),
                forward_windows=len(forward),
                gap_wins=int((gaps_wide[TARGET] < gaps_wide[reference]).sum()),
                gap_windows=len(gaps_wide)))
            for kind in ["temporal", "group_cv"]:
                pairs.append(dict(track=track, suite="main", **paired_interval(frames, TARGET, reference, kind)))
            pairs.append(dict(track=track, suite="gaps", **paired_interval(gap_frames, TARGET, reference, "temporal")))
        print("Compared", track, "file 10 against all 12 alternatives", flush=True)
    pair = pd.DataFrame(pairs)
    win_table = pd.DataFrame(wins)
    win_table.to_csv(out / "file10_window_wins.csv", index=False)
    pair.to_csv(out / "paired_comparisons.csv", index=False)
    pd.concat(windows, ignore_index=True).to_csv(out / "all_fold_scores.csv", index=False)
    figures = out / "figures"; figures.mkdir(exist_ok=True)
    strict = tables["strict"].sort_values("temporal", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 7), sharey=True)
    ys = np.arange(len(strict))
    colors = ["#118066" if k == TARGET else "#2f5da8" if k in ["ani_stack", "ani_first"] else "#748397" for k in strict.index]
    for ax, col, title in zip(axes, ["temporal", "cv"], ["Mean of three forward windows", "Repeated grouped CV"]):
        ax.scatter(strict[col], ys, c=colors, s=65)
        ax.axvline(strict.loc["ani_stack", col], color="#2f5da8", linestyle="--", linewidth=1)
        ax.set_xlabel("Equal-phase log loss (lower is better)"); ax.set_title(title)
        ax.grid(axis="x", alpha=.2)
    axes[0].set_yticks(ys, strict.index)
    fig.suptitle("All submissions on identical strict benchmark splits")
    fig.tight_layout(); fig.savefig(figures / "comparison.png", dpi=160)
    fig.savefig(figures / "comparison.pdf"); plt.close(fig)
    cols = ["official", "temporal", "worst", "cv", "gap_mean", "kaggle"]
    displayed = {k: v.loc[NAMES, cols].rename_axis("model").reset_index() for k, v in tables.items()}
    deltas = tables["strict"].loc[TARGET, ["official", "temporal", "cv", "gap_mean"]] - tables["strict"].loc["02_text_heavy", ["official", "temporal", "cv", "gap_mean"]]
    direct_wins = win_table[(win_table.track == "strict") & win_table.reference.isin(["02_text_heavy", "03_phase_specialized", "ani_stack"])]
    relevant_pairs = pair[(pair.track == "strict") & pair.reference.isin(["02_text_heavy", "03_phase_specialized", "ani_first", "ani_stack"])]
    text = f"""# All Part Two submissions on the same fixed benchmark

**Complete:** all ten numbered recipes, both Kaggle-scored Ani references, and
Ani's trend control, on the same ten original splits and three longer-gap splits.
Every table uses identical evaluation trials and equal-phase log loss. Lower is
better. These are frozen recipes; this run did not tune or generate submissions.

{score_update}

## Primary strict comparison

Every model fits preprocessing on its fitting rows only. `official` is the
supplied validation set; `temporal` averages the three original forward windows;
`worst` is the worst of those windows; `cv` pools held-out rows within each of two
grouped-CV repeats; `gap_mean` averages the three longer-gap windows. Known Kaggle
scores belong to the original file hashes, not these fold refits.

{markdown_table(displayed['strict'])}

## File 10 versus the actual non-Ani submissions

File 02 is the best scored non-Ani submission (Kaggle 0.58993); file 03 scored
0.59562. Their full original ensembles, including both CatBoost models, are now
evaluated directly. No linear-component proxy is substituted for either recipe.

For file 10 minus file 02 under strict preprocessing, the log-loss differences
are: supplied validation **{deltas.official:+.6f}**, chronological mean
**{deltas.temporal:+.6f}**, grouped CV **{deltas.cv:+.6f}**, and longer-gap mean
**{deltas.gap_mean:+.6f}**. Negative favors file 10; positive favors file 02.

File 10's number of winning periods, with the same trial IDs in each pair:

{markdown_table(direct_wins.drop(columns='track'))}

An average gain does not imply winning every period. In particular, file 03
beats file 10 in the first two original forward windows, while file 10 wins the
latest window by a larger margin. Retain this temporal variation when judging
whether the gains will transfer to a future cohort.

The old files' single historical temporal scores are not used in this table.
Files 01–06 originally trained their final submissions on 8,037 rows, while Ani
and 07–10 used 9,050. All recipes here receive the same fitting rows in each
split. That resolves the local population mismatch without inventing a Kaggle
score for a hypothetical refit of file 02.

![Chronological and grouped-CV comparison](runs/{out.name}/figures/comparison.png)

## Secondary original-recipe preprocessing comparison

This track preserves Ani/09/10 feature-only transduction and legacy training-only
preprocessing. Files 01–06 are therefore identical to the strict track; 07/08 use
the corresponding Ani predictions. This sensitivity table has matched supervised
fitting and evaluation rows but deliberately retains different original feature
policies. Use the strict table for common-preprocessing comparisons.

{markdown_table(displayed['published'])}

## Paired uncertainty for file 10

Every difference below uses the same held-out trials for both models. Repeated
CV losses are averaged per row before resampling; all phase rows of a trial share
bootstrap weights. These intervals are descriptive and do not correct for earlier
adaptive model selection. The longer-gap labels were also consulted previously.

{markdown_table(relevant_pairs[['reference','suite','kind','delta','lower','upper']])}

The complete [paired comparison table](runs/{out.name}/paired_comparisons.csv)
contains file 10 versus every alternative under both tracks. The
[window-level scores](runs/{out.name}/all_fold_scores.csv) retain every period and
phase, including unfavorable results.
All-model difference matrices are also saved for each metric and track; for
example, [strict chronological differences](runs/{out.name}/pairwise_strict_temporal.csv).
Each matrix cell is row-model loss minus column-model loss; negative favors the row.

## Fidelity, verification and reproduction

Original ensemble weights come directly from the original manifests. The legacy
adapter preserves text features, capped-year behavior, SVD dimensions, fixed tree
counts, recency half-lives and seeds. Vocabulary/IDF, SVD and metadata masks are
rebuilt on each fold's fitting rows. Already-scored Ani/09/10 predictions are
reused with source, input, split and probability checks; they are not refit with
different settings. Original submission CSVs are unchanged.

[Original replay](runs/{out.name}/original_replay.csv) compares all six original
components on supplied validation and all six numbered legacy test CSVs against
the rebuilt official-training model. [Verification](runs/{out.name}/verification.json)
independently rescores all 338 model/split/track results, checks trial boundaries,
fixed blends, source hashes and inference invariance to IDs and outcome labels.

```sh
uv pip install --python .venv/bin/python -r part2/requirements-all-submissions.txt
.venv/bin/python part2/all_submissions_benchmark.py --workers 2
.venv/bin/python part2/all_submissions_verify.py
.venv/bin/python part2/all_submissions_report.py
```

See [comparison policy](../../ALL_SUBMISSIONS_PROTOCOL.md),
[executed notebook](../../all_submissions.ipynb),
[strict machine-readable table](runs/{out.name}/comparison_strict.csv), and
[source/input manifest](runs/{out.name}/manifest.json). Native caches are ignored
by Git and can be rebuilt; per-row held-out predictions and fitting IDs are saved.
This benchmark does not upload submissions. Later user-reported Kaggle results
are joined by exact submission hashes from the current ledger.
"""
    (ROOT / "experiments/all_submissions/RESULTS.md").write_text(text)
    notebook(text, out)
    print(tables["strict"].loc[NAMES, cols].to_string(), flush=True)


def notebook(report_text, out):
    nb = nbformat.v4.new_notebook()
    nb.metadata.kernelspec = dict(display_name="Part 2", language="python", name="part2")
    root_rel = str(out.relative_to(ROOT))
    cells = [nbformat.v4.new_markdown_cell("# All Part Two submissions: direct fixed-benchmark comparison\n\nAll 13 recipes on the same 13 splits. This notebook reads verified results; rebuilding models is explicit."),
             nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT = Path.cwd() if (Path.cwd() / 'outcome_train.csv').exists() else Path.cwd() / 'part2'\nassert (ROOT / 'outcome_train.csv').exists()\nRUN = ROOT / " + repr(root_rel)),
             nbformat.v4.new_code_cell("display(pd.read_csv(RUN / 'comparison_strict.csv').set_index('model'))"),
             nbformat.v4.new_code_cell("display(pd.read_csv(RUN / 'comparison_published.csv').set_index('model'))"),
             nbformat.v4.new_code_cell("display(Image(filename=str(RUN / 'figures/comparison.png')))"),
             nbformat.v4.new_code_cell("paired = pd.read_csv(RUN / 'paired_comparisons.csv')\ndisplay(paired[(paired.track == 'strict') & paired.reference.isin(['02_text_heavy','03_phase_specialized','ani_first','ani_stack'])])"),
             nbformat.v4.new_code_cell("display(pd.read_csv(RUN / 'original_replay.csv'))\ndisplay(json.loads((RUN / 'verification.json').read_text()))"),
             nbformat.v4.new_markdown_cell(report_text.replace(f"runs/{out.name}/", root_rel + "/").replace("../../", ""))]
    sources = SOURCES + ["all_submissions_verify.py", "all_submissions_report.py", "ani_models.py", "improvement_models.py"]
    embedded = {name: (ROOT / name).read_text() for name in dict.fromkeys(sources)}
    cells.append(nbformat.v4.new_markdown_cell("## Embedded implementation source\n\nSource is included for review. This cell checks equality with the repository and compiles each module without running training."))
    cells.append(nbformat.v4.new_code_cell("embedded_sources = " + repr(embedded) + "\nfor filename, source in embedded_sources.items():\n    assert (ROOT / filename).read_text() == source, filename\n    compile(source, filename, 'exec')\nprint(f'{len(embedded_sources)} source files match and compile')"))
    nb.cells = cells
    nbformat.write(nb, ROOT / "all_submissions.ipynb")


if __name__ == "__main__":
    report()
