"""Build the three final Part 2 hand-in notebooks from preserved research files."""
from pathlib import Path
import hashlib
import inspect
import json
import textwrap
import nbformat as nbf


SETUP = '''from pathlib import Path
import hashlib, json, os, runpy, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

locations = [candidate for base in [Path.cwd(), *Path.cwd().parents]
             for candidate in (base, base / "submission/midterm/part2", base / "midterm/part2")]
HANDIN = next(p for p in locations if (p / "01_final_model.ipynb").is_file())
ROOT = HANDIN / "resources"
assert (ROOT / "run.py").is_file(), "Run python part2/packaging/build_handin.py first, or extract the complete hand-in ZIP."
WORKSPACE = runpy.run_path(str(ROOT / "run.py"))["prepare_workspace"](ROOT)
sys.path.insert(0, str(WORKSPACE))
GENERATED = HANDIN / "generated"
GENERATED.mkdir(exist_ok=True)
sha256 = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
from experiments import load_data, metrics, PHASES
data = load_data()
SPRINT = ROOT / "experiments/leaderboard_sprint/runs/4dc0f274d45bce9e"
ledger = json.loads((ROOT / "submissions/leaderboard_results.json").read_text())
pd.set_option("display.max_rows", 16)
pd.set_option("display.max_columns", 12)
'''


def md(text):
    return nbf.v4.new_markdown_cell(inspect.cleandoc(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip() + "\n")


def write(path, cells):
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    })
    # Stable cell IDs avoid unrelated diffs when regenerating the notebooks.
    for i, cell in enumerate(notebook.cells):
        cell.id = hashlib.sha256(f"{path.name}:{i}".encode()).hexdigest()[:12]
    nbf.write(notebook, path)


def build(repo, source_files):
    repo = Path(repo)
    destination = repo / "submission/midterm/part2"
    destination.mkdir(parents=True, exist_ok=True)
    training_source = (repo / "part2/packaging/train_final.py").read_text().split('if __name__ == "__main__":')[0]
    final = [
        md('''# Part 2 — final outcome model

        **Final scored artifact: file 13, reported Kaggle log loss 0.58465 (lower is better).**
        The exact CSV is `part2_best_submission.csv`. This notebook follows the supplied
        template's flow: data, model, validation, reason-error analysis, improvements,
        and prediction export. The template is a starting point, not a required cell layout.

        Run all cells to independently rescore the saved held-out predictions, regenerate
        figures, and replay the final blend. Set `RUN_FINAL_TRAINING=True` near the end
        to **fit the actual model from raw supplied data**, with no old model caches.
        Saved-probability replay and fresh training are identified separately.

        [Experiment history](02_experiments.ipynb) · [Full implementation](03_source_code.ipynb)
        · [Development narrative](resources/MODEL_DEVELOPMENT.md)
        · [Original template](https://colab.research.google.com/drive/1mECHFXxHQnHAQiq5T3VPxeAjHXwFK7TG)
        '''), code(SETUP),
        md('''## Supplied data and evaluation boundaries

        We have 8,037 training outcomes, 1,013 validation outcomes, and 1,500 unlabeled
        test rows. Outcome label 1 means success. IDs are used for joins and splitting,
        never as predictors. No additional dataset was downloaded and no deep learning
        model was trained. Input data and the supplied pathway catalog are included.

        The primary benchmark refits preprocessing inside each outer/inner training
        split. It uses the official validation split, three forward windows, two
        repeats of three-fold trial-grouped CV, and three longer-gap checks. The
        grouped CV pool contains train + validation, with every evaluation trial
        excluded from its fitting fold. The metric gives phases I, II and III equal weight.

        The final fit uses all 9,050 labeled rows. As in Ani's published recipe, its
        Ani-derived vocabulary/IDF fits use **unlabeled text from all three supplied
        splits**, including test. The original text/tree component fits preprocessing
        on labeled fitting rows. Test outcomes are never available. Strict benchmark
        and published-preprocessing results are reported separately.
        '''), code('''
        protocol = json.loads((ROOT / "experiments/local_benchmark/protocol.json").read_text())
        for split, expected in protocol["data_sha256"].items():
            assert sha256(ROOT / "data" / f"outcome_{split}.csv") == expected
        assert "label" not in data["test"]
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            assert not set(data[a].trial_id) & set(data[b].trial_id)
        display(pd.DataFrame([dict(split=k, rows=len(v), trials=v.trial_id.nunique(),
                                   outcome_labels="label" in v) for k, v in data.items()]))
        display(pd.DataFrame([{k: v for k, v in s.items() if k not in ["fit_ids", "eval_ids"]}
                              for s in protocol["splits"]]))
        '''),
        md('''## Frozen final recipe

        `p13 = 0.70 * p10 + 0.30 * p01`, clipped to [0.001, 0.999].
        File 10 is half Ani's recency stack and half our faster-recency stack with
        eight neighbor features. The original ensemble `p01` combines a
        three-logistic-regression mixture (50%) with CatBoost depth 6 and depth 8
        (25% each). The final weights are frozen, not searched again here.

        The Ani base learners use text TF-IDF, logistic regression, multinomial
        Naive Bayes, and same-phase neighbors. Grouped out-of-fold predictions train
        the logistic stacking layer. Eligibility, disease, drug and ICD text,
        phase, and supplied drug knowledge support the components; stopping reasons
        and Part 1 embeddings are excluded from outcome training.

        The full fresh-fit orchestration appears below. Every model, preprocessing,
        benchmark, and plot implementation is readable in the code appendix.
        '''),
        md('''## Validation against the baseline and both Ani references

        These predictions were generated with the 1,013 official validation rows
        held out of supervised fitting. They are **not** predictions from the final
        model fitted on train + validation. We recompute all metrics from saved row
        probabilities. F1 uses a 0.5 success threshold; PR-AUC is average precision.
        '''), code('''
        validation = data["val"]
        predictions = {"Eligibility LR baseline": np.load(
            ROOT / "experiments/model_search/linear_validation_predictions.npz"
        )["lr_criteria_none_C2_phase0"]}
        names = {"Ani unweighted": "ani_first", "Ani recency": "ani_stack",
                 "File 10": "file10", "File 13 (final)": "mix_legacy01_0.3"}
        for label, name in names.items():
            saved = pd.read_csv(SPRINT / "strict/predictions/official" / f"{name}.csv")
            assert saved[["ID", "trial_id", "phase", "label"]].equals(
                validation[["ID", "trial_id", "phase", "label"]])
            predictions[label] = saved.probability.to_numpy()
        scores = {name: metrics(validation, p) for name, p in predictions.items()}
        assert abs(scores["File 13 (final)"]["phase_mean_logloss"] - 0.5532789641183768) < 1e-12
        phase_table = pd.DataFrame([dict(model=name, phase=phase,
            **{metric: score[phase+"_"+metric] for metric in ["logloss", "f1", "roc_auc", "pr_auc"]})
            for name, score in scores.items() for phase in PHASES])
        display(phase_table)
        display(pd.DataFrame(scores).T[["phase_mean_logloss"]])
        phase_table.to_csv(GENERATED / "validation_phase_metrics.csv", index=False)

        ax = phase_table.pivot(index="phase", columns="model", values="logloss").plot.bar(figsize=(10, 4))
        ax.set(ylabel="Held-out log loss (lower is better)", xlabel="Clinical phase",
               title="Official validation: identical held-out rows")
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(GENERATED / "validation_comparison.png", dpi=160, bbox_inches="tight")
        plt.show()
        '''),
        md('''## Post-hoc stopping-reason analysis

        On known failed validation trials, a false-success error means predicted
        probability ≥ 0.5. The supplied frozen reason table uses known Part 1 labels
        where available and historical inferred labels elsewhere. Their counts are
        shown separately; unmatched trials remain a separate category. Reasons can
        overlap, and inferred reasons add uncertainty. This table is joined **after**
        outcome prediction and does not affect any model score or fit.

        This Part 2 package consumes the existing reason-label export and its provenance;
        it does not train or revise Part 1. No Part 1 checkout is needed to reproduce
        these tables and plots.
        '''), code('''
        from error_analysis import errors_by_reason
        reason_labels = pd.read_csv(ROOT / "experiments/model_search/reason_labels_for_analysis.csv")
        display(json.loads((ROOT / "experiments/model_search/reason_model_provenance.json").read_text()))
        reason_table = pd.concat([errors_by_reason(validation, predictions[name], reason_labels).assign(model=name)
            for name in ["Eligibility LR baseline", "File 13 (final)"]], ignore_index=True)
        display(reason_table)
        reason_table.to_csv(GENERATED / "reason_errors.csv", index=False)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for phase, ax in zip(PHASES, axes):
            subset = reason_table[reason_table.phase.eq(phase)]
            subset.pivot(index="reason", columns="model", values="false_success_rate").plot.bar(ax=ax)
            ax.set(title=f"Phase {phase}", xlabel="", ylim=(0, 1))
            ax.tick_params(axis="x", labelrotation=70, labelsize=8)
            ax.legend(fontsize=7)
        axes[0].set_ylabel("False-success rate among failed trials")
        fig.suptitle("Post-hoc reasons; known/inferred counts and Wilson intervals in the table")
        fig.tight_layout()
        fig.savefig(GENERATED / "reason_errors.png", dpi=160, bbox_inches="tight")
        plt.show()
        '''),
        md('''## Exact scored artifact and recorded-component replay

        The small component archive contains native predictions from the original
        fitted models, with their cache hashes recorded. Recombining these arrays
        checks the final recipe and CSV; **it is not a new fit**. We preserve the
        exact scored CSV bytes separately from newly serialized predictions.
        '''), code('''
        artifact = ROOT / "submissions/13_file10_legacy30.csv"
        expected_sha = "be29fc849f63ed4c23b31e67c97aaa97c6940f569b0af0920cf9799b428f7636"
        assert sha256(artifact) == sha256(HANDIN / "part2_best_submission.csv") == expected_sha
        provenance = json.loads((ROOT / "experiments/final_handin/recorded_components_provenance.json").read_text())
        component_path = ROOT / "experiments/final_handin/recorded_components.npz"
        assert sha256(component_path) == provenance["component_archive_sha256"]
        with np.load(component_path) as components:
            assert components["ID"].tolist() == data["test"].ID.tolist()
            file10 = .5*components["ani_stack"] + .5*components["fast2_neighbors"]
            replay = np.clip(.7*file10 + .3*components["legacy01"], .001, .999)
        recorded = pd.read_csv(artifact)
        sample = pd.read_csv(ROOT / "data/part2_sample_submission.csv")
        assert recorded.ID.equals(sample.ID) and recorded.ID.is_unique and len(recorded) == 1500
        replay_error = float(np.max(np.abs(recorded.label - replay)))
        assert replay_error < 1e-12
        pd.DataFrame(dict(ID=data["test"].ID, label=replay)).to_csv(GENERATED / "recorded_blend_replay.csv", index=False)
        (GENERATED / "exact_scored_submission.csv").write_bytes(artifact.read_bytes())
        display(dict(kaggle_score=ledger["current_best"]["kaggle_score"],
                     exact_csv_sha256=expected_sha, recorded_component_replay_max_error=replay_error))
        '''),
        md('''## Fresh final training from raw data

        This code calls the preserved model implementations, refits both stacking
        banks and the original text/tree ensemble, checks disjoint stacking folds,
        and tests native inference after replacing IDs and adding a dummy label.
        It writes to `generated/fresh_fit/`, leaving the scored artifact intact.
        Set the flag in the following cell to run it. This takes substantially longer
        than replaying saved probabilities. Machine/library differences may change
        floating-point results; the receipt records versions and numerical differences.
        '''), code(training_source), code('''
        RUN_FINAL_TRAINING = False
        if RUN_FINAL_TRAINING:
            fresh_receipt = refit_final(ROOT, GENERATED / "fresh_fit")
            display(fresh_receipt)
        else:
            print("Fresh training is off. The checks above rescore held-out predictions and replay recorded components.")
        verified_refit = ROOT / "experiments/final_handin/refit_verification.json"
        if verified_refit.exists():
            print("Recorded verification of the fresh-fit path during hand-in preparation:")
            display(json.loads(verified_refit.read_text()))
        '''),
        md('''## Interpretation and limits

        File 13 improved the reported Kaggle log loss from Ani's 0.58688 to 0.58465.
        File 15 improved some local averages but scored 0.58492, so we retained file 13.
        The final reserve was not submitted. Local validation was repeatedly used for
        selection; bootstrap intervals do not correct for this adaptive search.
        The reported rank and scores are historical user/commit reports, not a live
        leaderboard query. See the experiment notebook for all phases, failures,
        controls, and the distinction between configurations and independent fits.
        ''')]
    write(destination / "01_final_model.ipynb", final)

    history = [md('''# Part 2 — experiments and selection history

        This notebook organizes the complete working evidence by phase. It preserves
        unsuccessful trials and separates local measurements from Kaggle feedback.
        Full summary tables, split protocols, settings, manifests and figures are
        included under `resources/experiments/`, along with the row predictions used
        by these finalized comparisons. Other intermediate prediction arrays and
        repeated fold audits remain in the working project and can be regenerated.
        The original research notebooks are
        under `resources/notebooks/`. The source appendix includes experiment and
        figure-generation implementations, including the final files 11–16 sprint.
        '''), code(SETUP),
        md('''## Experiment inventory

        Counts below are rows in the retained result tables. They are **not** a total
        number of independent trained models: ensemble weights reuse predictions,
        controls recur across phases, and one recipe is evaluated on multiple splits
        and preprocessing tracks. The 318 component and 434 ensemble rows belong to
        the initial search, not to the later common benchmark.
        '''), code('''
        stages = {
            "Initial component search": "model_search/all_component_scores.csv",
            "Initial ensemble search": "model_search/ensemble_scores.csv",
            "Stack proposals and controls": "stack_improvements/runs/097ac8161372126d/strict/summary.csv",
            "Meta extensions and controls": "ani_extensions/meta/76d98f1d5e9d5ff1/strict/summary.csv",
            "Neighbor extensions and controls": "ani_extensions/neighbors/a27759052dfa9571/strict/summary.csv",
            "Files 01–10 on the common benchmark": "all_submissions/runs/8f68385e25e07615/comparison_strict.csv",
            "Sprint: 32 proposals plus controls": "leaderboard_sprint/runs/4dc0f274d45bce9e/summary_strict.csv",
            "Recent trees: 4 proposals plus controls": "leaderboard_sprint/recent_tree/dea847f57c7c07ee/summary_strict.csv",
            "Refinement: 19 proposals plus controls": "leaderboard_sprint/refine/2e5c792462013673/summary_strict.csv",
            "Final slot: 9 proposals plus controls": "leaderboard_sprint/finish/b235582538f63f9c/summary_strict.csv",
        }
        inventory = pd.DataFrame([dict(stage=stage, result_rows=len(pd.read_csv(ROOT / "experiments" / path)),
                                       table=path) for stage, path in stages.items()])
        display(inventory)
        inventory.to_csv(GENERATED / "experiment_inventory.csv", index=False)
        for stage, path in stages.items():
            display(Markdown(f"### {stage}"))
            display(pd.read_csv(ROOT / "experiments" / path))
        '''),
        md('''## Why these changes were tried

        1. Text, recency, structured drug features and tree models supplied diverse
           original components; the first submitted recipes trailed Ani's stacks.
        2. A fixed trial-grouped and chronological benchmark put both Ani winners and
           our proposals on the same rows. Stronger meta regularization, temporal
           features, and many alternative mixtures were retained even when they lost.
        3. Faster recency and additional same-phase neighbor views yielded file 10.
           A half-strength blend kept Ani's stack as an anchor.
        4. The original text/tree ensemble contributed complementary errors. A 20%
           contribution produced file 11; 30% produced the final file 13.
        5. Recent-tree and trend refinements looked attractive locally but file 15
           did not improve Kaggle. File 16's tiny, uncertain gains did not justify
           using the last submission slot.

        [Full rationale and links](resources/MODEL_DEVELOPMENT.md) ·
        [Frozen protocols](resources/docs/README.md) ·
        [Original notebook guide](resources/notebooks/README.md)
        '''),
        md('''## Direct comparison: the same benchmark and preprocessing track

        Lower is better. These rows come from the same strict run, using identical
        outer splits, phase weighting and grouped-CV aggregation. The published
        preprocessing track is shown separately. Local results are selection evidence;
        stronger local averages did not guarantee stronger Kaggle performance.
        '''), code('''
        chosen = ["ani_first", "ani_stack", "file10", "mix_legacy01_0.2", "mix_legacy01_0.3"]
        columns = ["official", "temporal", "worst", "cv", "gap_mean", "gap_worst"]
        comparisons = {}
        for track in ["strict", "published"]:
            table = pd.read_csv(SPRINT / f"summary_{track}.csv").set_index("model")
            comparisons[track] = table.loc[chosen, columns]
            display(Markdown(f"### {track.title()} preprocessing"))
            display(comparisons[track])
        comparisons["strict"].to_csv(GENERATED / "common_benchmark.csv")
        plotted = comparisons["strict"][["temporal", "gap_mean", "cv"]].rename(
            index={"ani_first": "Ani unweighted", "ani_stack": "Ani recency", "file10": "File 10",
                   "mix_legacy01_0.2": "File 11", "mix_legacy01_0.3": "File 13 (final)"},
            columns={"temporal": "Forward windows", "gap_mean": "Longer-gap windows", "cv": "Grouped CV"})
        ax = plotted.plot(marker="o", figsize=(10, 4))
        ax.set(ylabel="Equal-phase log loss", xlabel="Recipe", title="Shared strict benchmark")
        ax.set_xticks(range(len(plotted)), plotted.index, rotation=20, ha="right")
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(GENERATED / "common_benchmark.png", dpi=160, bbox_inches="tight")
        plt.show()
        '''),
        md('''## Verify the main comparisons from row-level predictions

        The following check recomputes every phase log loss on each of the ten original
        splits for the five recipes above, in both preprocessing tracks. It also checks
        split membership and original labels. The longer-gap results and paired
        uncertainty estimates remain in the linked phase reports.
        '''), code('''
        from benchmark_protocol import initialize, resolve
        universe, protocol = initialize()
        verification_rows = []
        for track in ["strict", "published"]:
            for split in protocol["splits"]:
                _, evaluation = resolve(universe, split)
                stored = pd.read_csv(SPRINT / track / f"scores_{split['name']}.csv").set_index("model")
                for model in chosen:
                    frame = pd.read_csv(SPRINT / track / "predictions" / split["name"] / f"{model}.csv")
                    assert frame[["ID", "trial_id", "phase", "label"]].equals(evaluation[["ID", "trial_id", "phase", "label"]])
                    measured = metrics(evaluation, frame.probability.to_numpy())
                    for phase in PHASES:
                        assert abs(measured[phase+"_logloss"] - stored.loc[model, phase+"_logloss"]) < 1e-12
                    assert abs(measured["phase_mean_logloss"] - stored.loc[model, "phase_mean_logloss"]) < 1e-12
                    verification_rows.append(dict(track=track, split=split["name"], model=model,
                                                  phase_mean_logloss=measured["phase_mean_logloss"]))
        verified = pd.DataFrame(verification_rows)
        assert len(verified) == 100
        verified.to_csv(GENERATED / "benchmark_rescoring.csv", index=False)
        print(f"Verified {len(verified)} recipe/split/track evaluations against original row labels.")
        '''),
        md('''## Kaggle feedback and final decision

        Only confirmed submitted artifacts have recorded scores. Local recipes and
        unsubmitted reserves do not inherit scores from similar filenames or weights.
        These are historical user/commit reports; no live leaderboard was queried.
        '''), code('''
        rows = []
        for name, item in ledger["ani_winners"].items():
            rows.append(dict(model=name, kaggle_log_loss=item["kaggle_score"], source=item["source"]))
        for name, item in ledger["submissions"].items():
            assert sha256(ROOT / "submissions" / name) == item["sha256"]
            rows.append(dict(model=name, kaggle_log_loss=item["kaggle_score"], source=item["source"]))
        feedback = pd.DataFrame(rows)
        display(feedback)
        feedback.to_csv(GENERATED / "kaggle_feedback.csv", index=False)
        ax = feedback.set_index("model").kaggle_log_loss.plot(marker="o", figsize=(10, 4))
        ax.set(ylabel="Reported Kaggle log loss", xlabel="Scored artifact")
        labels = ["Ani unweighted", "Ani recency", "File 03", "File 02", "File 10", "File 11", "File 13 (final)", "File 15"]
        ax.set_xticks(range(len(feedback)), labels, rotation=30, ha="right", fontsize=9)
        plt.tight_layout()
        plt.savefig(GENERATED / "kaggle_progress.png", dpi=160, bbox_inches="tight")
        plt.show()
        display(json.loads((ROOT / "experiments/leaderboard_sprint/refine/2e5c792462013673/submission_decision.json").read_text()))
        '''), md('''## Re-running research versus reproducing the final model

        The final model's fresh training is in notebook 01. The complete historical
        research suite is much larger and is not rerun by default. Its original
        notebooks, saved notebook outputs, settings and trial summary tables are
        included in `resources/`. Their execution may require regenerating omitted
        intermediate arrays or using the full working checkout. Use the commands in
        [reproduction instructions](resources/docs/CACHE_REPRODUCTION.md) and
        [organization guide](resources/docs/ORGANIZATION.md), replacing `part2/`
        with this package's `resources/` path. Rebuild omitted intermediate predictions
        and fitted caches first; some historical runners expect earlier phases to
        have completed. The full working project retains those intermediates.

        Historical scripts that recreate Part 1 reason labels require the separate
        Part 1 project. The supplied frozen labels fully reproduce Part 2 post-hoc
        analysis without that step. Historical Git-based provenance/tag checks also
        require the research Git repository; the hand-in uses file hashes instead.
        ''')]
    write(destination / "02_experiments.ipynb", history)

    paths = [p for p in source_files if p.suffix == ".py"]
    appendix = [md('''# Part 2 — complete source appendix

        This notebook embeds the preserved Python code for model fitting, feature
        construction, experimental searches, evaluation, figure generation, tests,
        launchers, and hand-in preparation. The two analysis notebooks contain their
        own table/figure code. The historical Ani notebooks are also included in
        `resources/experiments/ani_comparison/source/`.

        **Run all writes inspectable copies into `_source_export/`; it does not run
        every search or overwrite the canonical research sources.** Each write cell
        contains the actual source, with its relative path and hash. The final cell
        verifies these copies byte-for-byte against the included implementations.
        '''), code(SETUP), code(
        'export_directory = HANDIN / "_source_export"\n'
        'export_directory.mkdir(exist_ok=True)\n'
        'os.chdir(export_directory)\n' + "\n".join(
            f"Path({folder!r}).mkdir(parents=True, exist_ok=True)"
            for folder in sorted({str(p.relative_to(repo / "part2").parent) for p in paths}))) ]
    hashes = {}
    for path in paths:
        relative = str(path.relative_to(repo / "part2"))
        content = path.read_text()
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        appendix.append(md(f"## `{relative}`\n\nSHA-256: `{hashes[relative]}`"))
        cell = nbf.v4.new_code_cell(f"%%writefile {relative}\n" + content)
        cell.metadata["source_path"] = relative
        cell.metadata["source_sha256"] = hashes[relative]
        appendix.append(cell)
    source_lengths = {str(p.relative_to(repo / "part2")): p.stat().st_size for p in paths}
    appendix.append(code(f"expected_sources = {hashes!r}\nsource_lengths = {source_lengths!r}\n" + '''
for relative, expected in expected_sources.items():
    # IPython adds a final newline to %%writefile bodies. Two archived Ani
    # sources originally had no terminal newline; restore that exact EOF.
    exported = export_directory / relative
    content = exported.read_bytes()
    if len(content) == source_lengths[relative] + 1 and content.endswith(b"\\n"):
        exported.write_bytes(content[:-1])
    assert sha256(export_directory / relative) == expected, relative
    assert sha256(ROOT / relative) == expected, relative
os.chdir(HANDIN)
print(f"Verified {len(expected_sources)} complete source files.")
'''))
    write(destination / "03_source_code.ipynb", appendix)
