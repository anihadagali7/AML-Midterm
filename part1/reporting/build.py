"""Build report tables from retained measurements and audit scored artifacts.

No training, tuning, uploads, or changes to frozen study records.
"""

import csv
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
P1 = HERE.parent
ROOT = P1.parent
POOLS = ["train", "pooled", "validation", "chronology98"]
WINNER = "anchored80-cand2-inductive20-trainval235"


def read(p):
    return json.loads(p.read_text())


def rows(p):
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def resolve(name):
    if name.startswith("part1/"):
        return ROOT / name
    return P1 / name


def table(path, values, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(values)


def measurements():
    result = []

    def add(name, record, values, source):
        r = read(P1 / "submissions/results" / record)
        result.append(
            dict(
                model=name,
                recipe=r["recipe"],
                **values,
                kaggle=float(r["reported_kaggle_score"]),
                metric_source=str(source.relative_to(P1)),
                score_record="submissions/results/" + record,
            )
        )

    source = P1 / "experiments/cand2_followup_v1/results/summary.csv"
    data = rows(source)
    for name, recipe, record in [
        ("Anchored80/Cand2_20 (winner)", "blend", WINNER + ".json"),
        ("Anchored", "anchored", "anchored-trainval235.json"),
        (
            "Cand2 strict five-seed",
            "cand2",
            "cand2-cs-features-inductive5-trainval235.json",
        ),
    ]:
        values = {
            ("chronology98" if r["kind"] == "chronology" else r["kind"]): float(
                r["sample_f1_mean"]
            )
            for r in data
            if r["track"] == "current" and r["recipe"] == recipe
        }
        add(name, record, {k: values[k] for k in POOLS}, source)
    source = P1 / "research/pysr_v1/results/cv_summary.csv"
    cv = rows(source)
    supplementary = P1 / "research/pysr_v1/results/supplementary_scores.csv"
    sp = rows(supplementary)
    for name, recipe, record in [
        ("Anchored75/LDA25", "current75_25", "anchored75-lda25-trainval235.json"),
        (
            "80% prior 75/25 + 20% PySR",
            "current80_pysr20",
            "current80-pysr20-trainval235.json",
        ),
    ]:
        values = {
            r["pool"]: float(r["mean_sample_f1"]) for r in cv if r["mode"] == recipe
        }
        values.update(
            {
                r["case"]: float(r["sample_f1"])
                for r in sp
                if r["mode"] == recipe and r["case"] in POOLS
            }
        )
        add(name, record, {k: values[k] for k in POOLS}, source)
        result[-1]["supplementary_source"] = str(supplementary.relative_to(P1))
    for name, folder, recipe, record in [
        (
            "Anchored80/diverse-kernel20",
            "complementary_heads_v2",
            "blend_diverse30",
            "anchored80-diverse-kernel20-trainval235.json",
        ),
        (
            "Partial-Other80/Cand2_20",
            "partial_secondary_v1",
            "binary_unknown",
            "partial-other-unknown80-cand2-inductive20-trainval235.json",
        ),
    ]:
        source = P1 / "research" / folder / "results/summary/overview.csv"
        values = {
            r["pool"]: float(r["mean_sample_f1"])
            for r in rows(source)
            if r["recipe"] == recipe
        }
        add(name, record, {k: values[k] for k in POOLS}, source)
    source = P1 / "experiments/blend_refinement_v2/results/summary.csv"
    values = {
        ("chronology98" if r["pool"] == "chronology" else r["pool"]): float(r["mean"])
        for r in rows(source)
        if r["recipe"] == "log_pool"
    }
    add(
        "Log-odds anchored80/Cand2_20",
        "anchored80-cand2-inductive20-log-odds-trainval235.json",
        {k: values[k] for k in POOLS},
        source,
    )
    source = P1 / "experiments/current_leaders_grouped_v1/comparison.csv"
    r = next(r for r in rows(source) if r["name"] == "lda_ensemble")
    add(
        "LDA ensemble 70/15/15",
        "lda-ensemble-trainval235.json",
        {k: float(r[k]) for k in POOLS},
        source,
    )
    return sorted(result, key=lambda r: -r["kaggle"])


def audit_submissions():
    result = []
    sample = [r["ID"] for r in rows(P1 / "data/part1_sample_submission.csv")]
    for p in sorted((P1 / "submissions/results").glob("*.json")):
        r = read(p)
        csvfile = resolve(r["submission_file"])
        recipe = resolve(r["frozen_recipe_file"])
        assert sha(csvfile) == r["submission_sha256"], p
        assert sha(recipe) == r["frozen_recipe_sha256"], p
        tag = r["tag"]
        assert git("cat-file", "-t", "refs/tags/" + tag).strip() == b"tag"
        commit = git("rev-parse", tag + "^{commit}").decode().strip()
        assert (
            git("show", tag + ":" + str(csvfile.relative_to(ROOT)))
            == csvfile.read_bytes()
        )
        assert git("show", tag + ":" + str(p.relative_to(ROOT))) == p.read_bytes()
        submitted = rows(csvfile)
        assert [v["ID"] for v in submitted] == sample and len(sample) == 312
        result.append(
            dict(
                recipe=r["recipe"],
                reported_kaggle=float(r["reported_kaggle_score"]),
                selected_for_report=r["recipe"] == WINNER,
                score_record=str(p.relative_to(P1)),
                submission_file=str(csvfile.relative_to(P1)),
                submission_sha256=r["submission_sha256"],
                tag=tag,
                result_commit=commit,
            )
        )
    assert sum(r["selected_for_report"] for r in result) == 1
    return sorted(result, key=lambda r: -r["reported_kaggle"])


def main():
    ms = measurements()
    submissions = audit_submissions()
    fields = (
        ["model", "recipe"]
        + POOLS
        + ["kaggle", "metric_source", "supplementary_source", "score_record"]
    )
    table(HERE / "benchmark_comparison.csv", ms, fields)
    table(HERE / "scored_submissions.csv", submissions, list(submissions[0]))
    text = """# Part 1 report guide

The selected winner is **80% anchored / 20% strict-inductive five-seed Cand2**, user-reported Kaggle **0.89935**. This page is generated from retained measurement tables and immutable scored records by `python part1/reporting/build.py`. It does not fit models. [Complete method](MODEL.md) · [Report handoff](REPORT_HANDOFF.md) · [Source tables](reporting/benchmark_comparison.csv) · [Exact submissions](reporting/scored_submissions.csv).

## Comparable results

Every CV column below uses all eight saved grouped repeats. Validation is the original 157-label fit evaluated on 78 rows. Chronology pools 98 predictions from three fixed earlier-to-later windows. These are sample-wise F1 values. Kaggle numbers are user-reported, not independently queried.

| Model | Train157 | Pooled235 | Validation 78 | Chronology98 | Kaggle |
|---|---:|---:|---:|---:|---:|
"""
    for r in ms:
        text += (
            "| "
            + r["model"]
            + " | "
            + " | ".join(f"{r[k]:.6f}" for k in POOLS)
            + f" | {r['kaggle']:.5f} |\n"
        )
    text += """
The winner improves anchored by .003185 on Train157, .000266 on Pooled235, ties validation, and improves chronology by .003401. The reported Kaggle gain is .00299 over anchored and .00085 over the previous 75/25 blend. The earlier LDA blend retains the higher Pooled235 mean.

Its paired wins/ties/losses versus anchored are **5/1/2 on Train157 and 4/1/3 on Pooled235**. It therefore failed the original requirement of at least five wins in both pools. The subsequent manual trial, explicitly authorized despite that result, provided the highest reported Kaggle score. Preserve that history instead of rewriting the advancement rule.

## Facts to keep distinct

- The final artifact uses 235 gold labels and 1,014 originally masked training rows. The original validation result belongs to the earlier 157-label fit; it is not independent validation of that final artifact.
- The winner's local blend measurements recombine previously verified held-out component scores. They are not new fits of both components for every blend. Its final export refit anchored and reused verified five-seed Cand2 final scores.
- The assignment's missing-label generation is retained. Anchored uses clustering-assisted self-training; Cand2 adds graph-generated labels and learned cluster/neighbor features. Test embeddings enter only prediction under the strict adapter.
- The supplied Cand2 manifest retains its original `transductive: true` flag as provenance. The submitted strict adapter overrides that access. Do not attach native/transductive scores to the strict submitted artifact.
- The final blend uses fixed arithmetic raw scores and cutoff .3 plus argmax. It uses neither log-odds pooling nor a learned final combiner. Standalone Cand2 used a different expected-F1 decoder.
- All labels/splits are reused development evidence. Overlapping CV repeats are not independent significance tests. The Kaggle partition and actual upload bytes were not independently inspected.
- The PySR experiment's declared parent was the earlier .89850 blend. Its reported .89423 is a separate artifact from original Ani's numerically identical .89423.
- The observed .89316 complementary-head regression does not isolate the diversity penalty's Kaggle effect; its matched ordinary-kernel control was not submitted. Joint neural training was not performed.

## Required label-level reporting

For the winner's validation result, use `experiments/cand2_followup_v1/results/label_metrics.csv`, selecting `track=current`, `kind=validation`, `recipe=blend`. The complete repeat and window metrics are in the adjacent `repeat_metrics.csv`. Original validation label predictions match anchored: per-label F1 Safety .941176, Funding .666667, Efficacy .758621, Recruitment .990099, Other .470588; macro F1 .765430 and sample-wise F1 .891026. Their supports are 9, 4, 14, 50, 10. Do not present these as hidden-test label scores.

## Experimental story and limitations

Start with the embedding-based anchored classifier, then explain why adding a small contribution from the graph/cluster/neighbor Cand2 pipeline helped even though standalone Cand2 scored worse. Keep the previous LDA blend as a strong control. Summarize later negative directions: log-odds pooling, altered pseudo-label treatment for Other, learned combiners, explicit diversity penalties, and bounded symbolic regression.

The [research index](RESEARCH_INDEX.html) links each recipe, fixed settings, full results and outcome. The original J audit is a small selected historical panel, not proof of a general-purpose ranking metric. Its ungrouped CV and chronology boundaries differ from the current grouped/earlier-only checks. Use [the matched audit](experiments/cand2_followup_v1/README.md) for those exact definitions and counts; do not mix J and grouped sample-F1 values into one ranking.

Use the selected artifact and method as the report conclusion. No additional model search or submission is queued. Historical studies keep their original preparation-time conclusions; later result records and this guide determine current status.
"""
    (P1 / "REPORT_GUIDE.md").write_text(text)
    result = dict(
        passed=True,
        scored_artifacts=len(submissions),
        rows_per_artifact=312,
        exact_csv_recipe_hashes_and_annotated_tags_checked=True,
        comparison_rows=len(ms),
        selected_recipe=WINNER,
        selected_score=0.89935,
        model_fits=0,
    )
    (HERE / "artifact_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
