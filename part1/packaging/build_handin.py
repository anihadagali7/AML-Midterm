"""Build the standalone, report-focused Part 1 notebook hand-in; preserve exact scored artifacts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

REPO = Path(__file__).resolve().parents[2]
P1 = REPO / "part1"
HANDIN = REPO / "submission/midterm/part1"
FAMILIES = [
    "experiments/cand2_submission_v1",
    "experiments/cand2_followup_v1",
    "experiments/current_leaders_grouped_v1",
    "experiments/derived_benchmark_audit_v1",
    "experiments/blend_refinement_v1",
    "experiments/blend_refinement_v2",
    "experiments/anchored_graph_components_v1",
    "research/anchored_transductive_v1",
    "research/anchored_neighbor_blend_v1",
    "research/other_combiner_v1",
    "research/other_combiner_v2",
    "research/partial_secondary_v1",
    "research/complementary_heads_v1",
    "research/complementary_heads_v2",
    "research/pysr_v1",
    "research/pysr_blend_submission_v1",
    "experiments/method_comparison",
    "experiments/additional_methods",
    "experiments/classifier_features",
    "experiments/classifier_feature_tuning",
    "experiments/remaining_classifier_features",
    "experiments/self_training",
]
NOTEBOOKS = [
    "method_comparison.ipynb",
    "additional_methods.ipynb",
    "classifier_features.ipynb",
    "classifier_feature_tuning.ipynb",
    "remaining_classifier_features.ipynb",
    "self_training.ipynb",
]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def inventory():
    files = set((P1 / "data").glob("*.csv"))
    files.update(
        P1 / name
        for name in [
            "anchored_solution.py",
            "MODEL.md",
            "TESTING_POLICY.md",
            "requirements-simple.txt",
        ]
    )
    files.update(P1 / name for name in NOTEBOOKS)
    for family in FAMILIES:
        for p in (P1 / family).rglob("*"):
            if not p.is_file() or "__pycache__" in p.parts:
                continue
            relative = p.relative_to(P1 / family)
            # Keep original code/settings and compact result tables. No estimator dumps,
            # repeated generated datasets, broad row audits, or all-family cache copies.
            if {
                "datasets",
                "models",
                "classifier_models",
                "labeling_models",
                "label_audits",
                "profiles",
                "vocabularies",
                "clusters",
                "predictions",
                "validation_predictions",
            }.intersection(relative.parts):
                continue
            if p.suffix in {".py", ".md", ".txt", ".toml", ".csv", ".json"}:
                if p.stat().st_size > 1_000_000 and p.suffix == ".json":
                    continue
                files.add(p)
    # Exact packets used by the finalized notebooks and fresh reconstruction checks.
    follow = P1 / "experiments/cand2_followup_v1"
    files.update((follow / "inputs").glob("strict_inductive__*.npz"))
    files.update(
        follow / name
        for name in ["blend_final_predictions.npz", "cand2_final_predictions.npz"]
    )
    files.update(
        P1 / name
        for name in [
            "experiments/cand2_submission_v1/validation_reference.npz",
            "research/partial_secondary_v1/results/retained_comparison.npz",
            "research/other_combiner_v2/results/retained_comparison.npz",
            "research/complementary_heads_v2/results/matched_predictions.npz",
            "experiments/blend_refinement_v2/evaluation_inputs.npz",
        ]
    )
    for p in (P1 / "submissions/results").glob("*.json"):
        files.add(p)
        r = json.loads(p.read_text())
        for key in ["submission_file", "frozen_recipe_file"]:
            name = r[key]
            files.add(REPO / name if name.startswith("part1/") else P1 / name)
    files.update((P1 / "reporting").glob("*.csv"))
    return sorted(files)


def materialize():
    target = HANDIN / "resources"
    if target.exists() and not (target / "PACKAGE_MANIFEST.json").exists():
        raise RuntimeError("Refusing to replace an unrecognized resources directory")
    with tempfile.TemporaryDirectory(prefix="part1-handin-stage-") as tmp:
        stage = Path(tmp) / "resources"
        stage.mkdir()
        entries = {}
        for p in inventory():
            assert p.is_file() and p.resolve().is_relative_to(REPO), p
            name = str(p.relative_to(REPO))
            dest = stage / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            entries[name] = dict(sha256=sha(dest), bytes=dest.stat().st_size)
        for name, src in [
            ("handin.py", Path(__file__).with_name("handin.py")),
            (
                "legacy_feature_model.ipynb",
                REPO / "submission/midterm/part1_best_model.ipynb",
            ),
        ]:
            shutil.copy2(src, stage / name)
            entries[name] = dict(
                sha256=sha(stage / name), bytes=(stage / name).stat().st_size
            )
        manifest = dict(
            scope="Report-focused Part 1: winning model, representative feature/baseline comparisons and failed directions.",
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            note="Exact source/data hashes identify working-tree contents. Historical runners/notebooks are source records; the five finalized notebooks are portable. No claim that every archived search runs without omitted intermediates.",
            files=entries,
        )
        (stage / "PACKAGE_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(stage, target)
    return entries


def execute(directory):
    import nbformat
    from nbclient import NotebookClient

    records = []
    with tempfile.TemporaryDirectory(prefix="part1-notebook-kernel-") as tmp:
        kernel = Path(tmp) / "kernels/part1-handin"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(
            json.dumps(
                dict(
                    argv=[
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    display_name="Part 1 hand-in",
                    language="python",
                )
            )
        )
        old = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = tmp + (os.pathsep + old if old else "")
        try:
            for path in sorted(directory.glob("0*.ipynb")):
                print("Executing", path.name, flush=True)
                nb = nbformat.read(path, as_version=4)
                NotebookClient(
                    nb,
                    timeout=1800,
                    kernel_name="part1-handin",
                    resources={"metadata": {"path": str(directory)}},
                ).execute()
                nbformat.write(nb, path)
                records.append(
                    dict(
                        notebook=path.name,
                        code_cells=sum(c.cell_type == "code" for c in nb.cells),
                        errors=0,
                    )
                )
        finally:
            if old is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = old
    return records


def archive():
    destination = REPO / "submission/dist/part1-midterm-code.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = [
        p
        for p in HANDIN.rglob("*")
        if p.is_file()
        and not {"generated", "__pycache__", "_source_export"}.intersection(
            p.relative_to(HANDIN).parts
        )
    ]
    with zipfile.ZipFile(
        destination.with_suffix(".tmp"),
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as z:
        for p in sorted(files):
            assert not p.is_symlink()
            info = zipfile.ZipInfo(
                "midterm/part1/" + p.relative_to(HANDIN).as_posix(),
                date_time=(2026, 11, 7, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, p.read_bytes())
    destination.with_suffix(".tmp").replace(destination)
    print(
        "Portable ZIP:", destination, "bytes:", destination.stat().st_size, flush=True
    )
    return destination


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-execution", action="store_true")
    ap.add_argument("--zip-only", action="store_true")
    ap.add_argument(
        "--refresh-notebooks", action="store_true",
        help="Regenerate notebooks from templates, replacing manual notebook edits.",
    )
    args = ap.parse_args()
    if not args.zip_only:
        HANDIN.mkdir(parents=True, exist_ok=True)
        entries = materialize()
        expected = [
            "01_final_model.ipynb", "02_baselines_and_features.ipynb",
            "03_benchmarks_and_blends.ipynb", "04_unsuccessful_directions.ipynb",
            "05_source_code.ipynb",
        ]
        if args.refresh_notebooks or not all((HANDIN / name).is_file() for name in expected):
            from notebooks import build

            build(REPO, entries)
        if not args.skip_execution:
            records = execute(HANDIN)
            results = HANDIN / "results"
            results.mkdir(exist_ok=True)
            for p in (HANDIN / "generated").glob("*"):
                if p.is_file() and p.suffix in [".csv", ".png", ".json"]:
                    shutil.copy2(p, results / p.name)
            (results / "notebook_verification.json").write_text(
                json.dumps(
                    dict(
                        notebooks=records,
                        default_execution="Fresh validation and final235 winner fitting (all five Cand2 seeds), fresh historical feature/baseline validation, and retained benchmark/negative-evidence rescoring.",
                    ),
                    indent=2,
                )
                + "\n"
            )
    archive()


if __name__ == "__main__":
    main()
