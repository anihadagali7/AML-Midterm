"""Materialize and verify a portable Part 2 hand-in without changing research files.

Usage: python part2/packaging/build_handin.py [--skip-execution]
The ZIP contains midterm/part2/. It does not claim to package the Part 1 hand-in.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile

REPO = Path(__file__).resolve().parents[2]
HANDIN = REPO / "submission/midterm/part2"
WINNER_SHA = "be29fc849f63ed4c23b31e67c97aaa97c6940f569b0af0920cf9799b428f7636"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory():
    raw = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "part2"], cwd=REPO)
    return sorted({REPO / os.fsdecode(p) for p in raw.split(b"\0") if p
                   if (REPO / os.fsdecode(p)).is_file()})


def include_resource(source):
    """Keep all inputs/code/trial summaries, plus evidence used by final notebooks.

    Per-fold predictions for every unsuccessful candidate, repeated fit-ID audits,
    and intermediate feature/base-prediction arrays remain in the working project.
    They are outputs that can be regenerated, not additional input datasets.
    """
    relative = source.relative_to(REPO / "part2")
    parts = relative.parts
    if parts[0] in {"logs", "report"}:
        return False
    if parts[0] != "experiments":
        return True
    if {"audits", "bases", "input_snapshots"}.intersection(parts):
        return False
    if source.suffix in {".log", ".joblib"}:
        return False
    if source.suffix == ".npz":
        return str(relative) in {
            "experiments/model_search/linear_validation_predictions.npz",
            "experiments/final_handin/recorded_components.npz",
        }
    if "predictions" in parts:
        return (str(relative).startswith("experiments/leaderboard_sprint/runs/4dc0f274d45bce9e/")
                and "gaps" not in parts
                and source.stem in {"ani_first", "ani_stack", "file10",
                                    "mix_legacy01_0.2", "mix_legacy01_0.3"})
    return True


def materialize(files):
    target = HANDIN / "resources"
    if target.exists() and not (target / "PACKAGE_MANIFEST.json").is_file():
        raise RuntimeError(f"Refusing to replace an unrecognized directory: {target}")
    with tempfile.TemporaryDirectory(prefix="part2-handin-") as temporary:
        stage = Path(temporary) / "resources"
        stage.mkdir()
        entries = {}
        included = [p for p in files if include_resource(p)]
        for source in included:
            relative = source.relative_to(REPO / "part2")
            # Existing report support files are repository-local symlinks. Copy
            # their contents so the exported hand-in needs no external paths.
            assert source.resolve().is_relative_to(REPO), source
            assert ".runtime" not in relative.parts and source.suffix != ".joblib", source
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            entries[str(relative)] = {"sha256": sha(destination), "bytes": destination.stat().st_size}
        manifest = dict(
            scope="Part 2 only; all supplied inputs, complete source, all trial settings/summaries, selected row predictions",
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            note="Includes current working-tree Part 2 changes; file hashes identify the packaged contents.",
            excluded="Intermediate arrays, repeated row predictions/audits, logs and report drafts stay in the working project; regenerate from included code and inputs.",
            working_files=len(files), included_files=len(included),
            omitted_working_files=len(files)-len(included),
            files=entries)
        (stage / "PACKAGE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(stage, target)
    assert sha(HANDIN / "part2_best_submission.csv") == WINNER_SHA
    return entries


def execute_notebooks():
    import nbformat
    from nbclient import NotebookClient
    records = []
    for path in sorted(HANDIN.glob("0*.ipynb")):
        print(f"Executing {path.name}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(notebook, timeout=600, kernel_name="python3",
                                resources={"metadata": {"path": str(HANDIN)}})
        client.execute()
        nbformat.write(notebook, path)
        records.append(dict(notebook=path.name, code_cells=sum(c.cell_type == "code" for c in notebook.cells),
                            errors=0))
    result = HANDIN / "results"
    result.mkdir(exist_ok=True)
    for path in (HANDIN / "generated").iterdir():
        if path.is_file() and path.suffix in [".csv", ".png"]:
            shutil.copy2(path, result / path.name)
    verification = dict(notebooks=records, expected_winner_sha256=WINNER_SHA,
                        source_files=len(list((HANDIN / "_source_export").rglob("*.py"))),
                        note="Default execution rescores saved predictions and replays recorded components; see fresh-fit receipt separately.")
    (result / "notebook_verification.json").write_text(json.dumps(verification, indent=2) + "\n")


def archive(entries):
    destination = REPO / "submission/dist/part2-midterm-code.zip"
    destination.parent.mkdir(exist_ok=True)
    paths = sorted([p for p in HANDIN.iterdir() if p.is_file() and p.suffix in [".ipynb", ".csv", ".md"]]
                   + list((HANDIN / "results").glob("*"))
                   + [HANDIN / "resources" / p for p in entries]
                   + [HANDIN / "resources/PACKAGE_MANIFEST.json"])
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in paths:
            assert path.is_file() and not path.is_symlink(), path
            relative = "midterm/part2/" + path.relative_to(HANDIN).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2026, 11, 7, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, path.read_bytes())
    temporary.replace(destination)
    print(f"Created {destination} ({destination.stat().st_size / 1e6:.1f} MB)", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-execution", action="store_true", help="Generate notebooks without executing them.")
    parser.add_argument("--zip-only", action="store_true", help="Repackage already executed notebooks and materialized resources.")
    parser.add_argument(
        "--refresh-notebooks", action="store_true",
        help="Regenerate notebooks from templates, replacing manual notebook edits.",
    )
    args = parser.parse_args()
    if args.zip_only:
        entries = json.loads((HANDIN / "resources/PACKAGE_MANIFEST.json").read_text())["files"]
    else:
        from notebooks import build
        files = inventory()
        print(f"Including {sum(include_resource(p) for p in files)} of {len(files)} working files; all input datasets and Python code retained.", flush=True)
        assert all(include_resource(p) for p in files if p.suffix == ".py")
        expected = ["01_final_model.ipynb", "02_experiments.ipynb", "03_source_code.ipynb"]
        if args.refresh_notebooks or not all((HANDIN / name).is_file() for name in expected):
            build(REPO, files)
        entries = materialize(files)
        if not args.skip_execution:
            execute_notebooks()
    archive(entries)


if __name__ == "__main__":
    main()
