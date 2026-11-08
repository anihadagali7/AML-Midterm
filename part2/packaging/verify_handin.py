"""Extract the Part 2 ZIP away from Git and execute all three notebooks."""
from pathlib import Path
import argparse
import hashlib
import json
import tempfile
import zipfile


def verify(archive):
    import nbformat
    from nbclient import NotebookClient
    archive = Path(archive).resolve()
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="isolated-part2-handin-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(archive) as zipped:
            assert zipped.testzip() is None
            for name in zipped.namelist():
                assert name.startswith("midterm/part2/") and ".." not in Path(name).parts
                assert not name.endswith(".joblib")
            zipped.extractall(root)
        handin = root / "midterm/part2"
        resources = handin / "resources"
        assert not (root / ".git").exists() and not (root / "part1").exists()
        assert not any(p.is_symlink() for p in root.rglob("*"))
        manifest = json.loads((resources / "PACKAGE_MANIFEST.json").read_text())
        def check_resources():
            for relative, item in manifest["files"].items():
                path = resources / relative
                assert path.is_file() and path.stat().st_size == item["bytes"], relative
                assert sha(path) == item["sha256"], relative
        check_resources()
        records = []
        for path in sorted(handin.glob("0*.ipynb")):
            print(f"Standalone execution: {path.name}", flush=True)
            nb = nbformat.read(path, as_version=4)
            NotebookClient(nb, timeout=600, kernel_name="python3",
                           resources={"metadata": {"path": str(handin)}}).execute()
            records.append(dict(notebook=path.name, errors=0))
        check_resources()
        winner = sha(handin / "part2_best_submission.csv")
        assert winner == "be29fc849f63ed4c23b31e67c97aaa97c6940f569b0af0920cf9799b428f7636"
        assert not list(resources.rglob("*.joblib"))
        return dict(archive=archive.name, archive_sha256=sha(archive),
                    archive_bytes=archive.stat().st_size, notebooks=records,
                    resource_files_verified=len(manifest["files"]),
                    winner_sha256=winner, all_resource_bytes_unchanged=True,
                    required_git=False, required_part1_checkout=False,
                    required_existing_model_caches=False,
                    fresh_fit_receipt=json.loads((resources / "experiments/final_handin/refit_verification.json").read_text()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "fresh_fit_receipt"}, indent=2))
