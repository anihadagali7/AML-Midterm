"""Check source isolation, notebook startup, and migrated navigation."""
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest

PROJECT = Path(os.environ.get("PART2_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
RUNNER = runpy.run_path(str(PROJECT / "run.py"))


class LayoutTests(unittest.TestCase):
    def fixture(self, root):
        for folder in ["src", "data", "tests", "notebooks", "experiments", "submissions"]:
            (root / folder).mkdir(parents=True)
        (root / "src/model.py").write_text("from pathlib import Path\nROOT = Path(__file__).resolve().parent\n")
        (root / "data/outcome_train.csv").write_text("ID,label\n1,0\n")
        (root / "layout.json").write_text(json.dumps({"legacy_paths": {
            "model.py": "src/model.py", "outcome_train.csv": "data/outcome_train.csv",
        }}))
        (root / "run.py").write_bytes((PROJECT / "run.py").read_bytes())

    def test_execution_copies_preserve_source_and_share_canonical_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="part2 layout ") as directory:
            root = Path(directory); self.fixture(root)
            workspace = RUNNER["prepare_workspace"](root)
            original = (root / "src/model.py").read_bytes()
            self.assertEqual((workspace / "model.py").read_bytes(), original)
            self.assertEqual(runpy.run_path(str(workspace / "model.py"))["ROOT"], workspace)
            self.assertEqual((workspace / "outcome_train.csv").resolve(), (root / "data/outcome_train.csv").resolve())
            self.assertEqual((workspace / "submissions").resolve(), (root / "submissions").resolve())
            (workspace / "model.py").write_text("# notebook-generated copy\n")
            self.assertEqual((root / "src/model.py").read_bytes(), original)
            (root / "src/new_trial.py").write_text("result = 1\n")
            RUNNER["prepare_workspace"](root)
            self.assertEqual((workspace / "model.py").read_bytes(), original)
            self.assertEqual((workspace / "new_trial.py").read_bytes(), (root / "src/new_trial.py").read_bytes())

    def test_notebook_bootstrap_keeps_original_cells_and_is_repeatable(self):
        with tempfile.TemporaryDirectory(prefix="part2 notebook ") as directory:
            root = Path(directory); self.fixture(root)
            path = root / "notebooks/example.ipynb"
            original = {"cell_type": "code", "id": "original", "source": ["print(1)"],
                        "execution_count": 1, "outputs": [{"output_type": "stream", "name": "stdout", "text": ["1\n"]}], "metadata": {}}
            path.write_text(json.dumps({"cells": [original], "nbformat": 4, "nbformat_minor": 5, "metadata": {}}))
            RUNNER["install_notebook_bootstrap"](root)
            once = path.read_bytes(); RUNNER["install_notebook_bootstrap"](root)
            self.assertEqual(path.read_bytes(), once)
            cells = json.loads(once)["cells"]
            self.assertEqual(cells[1:], [original])
            previous, imports = Path.cwd(), sys.path[:]
            try:
                os.chdir(path.parent)
                exec("".join(cells[0]["source"]), {})
                self.assertEqual(Path.cwd(), root.resolve() / ".runtime/part2")
            finally:
                os.chdir(previous); sys.path[:] = imports

    def test_report_references_translate_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.fixture(root)
            path = root / "experiments/RESULTS.md"
            text = ("[Code](../model.py)\npython part2/model.py --input part2/outcome_test.csv\n"
                    "python -m unittest discover -s part2 -p 'test_*.py'\n")
            updated = RUNNER["update_references"](text, path, path, root)
            self.assertIn("[Code](../src/model.py)", updated)
            self.assertIn("python part2/run.py model.py --input part2/data/outcome_test.csv", updated)
            self.assertIn("python part2/run.py test 'test_*.py'", updated)
            self.assertEqual(RUNNER["update_references"](updated, path, path, root), updated)


if __name__ == "__main__":
    unittest.main()
