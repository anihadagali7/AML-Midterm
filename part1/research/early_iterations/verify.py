"""Check the compact import of earlier Part 1 work without fitting models."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def verify():
    manifest = json.loads((HERE / "manifest.json").read_text())
    size = 0
    for name, record in manifest["files"].items():
        data = (ROOT / name).read_bytes()
        assert len(data) == record["bytes"], name
        assert hashlib.sha256(data).hexdigest() == record["sha256"], name
        size += len(data)
    accounted = sum(manifest["coverage"].values()) + sum(
        manifest["omitted_generated_outputs_by_family"].values()
    )
    assert accounted == manifest["working_files_accounted_for"]
    original_checks = json.loads((HERE / manifest["pre_import_verification"]).read_text())
    assert original_checks["passed"]
    return {
        "passed": True,
        "files_checked": len(manifest["files"]),
        "bytes_checked": size,
        "working_files_accounted_for": accounted,
        "pre_import_repeat_scores_checked": original_checks["totals"]["recipe_repeat_scores"],
        "pre_import_scoring_repeated": False,
        "models_fitted": 0,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
