"""Verify retained legacy recipes and results without refitting or restoring caches."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def verify():
    record = json.loads((HERE / "manifest.json").read_text())
    counts = {}
    for name, expected in record["retained_files"].items():
        path = ROOT / name
        assert path.is_file(), name
        content = path.read_bytes()
        assert len(content) == expected["bytes"], name
        assert hashlib.sha256(content).hexdigest() == expected["sha256"], name
        study = Path(name).parts[2]
        rows, size = counts.get(study, (0, 0))
        counts[study] = (rows + 1, size + len(content))
    for study in record["studies"]:
        assert counts[study["study"]] == (study["kept_files"], study["kept_bytes"]), study
    for name, digest in record["source_files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    checks = json.loads((HERE / record["pre_cleanup_rescore"]).read_text())
    assert checks["passed"] and not checks["mismatches"]
    return {
        "passed": True,
        "retained_files_checked": len(record["retained_files"]),
        "recipe_sources_checked": len(record["source_files"]),
        "studies_checked": len(record["studies"]),
        "retained_bytes": sum(size for _, size in counts.values()),
        "pre_cleanup_checks": checks["totals"],
        "pre_cleanup_checks_repeated": False,
        "note": "Discarded predictions were rescored before cleanup; this run checks retained hashes and coverage. Use current_leaders_local.py verify to rescore current leaders.",
        "models_fitted": 0,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
