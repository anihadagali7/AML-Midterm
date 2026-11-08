"""Rescore the frozen J audit while preserving later documentation/ledger updates.

Only RETENTION.md and submissions/index.csv use their historical bytes. Every
other protected input is verified from the current checkout before rescoring.
The original verifier runs unchanged in a temporary symlink tree; its output
cannot overwrite the archived verification record.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path("part1/experiments/derived_benchmark_audit_v1")
HISTORICAL = ("part1/RETENTION.md", "part1/submissions/index.csv")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    protocol = json.loads((ROOT / STUDY / "protocol.json").read_text())
    replacements = {}
    for name, expected in protocol["source_hashes"].items():
        if name in HISTORICAL:
            data = subprocess.check_output(
                ["git", "-C", str(ROOT), "show", protocol["base_commit"] + ":" + name]
            )
            replacements[name] = data
        else:
            data = (ROOT / name).read_bytes()
        assert digest(data) == expected, name
    replacements[str(STUDY / "verify.py")] = (ROOT / STUDY / "verify.py").read_bytes()
    replacements[str(STUDY / "verification.json")] = None

    def mirror(source, target, relative):
        key = relative.as_posix()
        if key in replacements:
            if replacements[key] is not None:
                target.write_bytes(replacements[key])
        elif any(name.startswith(key + "/") for name in replacements):
            target.mkdir()
            for child in source.iterdir():
                mirror(child, target / child.name, relative / child.name)
        else:
            target.symlink_to(source, target_is_directory=source.is_dir())

    with tempfile.TemporaryDirectory(prefix="part1-derived-check-") as folder:
        shadow = Path(folder)
        mirror(ROOT / "part1", shadow / "part1", Path("part1"))
        result = subprocess.run(
            [sys.executable, str(shadow / STUDY / "verify.py")],
            cwd=shadow,
            check=True,
            capture_output=True,
            text=True,
        )
        checked = json.loads(result.stdout)
    print(
        json.dumps(
            dict(
                passed=checked["status"] == "passed",
                case_prediction_rescores=checked["case_prediction_rescores"],
                live_protected_inputs=len(protocol["source_hashes"]) - len(HISTORICAL),
                historical_documentation_and_ledger=list(HISTORICAL),
                historical_commit=protocol["base_commit"],
                frozen_protocol_and_verifier_unchanged=True,
                archived_outputs_rewritten=False,
                models_fitted=0,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
