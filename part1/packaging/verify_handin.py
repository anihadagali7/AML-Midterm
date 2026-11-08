"""Execute the exported Part 1 hand-in in isolation, including fresh winner fits."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from build_handin import REPO, execute

WINNER_SHA = "c20aa937fa8b1e5aea55bcd6564c6acd952f0d74030cc7c1a437eac54a1ee257"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    archive = REPO / "submission/dist/part1-midterm-code.zip"
    with tempfile.TemporaryDirectory(prefix="part1-isolated-handin-") as tmp:
        directory = Path(tmp)
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                relative = Path(info.filename)
                assert not relative.is_absolute() and ".." not in relative.parts
                assert info.filename.startswith("midterm/part1/")
                assert not ((info.external_attr >> 16) & 0o170000) == 0o120000
                assert relative.suffix not in {".pkl", ".pickle", ".joblib", ".pyc"}
            z.extractall(directory)
            file_count = len(z.infolist())
        package = directory / "midterm/part1"
        assert not (directory / ".git").exists()
        records = execute(package)
        assert sha(package / "part1_best_submission.csv") == WINNER_SHA
        assert sha(package / "generated/reconstructed_submission.csv") == WINNER_SHA
        assert sha(package / "generated/fresh_final/fresh_submission.csv") == WINNER_SHA
        validation = json.loads(
            (package / "generated/fresh_validation/receipt.json").read_text()
        )
        final = json.loads((package / "generated/fresh_final/receipt.json").read_text())
        assert validation["identical_predictions"] and final["identical_predictions"]
        assert validation["fresh_cand2"] and final["fresh_cand2"]
        assert validation["seeds"] == final["seeds"] == list(range(5))
        assert validation["gold"] == 157 and final["gold"] == 235
        assert not validation["test_inputs_in_fit"] and not final["test_inputs_in_fit"]
        completed_path = package / "generated/completed_training_reason_labels.csv"
        with completed_path.open() as handle:
            completed = list(csv.DictReader(handle))
        assert len(completed) == 1171
        assert sum(row["label_source"] == "supplied" for row in completed) == 157
        assert sum(row["label_source"] == "anchored_prediction" for row in completed) == 1014
        assert all(row[label] in {"0", "1"} for row in completed
                   for label in ["Safety", "Funding", "Efficacy", "Recruitment", "Other"])
        assert sha(completed_path) == sha(package / "results/completed_training_reason_labels.csv")
        # Preserve independent fresh-fit receipts, not the temporary working files.
        result = dict(
            passed=True,
            archive="submission/dist/part1-midterm-code.zip",
            zip_sha256=sha(archive),
            zip_bytes=archive.stat().st_size,
            zip_files=file_count,
            notebooks=records,
            extracted_without_git=True,
            extracted_without_research_checkout=True,
            exact_winner_sha256=WINNER_SHA,
            fresh_validation=validation,
            fresh_final=final,
            completed_training_labels=dict(rows=1171, supplied=157, generated=1014,
                                           binary_labels=True, packaged_export_reproduced=True),
            scope="Portable Part 1 package; no Part 2 or report changes. Default cells train the winner and historical feature comparison; benchmark/negative studies rescore retained evidence.",
        )
    (REPO / "submission/part1_package_verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
