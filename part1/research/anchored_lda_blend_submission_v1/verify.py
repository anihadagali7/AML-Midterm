"""Verify prepared blend independently with CSV parsing and rational centering."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def main():
    manifest = json.loads((HERE/"manifest.json").read_text())
    recipe = json.loads((HERE/"frozen_recipe.json").read_text())
    assert manifest["frozen_recipe_sha256"] == sha(HERE/"frozen_recipe.json")
    for name, digest in recipe["source_sha256"].items():
        assert sha(ROOT/name) == digest, name
    for name, digest in manifest["artifacts"].items():
        assert sha(HERE/name) == digest, name
    path = ROOT/manifest["submission_file"]
    assert sha(path) == manifest["submission_sha256"]
    columns, submission = rows(path)
    _, sample = rows(ROOT/"part1/data/part1_sample_submission.csv")
    assert columns == ["ID", "Predicted"]
    ids = [row["ID"] for row in submission]
    assert ids == [row["ID"] for row in sample] and len(ids) == len(set(ids)) == 312
    labels = ["Safety", "Funding", "Efficacy", "Recruitment", "Other"]
    encoded = []
    for row in submission:
        selected = row["Predicted"].split()
        assert selected and len(selected) == len(set(selected)) and set(selected) <= set(labels)
        assert row["Predicted"] == " ".join(label for label in labels if label in selected)
        encoded.append([int(label in selected) for label in labels])
    with np.load(HERE/"prediction_scores.npz", allow_pickle=False) as z:
        assert z["ids"].tolist() == ids and z["labels"].tolist() == labels
        raw = np.clip(z["anchored_raw"], 1e-9, 1-1e-9)
        # Algebraically equivalent to the registered logit transform, computed independently.
        centered = .7*raw/(.7*raw+.3*(1-raw))
        np.testing.assert_allclose(centered, z["anchored_centered"], atol=1e-14, rtol=0)
        combined = .75*centered+.25*z["lda_ensemble"]
        np.testing.assert_allclose(combined, z["scores"], atol=1e-14, rtol=0)
        assert np.isfinite(combined).all() and combined.min() >= 0 and combined.max() <= 1
        expected = (combined >= .5).astype(int)
        for i in range(len(expected)):
            if not expected[i].any():
                expected[i, int(combined[i].argmax())] = 1
        np.testing.assert_array_equal(expected, encoded)
        np.testing.assert_array_equal(expected, z["predictions"])
    _, anchored = rows(ROOT/"part1/submissions/archive/submitted/anchored_reason_trainval235__kaggle_0.89636.csv")
    baseline = {row["ID"]: row["Predicted"] for row in anchored}
    changed = [row for row in submission if row["Predicted"] != baseline[row["ID"]]]
    assert len(changed) == manifest["changed_labelsets_vs_anchored"]
    _, saved_changes = rows(HERE/"changes_vs_anchored.csv")
    assert [row["ID"] for row in changed] == [row["ID"] for row in saved_changes]
    for new, saved in zip(changed, saved_changes):
        assert new["Predicted"] == saved["blend"] and saved["anchored"] == baseline[new["ID"]]
    report = dict(passed=True, rows=312, correct_id_order=True, independent_blend_and_decoder_match=True,
                  source_hashes_checked=len(recipe["source_sha256"]),
                  changed_labelsets_vs_anchored=len(changed), new_model_fits=0,
                  submission_sha256=manifest["submission_sha256"])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
