"""Check final CSV independently of model/scoring imports."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

OUT = Path(__file__).resolve().parent
PART1 = OUT.parents[1]
LABELS = ["Safety", "Funding", "Efficacy", "Recruitment", "Other"]


def read(path):
    return json.loads(path.read_text())


def main():
    recipe, manifest = read(OUT/"frozen_recipe.json"), read(OUT/"manifest.json")
    for name, digest in recipe["source_sha256"].items():
        assert hashlib.sha256((PART1/name).read_bytes()).hexdigest()==digest, name
    path = PART1/manifest["submission_file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest["submission_sha256"]
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames==["ID", "Predicted"]
        rows = list(reader)
    with (PART1/"data/part1_sample_submission.csv").open(newline="") as file:
        sample = list(csv.DictReader(file))
    ids = [row["ID"] for row in rows]
    assert ids==[row["ID"] for row in sample]
    assert len(ids)==len(set(ids))==312
    for row in rows:
        labels = row["Predicted"].split()
        assert labels and len(labels)==len(set(labels)) and set(labels)<=set(LABELS)
        assert row["Predicted"]==" ".join(label for label in LABELS if label in labels)
    encoded = np.array([[int(label in row["Predicted"].split()) for label in LABELS] for row in rows])
    with np.load(OUT/"prediction_scores.npz") as scores:
        assert scores["ids"].tolist()==ids
        p = scores["scores"]
        assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1
        expected = (p>=.5).astype(int)
        for i in range(len(expected)):
            if not expected[i].any():
                expected[i, int(np.argmax(p[i]))]=1
        np.testing.assert_array_equal(encoded, expected)
        np.testing.assert_array_equal(encoded, scores["predictions"])
        np.testing.assert_array_equal(p, scores["lda_ensemble__inductive"])
    audit = read(OUT/"fit_audit.json")
    assert len(audit["main"]["gold_ids"])==235
    assert len(audit["main"]["masked_ids"])==1014
    assert len(audit["main"]["fitting_ids"])==1249
    assert not set(audit["main"]["fitting_ids"])&set(ids)
    assert audit["main"]["target_input_rows"]==0
    assert all(v<1e-10 for v in audit["main"]["notebook_function_max_differences"].values())
    assert all(manifest["checks"].values())
    result = dict(passed=True, rows=312, gold_rows=235, masked_training_rows=1014,
        exact_sample_id_order=True, valid_nonempty_labels=True, manual_decoder_matches=True,
        protected_sources_unchanged=True, test_inputs_excluded_from_fit=True,
        submission_sha256=manifest["submission_sha256"], uploaded=False)
    (OUT/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__=="__main__":
    main()
