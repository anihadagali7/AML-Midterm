"""Independently audit saved fold predictions, OOF assembly and supplementary scores.

Uses manual confusion-count metrics without the experiment's scoring functions.
No model fitting and no writes outside this study's verification record.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PART1 = HERE.parents[1]
ROOT = PART1 / "research/independent"


def read(path):
    return json.loads(path.read_text())


def check_metrics(truth, predictions, record):
    score = np.mean(2 * (truth * predictions).sum(1) / (truth.sum(1) + predictions.sum(1)))
    assert abs(score - record["sample_f1"]) < 1e-12
    tp = (truth * predictions).sum(0)
    fp = ((1 - truth) * predictions).sum(0)
    fn = (truth * (1 - predictions)).sum(0)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    assert abs(f1.mean() - record["macro_f1"]) < 1e-12
    for j, label in enumerate(["Safety", "Funding", "Efficacy", "Recruitment", "Other"]):
        expected = record["reasons"][label]
        assert abs(tp[j] / max(tp[j] + fp[j], 1) - expected["precision"]) < 1e-12
        assert abs(tp[j] / max(tp[j] + fn[j], 1) - expected["recall"]) < 1e-12
        assert abs(f1[j] - expected["f1"]) < 1e-12
    return float(score)


def decoder(scores):
    prediction = scores >= .5
    for i in range(len(prediction)):
        if not prediction[i].any():
            prediction[i, int(np.argmax(scores[i]))] = True
    return prediction.astype(int)


def main():
    protocol = read(HERE / "protocol.json")
    fold_arrays = repeated_scores = supplementary_scores = 0
    for name, expected in protocol["source_sha256"].items():
        assert hashlib.sha256((PART1 / name).read_bytes()).hexdigest() == expected
    for stage, source in protocol["folds"].items():
        directory = HERE / stage
        if not (directory / "index.json").exists():
            continue
        plan = read(ROOT / "experiments" / source)
        for row in read(directory / "index.json"):
            with np.load(directory / (row["candidate"] + ".npz")) as saved:
                assert saved["ids"].tolist() == plan["gold_ids"]
                values = []
                for ri, seed in enumerate(plan["seeds"]):
                    predictions = decoder(saved["scores"][ri])
                    np.testing.assert_array_equal(predictions, saved["predictions"][ri])
                    values.append(check_metrics(saved["truth"], predictions, row["repeat_scores"][ri]))
                    repeated_scores += 1
                    seen = set()
                    for fold in [f for f in plan["folds"] if f["seed"] == seed]:
                        positions = fold["heldout_gold_positions"]
                        assert not seen & set(positions)
                        seen.update(positions)
                        with np.load(directory / f"seed{seed}_fold{fold['fold']}.npz") as part:
                            assert part["ids"].tolist() == saved["ids"][positions].tolist()
                            np.testing.assert_allclose(part[row["candidate"]], saved["scores"][ri, positions], atol=1e-12, rtol=0)
                            fold_arrays += 1
                    assert seen == set(range(len(saved["ids"])))
                assert abs(np.mean(values) - row["mean"]) < 1e-12
                assert abs(np.std(values) - row["sd"]) < 1e-12
    for record in read(HERE / "supplementary/results.json"):
        path = HERE / "supplementary" / f"window{record['window']}_{record['candidate']}.npz"
        with np.load(path) as saved:
            predictions = decoder(saved["scores"])
            np.testing.assert_array_equal(predictions, saved["predictions"])
            check_metrics(saved["truth"], predictions, record)
            supplementary_scores += 1
    result = dict(passed=True, manually_rescored_repeats=repeated_scores,
                  fold_arrays_checked=fold_arrays, supplementary_scores_checked=supplementary_scores,
                  source_hashes_unchanged=True, row_coverage_and_alignment=True,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE / "independent_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
