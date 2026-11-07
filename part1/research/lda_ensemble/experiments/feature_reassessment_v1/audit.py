"""Inspect saved OOF errors and Part 1 metadata; fit no models.

Run with the recorded research environment. Writes only this study's findings.json.
Only Part 1 inputs and saved Part 1 predictions are read.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PART1 = HERE.parents[1]
REPO = PART1.parent
RESEARCH = PART1 / "research/independent"
LABELS = ["Safety", "Funding", "Efficacy", "Recruitment", "Other"]
SOURCES = {}


def record(path):
    SOURCES[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_csv(path, **kwargs):
    return pd.read_csv(record(path), **kwargs)


def saved_predictions(candidate, gold):
    parts = []
    for stage in ["pooled3", "pooled5"]:
        folder = RESEARCH / "experiments/ani_improvement_v1" / stage
        index = json.loads(record(folder / "index.json").read_text())
        row = next(item for item in index if item["candidate"] == candidate)
        with np.load(record(folder / (row["result_key"] + ".npz")), allow_pickle=True) as saved:
            assert saved["ids"].tolist() == gold.ID.tolist()
            np.testing.assert_array_equal(saved["truth"], gold[LABELS].to_numpy())
            assert len(saved["predictions"]) == int(stage[-1])
            parts.append(saved["predictions"].copy())
    return np.concatenate(parts)


def sample_scores(prediction, truth):
    denominator = prediction.sum(-1) + truth.sum(-1)[None, :]
    assert (denominator > 0).all()
    return 2 * (prediction * truth[None, :, :]).sum(-1) / denominator


def main():
    frames = {
        split: read_csv(PART1 / f"data/part1_{split}.csv")
        for split in ["train", "val", "test"]
    }
    train_gold = frames["train"].loc[frames["train"][LABELS].ge(0).all(axis=1)]
    gold = pd.concat([train_gold, frames["val"]], ignore_index=True)
    truth = gold[LABELS].to_numpy(int)
    ani = saved_predictions("Ani frozen control", gold)
    three_way = saved_predictions("Ani + tri/SVM 15% each", gold)
    exact = lambda pred: (pred == truth[None, :, :]).all(axis=2).sum(axis=0)
    persistent = (exact(ani) <= 2) & (exact(three_way) <= 2)
    sample = sample_scores(three_way, truth)
    loss = 1 - sample
    baseline = float(sample.mean())
    label_diagnostics = {}
    for j, label in enumerate(LABELS):
        prediction = three_way[:, :, j]
        actual = truth[:, j][None, :]
        tp = (prediction * actual).sum(axis=1)
        fp = (prediction * (1 - actual)).sum(axis=1)
        fn = ((1 - prediction) * actual).sum(axis=1)
        oracle = three_way.copy()
        oracle[:, :, j] = truth[:, j]
        label_diagnostics[label] = {
            "gold_positives": int(actual.sum()),
            "mean_precision": float(np.mean(tp / np.maximum(tp + fp, 1))),
            "mean_recall": float(np.mean(tp / np.maximum(tp + fn, 1))),
            "mean_label_f1": float(np.mean(2 * tp / np.maximum(2 * tp + fp + fn, 1))),
            "diagnostic_perfect_label_replacement_sample_f1_gain": float(
                sample_scores(oracle, truth).mean() - baseline
            ),
        }

    coverage = {}
    cohorts = {"train157": train_gold, "validation78": frames["val"],
               "test312": frames["test"]}
    for name, frame in cohorts.items():
        text = frame.criteria.fillna("")
        coverage[name] = {
            "rows": len(frame), "median_start_year": float(frame.start_year.median()),
            "matched_drug_rate": float(frame.matched_drugs.notna().mean()),
            "criteria_median_characters": float(text.str.len().median()),
            "explicit_inclusion_heading": int(text.str.contains(
                r"inclusion\s+criteria", case=False, regex=True).sum()),
            "explicit_exclusion_heading": int(text.str.contains(
                r"exclusion\s+criteria", case=False, regex=True).sum()),
        }
    label_sets = [" ".join(label for label, flag in zip(LABELS, row) if flag)
                  for row in truth[persistent]]
    result = {
        "purpose": "Feature-priority investigation using existing eight-repeat pooled OOF predictions",
        "models_fitted": 0, "submissions_created": 0,
        "three_way_pooled_sample_f1": baseline,
        "persistent_errors": {
            "definition": "Both Ani and three-way fail exact label set on at least six of eight repeats",
            "rows": int(persistent.sum()),
            "ids": gold.loc[persistent, "ID"].tolist(),
            "actual_label_sets": pd.Series(label_sets).value_counts().to_dict(),
            "share_of_total_three_way_sample_f1_loss": float(loss[:, persistent].sum() / loss.sum()),
        },
        "label_diagnostics": label_diagnostics,
        "oracle_note": "Perfect single-label replacement uses known answers only to locate error budget. It is not a model, performance forecast, or an attainable-gain claim; gains need not be additive.",
        "input_scope": "Part 1 only; no Part 2 phase or outcome data.",
        "input_coverage": coverage,
        "limitations": "Development labels and folds have already influenced selection. Coverage and retrospective error concentration do not establish predictive gains or explain leaderboard competitors.",
        "source_sha256": SOURCES,
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    for name, digest in SOURCES.items():
        assert hashlib.sha256((REPO / name).read_bytes()).hexdigest() == digest
    (HERE / "findings.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in [
        "models_fitted", "three_way_pooled_sample_f1", "persistent_errors", "input_scope"
    ]}, indent=2))


if __name__ == "__main__":
    main()
