"""Portable Part 1 notebook helpers. Frozen model implementations stay unchanged."""

from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
P1 = ROOT / "part1"
LABELS = ["Safety", "Funding", "Efficacy", "Recruitment", "Other"]
EMB = [f"emb_{i}" for i in range(150)]
WINNER_SHA = "c20aa937fa8b1e5aea55bcd6564c6acd952f0d74030cc7c1a437eac54a1ee257"
sys.path.insert(0, str(P1))
from anchored_solution import AnchoredReasonClassifier, select_reasons


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def data():
    return [
        pd.read_csv(P1 / "data" / f"part1_{name}.csv")
        for name in ["train", "val", "test"]
    ]


def metrics(y, prediction):
    precision, recall, f, support = precision_recall_fscore_support(
        y, prediction, zero_division=0
    )
    return dict(
        sample_f1=float(f1_score(y, prediction, average="samples", zero_division=0)),
        macro_f1=float(f.mean()),
    ), pd.DataFrame(
        dict(label=LABELS, precision=precision, recall=recall, f1=f, support=support)
    )


def frame(ids, prediction):
    return pd.DataFrame(
        dict(
            ID=ids,
            Predicted=[
                " ".join(l for l, yes in zip(LABELS, row) if yes) for row in prediction
            ],
        )
    )


def verify_resources():
    manifest = json.loads((ROOT / "PACKAGE_MANIFEST.json").read_text())
    for name, record in manifest["files"].items():
        assert sha(ROOT / name) == record["sha256"], name
    return dict(
        files=len(manifest["files"]),
        hashes_match=True,
        external_checkout_required=False,
    )


def fit_winner(validation=True, output=None):
    """Fresh anchored and all five Cand2 seeds; no stored scores enter training."""
    started = time.monotonic()
    study = P1 / "experiments/cand2_submission_v1"
    sys.path[:0] = [str(study / "source"), str(study)]
    spec = importlib.util.spec_from_file_location(
        "_frozen_cand2_export", study / "build.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    train, val, test = data()
    fit = train if validation else pd.concat([train, val], ignore_index=True)
    query = val if validation else test
    with threadpool_limits(limits=1):
        anchored = AnchoredReasonClassifier().fit(
            fit[EMB].to_numpy(float), fit[LABELS].to_numpy(int)
        )
        a = anchored.predict_proba(query[EMB].to_numpy(float))
        outputs = [module.one_seed((validation, seed)) for seed in range(5)]
        D = module.load(str(P1 / "data"))
        _, Q, _ = module.average(outputs, D.CM)
    c = Q @ D.CM
    score = 0.8 * a + 0.2 * c
    prediction = select_reasons(score)
    if validation:
        ref = arrays(
            P1 / "experiments/cand2_followup_v1/inputs/strict_inductive__validation.npz"
        )
        expected = select_reasons(
            0.8 * ref["anchored_scores"] + 0.2 * ref["cand2_scores"]
        )
    else:
        ref = arrays(P1 / "experiments/cand2_followup_v1/blend_final_predictions.npz")
        expected = ref["predictions"]
    np.testing.assert_array_equal(query.ID.to_numpy(str), ref["ids"])
    np.testing.assert_allclose(a, ref["anchored_scores"], atol=1e-7, rtol=0)
    np.testing.assert_allclose(c, ref["cand2_scores"], atol=1e-7, rtol=0)
    np.testing.assert_array_equal(prediction, expected)
    receipt = dict(
        validation=validation,
        gold=int(fit[LABELS].ge(0).all(axis=1).sum()),
        originally_masked=int(fit[LABELS].eq(-100).all(axis=1).sum()),
        rows=len(query),
        seeds=list(range(5)),
        fresh_anchored=True,
        fresh_cand2=True,
        test_inputs_in_fit=False,
        max_anchored_error=float(np.abs(a - ref["anchored_scores"]).max()),
        max_cand2_error=float(np.abs(c - ref["cand2_scores"]).max()),
        identical_predictions=True,
        seconds=time.monotonic() - started,
        anchored_refreshes=anchored.round_audit_,
    )
    result = frame(query.ID, prediction)
    if not validation:
        digest = hashlib.sha256(
            result.to_csv(index=False, lineterminator="\n").encode()
        ).hexdigest()
        assert digest == WINNER_SHA
        receipt["submission_sha256"] = digest
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        result.to_csv(
            output
            / ("validation_predictions.csv" if validation else "fresh_submission.csv"),
            index=False,
        )
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        # Preserve gold; retain -100 where anchored did not accept a pseudo-label.
        augmented = fit[["ID"] + LABELS].copy()
        missing = np.flatnonzero(fit[LABELS].eq(-100).all(axis=1))
        accepted = missing[anchored.generated_mask_]
        augmented.loc[accepted, LABELS] = anchored.generated_labels_
        augmented["label_source"] = np.where(
            fit[LABELS].ge(0).all(axis=1), "supplied", "unaccepted"
        )
        augmented.loc[accepted, "label_source"] = "anchored_pseudo"
        augmented.to_csv(output / "anchored_augmented_labels.csv", index=False)
    return receipt, result


def winner_repeats():
    """Recompute complete-repeat F1, never an unweighted mean of fold F1."""
    buckets = {}
    cases = []
    for path in sorted(
        (P1 / "experiments/cand2_followup_v1/inputs").glob("strict_inductive__*.npz")
    ):
        key = path.stem.removeprefix("strict_inductive__")
        if key.startswith("grouped_"):
            _, pool, seed, fold = key.split("_")
            repeat = int(seed)
        elif key == "validation":
            pool, repeat = "validation", -1
        elif key.startswith("through_"):
            pool, repeat = "chronology98", -1
        else:
            continue
        z = arrays(path)
        for recipe, pred in [
            ("anchored", z["anchored_predictions"]),
            ("cand2", z["cand2_predictions"]),
            (
                "winner",
                select_reasons(0.8 * z["anchored_scores"] + 0.2 * z["cand2_scores"]),
            ),
        ]:
            buckets.setdefault((pool, repeat, recipe), []).append(
                (z["ids"], z["truth"], pred)
            )
        cases.append(key)
    assert len(cases) == 84
    rows = []
    for (pool, seed, recipe), chunks in buckets.items():
        ids, y, pred = [np.concatenate([c[j] for c in chunks]) for j in range(3)]
        assert len(ids) == len(set(ids))
        assert (
            len(ids)
            == {"train": 157, "pooled": 235, "validation": 78, "chronology98": 98}[pool]
        )
        scores, _ = metrics(y, pred)
        rows.append(dict(pool=pool, seed=seed, recipe=recipe, rows=len(ids), **scores))
    return pd.DataFrame(rows)


def compact_rescore(relative):
    """Rescore useful matched predictions; omitted variants keep their full summaries."""
    z = arrays(P1 / relative)
    rows = []
    for pool in np.unique(z["pool"]):
        for seed in np.unique(z["seed"][z["pool"] == pool]):
            keep = (z["pool"] == pool) & (z["seed"] == seed)
            for j, name in enumerate(z["names"]):
                m, _ = metrics(z["truth"][keep], z["predictions"][j, keep])
                rows.append(dict(pool=str(pool), seed=int(seed), recipe=str(name), **m))
    chrono = np.char.startswith(z["pool"], "through_")
    if chrono.any():
        for j, name in enumerate(z["names"]):
            m, _ = metrics(z["truth"][chrono], z["predictions"][j, chrono])
            rows.append(dict(pool="chronology98", seed=-1, recipe=str(name), **m))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", choices=["validation", "final"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fit_winner(args.fresh == "validation", args.output)[0], indent=2))
