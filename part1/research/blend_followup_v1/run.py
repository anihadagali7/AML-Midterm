"""Small fixed blend/decoder follow-up; all 84 matched cases, no new fitting."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PART1, ROOT = HERE.parents[1], HERE.parents[2]
PRIOR = HERE.parent/"anchored_neighbor_blend_v1"
FINAL = HERE.parent/"anchored_lda_blend_submission_v1"
sys.path.insert(0, str(PRIOR))
import evidence as reference

LABELS = reference.LABELS
SPECS = {
    "anchored": dict(lda_weight=0., cutoff=.5),
    "current75_25": dict(lda_weight=.25, cutoff=.5),
    "lda_control": dict(lda_weight=1., cutoff=.5),
    "blend90_10": dict(lda_weight=.10, cutoff=.5),
    "blend80_20": dict(lda_weight=.20, cutoff=.5),
    "blend70_30": dict(lda_weight=.30, cutoff=.5),
    "blend60_40": dict(lda_weight=.40, cutoff=.5),
    "blend50_50": dict(lda_weight=.50, cutoff=.5),
    "blend75_25_cut475": dict(lda_weight=.25, cutoff=.475),
    "blend75_25_cut525": dict(lda_weight=.25, cutoff=.525),
}
CANDIDATES = list(SPECS)[3:]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def sources():
    paths = [Path(__file__), PRIOR/"evidence.py", PRIOR/"protocol.json", PRIOR/"retention.json",
             PRIOR/"results/useful_comparison.npz", PRIOR/"results/repeat_scores.csv",
             PRIOR/"results/supplementary_scores.csv", FINAL/"prediction_scores.npz", FINAL/"manifest.json",
             FINAL/"frozen_recipe.json", PART1/"submissions/results/anchored75-lda25-trainval235.json",
             PART1/"data/part1_sample_submission.csv", PART1/"submissions/ready/part1_anchored75_lda25_trainval235.csv"]
    inherited = json.loads((PRIOR/"protocol.json").read_text())["sources"]
    return inherited | {str(p.relative_to(ROOT)): sha(p) for p in paths}


def register():
    assert not (HERE/"protocol.json").exists()
    save(HERE/"protocol.json", dict(version="blend-followup-v1", registered_at=datetime.now(timezone.utc).isoformat(),
        authorization="User requested one or two improved submission candidates within 15 minutes.",
        parent="Anchored 0.89636 retained as policy parent; submitted 75/25 blend 0.89850 is primary practical comparator.",
        hypotheses=["Nearby fixed component weights may improve complementary error correction.",
                    "A small global decoder change may improve sample F1 by adjusting label-set size."],
        specs=SPECS, search_size="Seven fixed candidates, three controls. No crossed weight/threshold grid, per-label fitting, seed selection or test-label feedback.",
        schedule="Every recipe evaluated on all 80 saved CV folds, validation78, and three chronological windows. Reuse exact matched fold-local components; no new component model fits.",
        supplementary_exception="All seven candidates receive all supplementary checks regardless of CV outcome, fixed before scoring.",
        scoring="Complete OOF sample F1 per repeat, population SD, macro and all per-label metrics, paired repeats against both controls, every chronological window and pooled98.",
        strict_confirmation="Positive mean paired delta and at least 5/8 wins in both pools, reported separately against anchored and the current75_25 blend.",
        shortlist="Prefer improved mean CV in at least one pool with neither CV mean lower and no validation/chronology sample-F1 loss. Any tradeoff is explicit. Select at most two locally supported candidates, then check final prediction equality only to avoid duplicate submissions. No automatic promotion or upload.",
        access="Same inductive saved fits, group purges and query IDs as the verified prior study. Actual full235 test score arrays are opened only after finalist selection is recorded.",
        limitation="Small adaptive development search on reused labels, motivated by an earlier Kaggle result. Local ranking and Kaggle gain remain uncertain.",
        retention="Keep source, protocol and all compact results for every arm. Full prediction evidence for exported finalists; existing controls referenced without duplication.",
        source_sha256=sources()))


def guard():
    assert sources() == json.loads((HERE/"protocol.json").read_text())["source_sha256"]


def blend(a, b, spec):
    w = spec["lda_weight"]
    score = (1-w)*a+w*b
    p = (score >= spec["cutoff"]).astype(int)
    p[np.arange(len(p)), score.argmax(axis=1)] = 1
    independent = (score >= spec["cutoff"]).astype(int)
    for i in range(len(independent)):
        if not independent[i].any():
            independent[i, int(score[i].argmax())] = 1
    np.testing.assert_array_equal(p, independent)
    return score, p


def load_cv():
    with np.load(PRIOR/"results/useful_comparison.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    assert data["modes"].tolist() == ["anchored", "anchored75_lda25", "lda_control"]
    assert len(np.unique(data["case"])) == 84
    return data


def evaluate():
    guard()
    reference.verify(PRIOR)
    out = HERE/"results"
    assert not out.exists()
    out.mkdir()
    data = load_cv()
    records, supp, changes, errors = [], [], [], []
    for name, spec in SPECS.items():
        scores, pred = blend(data["scores"][0], data["scores"][2], spec)
        if name in ["anchored", "current75_25", "lda_control"]:
            index = ["anchored", "current75_25", "lda_control"].index(name)
            np.testing.assert_allclose(scores, data["scores"][index], atol=0, rtol=0)
            np.testing.assert_array_equal(pred, data["predictions"][index])
        for pool in ["train", "pooled"]:
            for seed in np.unique(data["seed"][data["pool"] == pool]):
                mask = (data["pool"] == pool) & (data["seed"] == seed)
                assert len(set(data["ids"][mask])) == mask.sum() == (157 if pool == "train" else 235)
                m = reference.study.benchmark.metrics(data["truth"][mask], pred[mask])
                manual = reference.metrics(data["truth"][mask], pred[mask])
                errors.extend(abs(m[k]-manual[k]) for k in m)
                records.append(dict(pool=pool, seed=int(seed), mode=name, **m))
        for case in list(np.unique(data["case"][data["pool"] == "supplementary"]))+["chronology98"]:
            mask = (data["pool"] == "supplementary") & (data["case"] != "validation") if case == "chronology98" else data["case"] == case
            m = reference.study.benchmark.metrics(data["truth"][mask], pred[mask])
            manual = reference.metrics(data["truth"][mask], pred[mask])
            errors.extend(abs(m[k]-manual[k]) for k in m)
            supp.append(dict(case=case, mode=name, **m))
        for case in np.unique(data["case"]):
            mask = data["case"] == case
            y = data["truth"][mask]
            delta = reference.row_f1(y, pred[mask])-reference.row_f1(y, data["predictions"][1, mask])
            changes.append(dict(case=case, mode=name, changed_labelsets=int((pred[mask] != data["predictions"][1, mask]).any(axis=1).sum()),
                                improved=int((delta > 1e-12).sum()), worsened=int((delta < -1e-12).sum())))
    assert len(errors) == 4620 and max(errors) < 1e-12
    repeats, supp = pd.DataFrame(records), pd.DataFrame(supp)
    summary = repeats.groupby(["pool", "mode"]).agg(mean_sample_f1=("sample_f1", "mean"),
        population_sd=("sample_f1", lambda s: s.std(ddof=0)), mean_macro_f1=("macro_f1", "mean")).reset_index()
    pairs, deltas = [], []
    for baseline in ["anchored", "current75_25"]:
        for pool in ["train", "pooled"]:
            pivot = repeats[repeats.pool == pool].pivot(index="seed", columns="mode", values="sample_f1")
            for name in CANDIDATES:
                d = pivot[name]-pivot[baseline]
                wins, ties, losses = int((d > 1e-12).sum()), int((abs(d) <= 1e-12).sum()), int((d < -1e-12).sum())
                pairs.append(dict(baseline=baseline, pool=pool, mode=name, mean_delta=float(d.mean()), wins=wins, ties=ties, losses=losses,
                                  meets_pool_rule=bool(d.mean() > 0 and wins >= 5)))
                deltas.extend(dict(baseline=baseline, pool=pool, mode=name, seed=int(s), sample_f1_delta=float(v)) for s, v in d.items())
    for name, table in [("repeat_scores", repeats), ("cv_summary", summary), ("supplementary_scores", supp),
                        ("paired_summary", pd.DataFrame(pairs)), ("paired_repeat_deltas", pd.DataFrame(deltas)),
                        ("prediction_changes", pd.DataFrame(changes))]:
        table.to_csv(out/(name+".csv"), index=False)
    save(out/"verification.json", dict(passed=True, new_component_fits=0, reused_component_cases=84,
        recipes_scored=10, complete_cv_repeat_records=160, supplementary_records=50,
        independent_metric_values_checked=len(errors), max_metric_error=max(errors),
        exact_control_predictions=True, protocol_sha256=sha(HERE/"protocol.json")))
    guard()
    print(summary.pivot(index="mode", columns="pool", values="mean_sample_f1").to_string())
    print(supp.pivot(index="mode", columns="case", values="sample_f1").to_string())
    print(pd.DataFrame(pairs).query("baseline == 'current75_25'").to_string(index=False))


def prepare(names):
    guard()
    assert 1 <= len(names) <= 2 and all(n in CANDIDATES for n in names)
    assert not (HERE/"selection.json").exists()
    assert json.loads((HERE/"results/verification.json").read_text())["passed"]
    save(HERE/"selection.json", dict(selected=names, specs={n:SPECS[n] for n in names},
        selected_at=datetime.now(timezone.utc).isoformat(), reason="Selected after reviewing all fixed local checks and paired repeats; see README for tradeoffs.",
        test_scores_opened_before_selection=False, upload_status="Prepared only; manual submission required"))
    cv = load_cv()
    cv_scores, cv_pred = zip(*(blend(cv["scores"][0], cv["scores"][2], SPECS[n]) for n in names))
    np.savez_compressed(HERE/"finalist_cv.npz", modes=np.array(names), scores=np.stack(cv_scores), predictions=np.stack(cv_pred),
        **{k:cv[k] for k in ["ids", "truth", "case", "pool", "seed"]})
    with np.load(FINAL/"prediction_scores.npz", allow_pickle=False) as z:
        final = {k:z[k] for k in z.files}
    sample = pd.read_csv(PART1/"data/part1_sample_submission.csv")
    np.testing.assert_array_equal(final["ids"], sample.ID)
    _, control = blend(final["anchored_centered"], final["lda_ensemble"], SPECS["current75_25"])
    np.testing.assert_array_equal(control, final["predictions"])
    old = pd.read_csv(PART1/"submissions/ready/part1_anchored75_lda25_trainval235.csv")
    encode = lambda p: [" ".join(label for label, flag in zip(LABELS, row) if flag) for row in p]
    assert encode(control) == old.Predicted.tolist()
    outputs, all_pred = [], [control]
    for name in names:
        score, pred = blend(final["anchored_centered"], final["lda_ensemble"], SPECS[name])
        assert not any(np.array_equal(pred, p) for p in all_pred), "Duplicate submission: do not spend a slot."
        all_pred.append(pred)
        path = PART1/"submissions/ready"/f"part1_{name}_trainval235.csv"
        assert not path.exists()
        frame = pd.DataFrame(dict(ID=sample.ID, Predicted=encode(pred)))
        frame.to_csv(path, index=False)
        pd.testing.assert_frame_equal(frame, pd.read_csv(path))
        assert frame.ID.is_unique and len(frame) == 312 and (pred.sum(axis=1) > 0).all()
        np.savez_compressed(HERE/(name+"_final.npz"), ids=final["ids"], scores=score, predictions=pred)
        outputs.append(dict(recipe=name, spec=SPECS[name], file=str(path.relative_to(ROOT)), sha256=sha(path),
            rows=312, changed_labelsets_vs_current=int((pred != control).any(axis=1).sum()),
            both_components_verified_labels=235, original_masked_rows=1014, new_model_fits=0,
            test_inputs_used_for_fit=False, kaggle_score=None, uploaded=False))
    save(HERE/"submissions.json", outputs)
    guard()
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["register", "evaluate", "prepare"])
    p.add_argument("--names", nargs="+")
    args = p.parse_args()
    if args.command == "register": register()
    elif args.command == "evaluate": evaluate()
    else: prepare(args.names)
