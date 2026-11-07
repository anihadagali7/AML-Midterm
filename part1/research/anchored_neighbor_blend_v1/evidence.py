"""Independently rescore, retain useful evidence, or verify it without fitting.

The metric implementation uses TP/FP/FN arithmetic, not the fitting runner's
sklearn metrics. Retention never deletes its input directory.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import study

STUDY = Path(__file__).resolve().parent
LABELS = study.LABELS
KEEP = ["anchored", "anchored75_lda25", "lda_control"]
TABLES = ["repeat_scores.csv", "cv_summary.csv", "paired_summary.csv",
          "paired_repeat_deltas.csv", "supplementary_scores.csv", "verification.json"]


def ratio(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)


def row_f1(y, p):
    return ratio(2*(y*p).sum(axis=1), y.sum(axis=1)+p.sum(axis=1))


def metrics(y, p):
    assert np.isin(y, [0, 1]).all() and np.isin(p, [0, 1]).all()
    tp, fp, fn = (y*p).sum(axis=0), ((1-y)*p).sum(axis=0), (y*(1-p)).sum(axis=0)
    values = dict(precision=ratio(tp, tp+fp), recall=ratio(tp, tp+fn),
                  f1=ratio(2*tp, 2*tp+fp+fn), support=y.sum(axis=0))
    out = dict(sample_f1=float(row_f1(y, p).mean()), macro_f1=float(values["f1"].mean()))
    out.update({kind+"_"+label: float(values[kind][i])
                for kind in values for i, label in enumerate(LABELS)})
    return out


def read_raw(raw):
    arrays = {k: [] for k in ["case", "pool", "seed", "ids", "truth", "scores", "predictions", "parent_raw_scores"]}
    for case in study.benchmark.cases():
        with np.load(raw/"predictions"/(case["key"]+".npz"), allow_pickle=False) as z:
            np.testing.assert_array_equal(z["ids"], case["query"].ID)
            np.testing.assert_array_equal(z["truth"], case["query"][LABELS])
            for key in ["ids", "truth", "scores", "predictions", "parent_raw_scores"]:
                arrays[key].append(z[key])
            for key, value in [("case", case["key"]), ("pool", case["pool"]), ("seed", case["seed"] or 0)]:
                arrays[key].append(np.repeat(value, len(z["ids"])))
    return {k: np.concatenate(v, axis=1 if k in ["scores", "predictions"] else 0) for k, v in arrays.items()}


def rescore(data, modes, results):
    records, errors = 0, []
    repeats = pd.read_csv(results/"repeat_scores.csv")
    supplementary = pd.read_csv(results/"supplementary_scores.csv")
    for i, mode in enumerate(modes):
        np.testing.assert_array_equal(study.decide(data["scores"][i]), data["predictions"][i])
        for table in [repeats, supplementary]:
            for row in table[table["mode"] == mode].to_dict("records"):
                if "pool" in row:
                    mask = (data["pool"] == row["pool"]) & (data["seed"] == row["seed"])
                elif row["case"] == "chronology98":
                    mask = (data["pool"] == "supplementary") & (data["case"] != "validation")
                else:
                    mask = data["case"] == row["case"]
                assert mask.any() and len(np.unique(data["ids"][mask])) == int(mask.sum())
                actual = metrics(data["truth"][mask], data["predictions"][i, mask])
                for key, value in actual.items():
                    error = abs(value-row[key])
                    assert error < 1e-12, (mode, row, key, value)
                    errors.append(error)
                records += 1
    return dict(passed=True, method="Direct TP/FP/FN arithmetic; no sklearn metric calls",
                records_checked=records, metric_values_checked=len(errors),
                max_absolute_error=max(errors), modes=list(modes), models_fitted=0)


def changes(data, modes):
    rows = []
    for key in np.unique(data["case"]):
        mask = data["case"] == key
        y, p = data["truth"][mask], data["predictions"][:, mask]
        base = row_f1(y, p[0])
        for i, mode in enumerate(modes[1:], 1):
            d = row_f1(y, p[i])-base
            changed = (p[i] != p[0]).any(axis=1)
            rows.append(dict(case=key, pool=data["pool"][mask][0], mode=mode,
                             changed_labelsets=int(changed.sum()), improved=int((d > 1e-12).sum()),
                             worsened=int((d < -1e-12).sum()),
                             same_score_changed=int((changed & (abs(d) <= 1e-12)).sum()),
                             sum_row_f1_delta=float(d.sum())))
    return pd.DataFrame(rows)


def summarize_access(audits):
    assert len(audits) == 84
    assert all(not a["query_inputs_used_for_fit"] and not a["query_labels_used_for_fit"] for a in audits)
    records = []
    for a in audits:
        base = {k: a[k] for k in ["key", "pool", "seed", "input_rows", "gold_rows", "target_rows", "fit_groups", "fit_ids_sha256", "purged_masked"]}
        base["masked_rows"] = a["input_rows"]-a["gold_rows"]
        for model in a["models"]:
            row = dict(base, mode=model["mode"], generated_rows=model["rounds"][-1]["generated"])
            if "neighbors" in model:
                row.update({k: v for k, v in model["neighbors"].items() if k not in ["mode", "gold_rows", "generated_rows"]})
            records.append(row)
    return pd.DataFrame(records)


def retain(raw, destination):
    assert not (destination/"results").exists() and not (destination/"retention.json").exists()
    assert study.sources() == json.loads((STUDY/"protocol.json").read_text())["sources"]
    assert json.loads((raw/"verification.json").read_text())["passed"]
    data = read_raw(raw)
    # Verify all arms before retaining only the useful blend and its controls.
    independent = rescore(data, study.MODES, raw)
    assert independent["records_checked"] == 105 and independent["metric_values_checked"] == 2310
    results = destination/"results"
    results.mkdir(parents=True)
    for name in TABLES:
        shutil.copyfile(raw/name, results/name)
    study.save(results/"independent_rescore.json", independent)
    changes(data, study.MODES).to_csv(results/"prediction_change_summary.csv", index=False)
    audits = json.loads((raw/"fit_audits.json").read_text())
    summarize_access(audits).to_csv(results/"fit_access_summary.csv", index=False)
    indices = [study.MODES.index(m) for m in KEEP]
    for key in ["scores", "predictions"]:
        data[key] = data[key][indices]
    np.savez_compressed(results/"useful_comparison.npz", modes=np.array(KEEP), **data)
    manifest = dict(
        policy="No candidate passed both-pool advancement. Retain full blend/anchored/LDA predictions as a useful matched comparison: the fixed blend won all eight Pooled235 repeats without a sample-F1 supplementary regression. Keep compact results and fit-access counts for both unsuccessful neighbor arms.",
        registration_commit="924ae04",
        source_packets_verified=84,
        raw_prediction_bytes=sum(p.stat().st_size for p in (raw/"predictions").glob("*.npz")),
        raw_fit_audit_bytes=(raw/"fit_audits.json").stat().st_size,
        omitted="Unsuccessful neighbor predictions and detailed fit-round audits; no fitted models were saved. Raw input is disposable after retained verification passes.",
        files={str(p.relative_to(destination)): dict(sha256=study.sha(p), bytes=p.stat().st_size)
               for p in sorted(results.iterdir())})
    study.save(destination/"retention.json", manifest)
    print(json.dumps(independent, indent=2))
    verify(destination)


def verify(destination):
    manifest = json.loads((destination/"retention.json").read_text())
    protocol = json.loads((STUDY/"protocol.json").read_text())
    assert study.sources() == protocol["sources"]
    for relative, record in manifest["files"].items():
        path = destination/relative
        assert path.stat().st_size == record["bytes"] and study.sha(path) == record["sha256"], relative
    results = destination/"results"
    check = json.loads((results/"verification.json").read_text())
    assert check["passed"] and check["protocol_sha256"] == study.sha(STUDY/"protocol.json")
    assert (check["new_cv_fits"], check["new_supplementary_fits"], check["implementation_control_fits"]) == (240, 12, 1)
    with np.load(results/"useful_comparison.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    assert list(data["modes"]) == KEEP
    np.testing.assert_allclose(data["scores"][0], study.centered(data["parent_raw_scores"]), atol=0, rtol=0)
    np.testing.assert_allclose(data["scores"][1], .75*data["scores"][0]+.25*data["scores"][2], atol=0, rtol=0)
    cases = list(study.benchmark.cases())
    assert len(cases) == len(np.unique(data["case"])) == 84
    access = pd.read_csv(results/"fit_access_summary.csv")
    assert len(access) == 252
    for c in cases:
        mask = data["case"] == c["key"]
        np.testing.assert_array_equal(data["ids"][mask], c["query"].ID)
        np.testing.assert_array_equal(data["truth"][mask], c["query"][LABELS])
        assert (data["pool"][mask] == c["pool"]).all() and (data["seed"][mask] == (c["seed"] or 0)).all()
        np.testing.assert_allclose(data["scores"][2, mask], study.lda_for(c), atol=0, rtol=0)
        audit = access[access.key == c["key"]]
        assert set(audit["mode"]) == set(study.MODES[:3]) and len(audit) == 3
        gold = int(c["fit"][LABELS].ge(0).all(axis=1).sum())
        assert (audit.input_rows == len(c["fit"])).all() and (audit.gold_rows == gold).all()
        assert (audit.target_rows == len(c["query"])).all()
        assert (audit.masked_rows == len(c["fit"])-gold).all()
        assert (audit.purged_masked == c["purged_masked"]).all()
        neighbor = audit.set_index("mode").loc["grouped_neighbors"]
        assert neighbor.voting_bank_rows == neighbor.distinct_embedding_groups
        neighbor = audit.set_index("mode").loc["pseudo_neighbor_quarter"]
        assert neighbor.generated_vote_weight == .25 and neighbor.voting_bank_rows == neighbor.original_bank_rows
        if c["pool"] != "supplementary":
            with np.load(study.REFERENCE/f"anchored_{c['pool']}.npz", allow_pickle=True) as z:
                np.testing.assert_allclose(data["parent_raw_scores"][mask], z["scores"][c["repeat_index"], c["positions"]], atol=0, rtol=0)
    independent = rescore(data, KEEP, results)
    assert independent["records_checked"] == 63
    old = json.loads((results/"independent_rescore.json").read_text())
    assert old["passed"] and old["records_checked"] == 105 and old["metric_values_checked"] == 2310
    repeats, summary, pairs, deltas, supplementary = [pd.read_csv(results/(n+".csv")) for n in
        ["repeat_scores", "cv_summary", "paired_summary", "paired_repeat_deltas", "supplementary_scores"]]
    assert (len(repeats), len(summary), len(pairs), len(deltas), len(supplementary)) == (80, 10, 6, 48, 25)
    for frame in [repeats, supplementary]:
        np.testing.assert_allclose(frame[["f1_"+label for label in LABELS]].mean(axis=1), frame.macro_f1, atol=1e-12, rtol=0)
    for row in summary.itertuples():
        group = repeats[(repeats.pool == row.pool) & (repeats["mode"] == row.mode)]
        assert len(group) == 8 and group.seed.is_unique
        np.testing.assert_allclose([group.sample_f1.mean(), group.sample_f1.std(ddof=0), group.macro_f1.mean()],
                                   [row.mean_sample_f1, row.population_sd, row.mean_macro_f1], atol=1e-12, rtol=0)
    qualified = {m: True for m in study.CANDIDATES}
    for row in pairs.itertuples():
        pivot = repeats[repeats.pool == row.pool].pivot(index="seed", columns="mode", values="sample_f1")
        d = pivot[row.mode]-pivot.anchored
        saved = deltas[(deltas.pool == row.pool) & (deltas["mode"] == row.mode)].set_index("seed").loc[d.index]
        np.testing.assert_allclose(d, saved.sample_f1_delta, atol=1e-12, rtol=0)
        wins, ties, losses = int((d > 1e-12).sum()), int((abs(d) <= 1e-12).sum()), int((d < -1e-12).sum())
        assert (wins, ties, losses) == (row.wins, row.ties, row.losses)
        np.testing.assert_allclose(d.mean(), row.mean_delta, atol=1e-12, rtol=0)
        passed = bool(d.mean() > 0 and wins >= 5)
        assert passed == row.meets_pool_rule
        qualified[row.mode] &= passed
    assert qualified == check["qualified_cv"]
    windows = [c for c in cases if c["pool"] == "supplementary" and c["key"] != "validation"]
    for mode in study.MODES:
        scores = supplementary[supplementary["mode"] == mode].set_index("case").sample_f1
        pooled = sum(scores[c["key"]]*len(c["query"]) for c in windows)/98
        np.testing.assert_allclose(pooled, scores["chronology98"], atol=1e-12, rtol=0)
    observed = changes(data, KEEP).sort_values(["case", "mode"]).reset_index(drop=True)
    saved = pd.read_csv(results/"prediction_change_summary.csv")
    saved = saved[saved["mode"].isin(KEEP)].sort_values(["case", "mode"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(observed, saved, check_dtype=False, atol=1e-12, rtol=0)
    print(json.dumps(dict(passed=True, retained_files_checked=len(manifest["files"]),
                         protected_sources_checked=len(protocol["sources"]),
                         retained_prediction_metric_values_rescored=independent["metric_values_checked"],
                         qualified_cv=qualified, models_fitted=0), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["retain", "verify"])
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--destination", type=Path, default=STUDY)
    args = parser.parse_args()
    if args.command == "retain":
        assert args.raw is not None
        retain(args.raw.resolve(), args.destination.resolve())
    else:
        verify(args.destination.resolve())
