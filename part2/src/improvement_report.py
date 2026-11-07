"""Audit saved predictions, compare BOTH winners, freeze selection, and export."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark_protocol import initialize, resolve, digest
from experiments import ROOT, metrics, load_data
from improvement_models import RECIPES

OUT = ROOT / "experiments/stack_improvements"
REFERENCES = ["ani_stack", "ani_first"]


def current():
    return OUT / "runs" / json.loads((OUT / "current_run.json").read_text())["run"]


def row_loss(frame):
    p = np.clip(frame.probability.to_numpy(), 1e-6, 1 - 1e-6)
    y = frame.label.to_numpy()
    return -(y * np.log(p) + (1-y) * np.log1p(-p))


def paired_interval(frames, model, reference, kind, repetitions=2000):
    pieces = []
    for name, data in frames.items():
        if data["split"]["kind"] != kind:
            continue
        a, b = data[model], data[reference]
        assert a[["ID", "trial_id", "phase", "label"]].equals(b[["ID", "trial_id", "phase", "label"]])
        d = a[["ID", "trial_id", "phase"]].copy()
        d["period"] = name if kind == "temporal" else "cv"
        d["delta"] = row_loss(a) - row_loss(b)
        pieces.append(d)
    d = pd.concat(pieces, ignore_index=True)
    d = d.groupby(["ID", "trial_id", "phase", "period"], as_index=False).delta.mean()
    trials, indices = np.unique(d.trial_id, return_inverse=True)
    rng = np.random.default_rng(20261106)
    # Chunking avoids allocating a repetitions-by-row dense array.
    samples = []
    for start in range(0, repetitions, 100):
        weights = rng.poisson(1, (min(100, repetitions-start), len(trials)))
        group_scores = []
        for _, group in d.groupby(["period", "phase"]):
            w = weights[:, indices[group.index]]
            denom = w.sum(axis=1)
            group_scores.append(np.divide(w @ group.delta.to_numpy(), denom,
                out=np.full(len(w), np.nan), where=denom > 0))
        samples.extend(np.nanmean(group_scores, axis=0))
    point = d.groupby(["period", "phase"]).delta.mean().mean()
    return dict(model=model, reference=reference, kind=kind, delta=float(point),
                lower=float(np.quantile(samples, .025)), upper=float(np.quantile(samples, .975)),
                trials=len(trials), bootstrap_repetitions=repetitions)


def audit_track(folder, universe, protocol, names=None):
    names = list(RECIPES) if names is None else names
    frames, rows = {}, []
    for split in protocol["splits"]:
        train, ev = resolve(universe, split)
        group = {"split": split}
        for name in names:
            file = folder / "predictions" / split["name"] / f"{name}.csv"
            d = pd.read_csv(file)
            assert d.ID.is_unique and d.ID.tolist() == ev.ID.tolist(), str(file)
            for col in ["trial_id", "phase", "label"]:
                assert d[col].tolist() == ev[col].tolist(), (file, col)
            assert np.isfinite(d.probability).all() and d.probability.between(0, 1, inclusive="neither").all()
            assert not set(d.trial_id) & set(train.trial_id)
            group[name] = d
            loss = row_loss(d)
            row = dict(model=name, split=split["name"], kind=split["kind"], repeat=split.get("repeat"))
            for phase in ["I", "II", "III"]:
                row[phase+"_logloss"] = float(loss[d.phase.eq(phase)].mean())
            row["phase_mean_logloss"] = np.mean([row[p+"_logloss"] for p in ["I", "II", "III"]])
            saved = pd.read_csv(folder / f"scores_{split['name']}.csv").set_index("model").loc[name]
            assert abs(row["phase_mean_logloss"] - saved.phase_mean_logloss) < 1e-12
            rows.append(row)
        frames[split["name"]] = group
    scores = pd.DataFrame(rows)
    summaries = []
    for name, g in scores.groupby("model"):
        temporal = g[g.kind.eq("temporal")]
        official = g[g.kind.eq("official")]
        cv = []
        for repeat in sorted({s.get("repeat") for s in protocol["splits"] if s["kind"] == "group_cv"}):
            d = pd.concat([frames[s["name"]][name] for s in protocol["splits"]
                           if s["kind"] == "group_cv" and s["repeat"] == repeat])
            assert d.ID.is_unique
            cv.append(float(pd.Series(row_loss(d)).groupby(d.phase.to_numpy()).mean().mean()))
        summaries.append(dict(model=name, completed_splits=len(g),
                              official=float(official.phase_mean_logloss.iloc[0]) if len(official) else None,
                              temporal=float(temporal.phase_mean_logloss.mean()),
                              worst=float(temporal.phase_mean_logloss.max()), cv=float(np.mean(cv)) if cv else None))
    return pd.DataFrame(summaries).set_index("model"), scores, frames


def eligible(summary):
    assert (summary.completed_splits == 10).all()
    refs = summary.loc[REFERENCES]
    return [name for name, row in summary.iterrows() if name not in REFERENCES
            and row.temporal < refs.temporal.min()
            and row.cv <= refs.cv.min() + .001
            and row.worst <= refs.worst.min() + .003]


def select():
    out = current()
    universe, protocol = initialize()
    summaries = {}
    for track in ["strict", "published"]:
        summary, scores, _ = audit_track(out / track, universe, protocol)
        summary.to_csv(out / track / "independent_summary.csv")
        scores.to_csv(out / track / "independent_scores.csv", index=False)
        summaries[track] = summary
    candidates = eligible(summaries["strict"])
    shortlist = summaries["strict"].loc[candidates].sort_values("temporal").head(2).index.tolist()
    decision = dict(shortlist=shortlist, selection_track="strict", strict_eligible=candidates,
                    published_eligible=eligible(summaries["published"]),
                    protocol_document_sha256=digest(ROOT / "IMPROVEMENT_PROTOCOL.md"),
                    selection_code_sha256=digest(Path(__file__)),
                    input_sha256={track: digest(out / track / "independent_summary.csv") for track in summaries},
                    status="Frozen before gapped-window checks and final fitting",
                    known_kaggle_best=dict(model="ani_stack", score=.58688))
    path = out / "selection.json"
    if path.exists():
        old = json.loads(path.read_text())
        assert old["shortlist"] == shortlist and old["input_sha256"] == decision["input_sha256"]
    else:
        path.write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))
    for track, summary in summaries.items():
        print(track, summary.sort_values("temporal").to_string())


def comparisons():
    from improvement_trends import TRENDS
    from improvement_text import TEXT_RECIPES
    out = current()
    universe, protocol = initialize()
    all_intervals = []
    combined = (out / "trend_followup/strict/independent_summary.csv").exists()
    names = list(RECIPES) + list(TRENDS) if combined else list(RECIPES)
    root = out / "trend_followup" if combined else out
    if (out / "text_followup/strict/independent_summary.csv").exists():
        names += list(TEXT_RECIPES)
        root = out / "text_followup"
    for track in ["strict", "published"]:
        _, _, frames = audit_track(root / track, universe, protocol, names=names)
        for name in names:
            for ref in REFERENCES:
                if name == ref:
                    continue
                for kind in ["temporal", "group_cv"]:
                    row = paired_interval(frames, name, ref, kind)
                    all_intervals.append(dict(track=track, **row))
    pd.DataFrame(all_intervals).to_csv(out / "paired_comparisons.csv", index=False)
    print("Saved paired intervals against BOTH Ani winners:", len(all_intervals))


def export():
    from improvement_trends import TRENDS
    from improvement_text import TEXT_RECIPES
    out = current()
    selection = json.loads((out / "selection.json").read_text())
    assert (out / "robustness/verification.json").exists(), "Complete the predeclared robustness check first"
    parity = json.loads((out / "final/parity.json").read_text())
    assert set(parity) == set(REFERENCES)
    assert all(v["max_probability_error"] < 1e-6 for v in parity.values())
    sample = pd.read_csv(ROOT / "part2_sample_submission.csv")
    records = []
    for i, name in enumerate(selection["shortlist"], start=9):
        source = out / "final" / f"{name}.csv"
        df = pd.read_csv(source).set_index("ID").loc[sample.ID].reset_index()
        assert list(df.columns) == ["ID", "label"] and len(df) == 1500
        assert df.ID.is_unique and df.ID.equals(sample.ID)
        assert np.isfinite(df.label).all() and df.label.between(0, 1, inclusive="neither").all()
        path = ROOT / "submissions" / f"{i:02d}_{name}.csv"
        if path.exists():
            assert np.allclose(pd.read_csv(path).label, df.label, rtol=0, atol=1e-14)
        else:
            df.to_csv(path, index=False)
        records.append(dict(model=name, file=path.name, sha256=digest(path), rows=len(df), kaggle_score=None,
                            recipe=(RECIPES | TRENDS | TEXT_RECIPES)[name], training_rows=9050,
                            preprocessing="Published track: all supplied covariates, training labels only",
                            benchmark_run=out.name, selection_sha256=digest(out / "selection.json"),
                            robustness_sha256=digest(out / "robustness/verification.json")))
    status = ("Exploratory, unsubmitted; Ani 0.58688 remains the confirmed Kaggle best" if records
              else "No proposal passed the predeclared rule; no new numbered submission exported. Retain Ani 0.58688.")
    manifest = dict(status=status,
                    submissions=records, source_parity=parity)
    (ROOT / "submissions/stack_improvements_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["select", "compare", "export"])
    args = parser.parse_args()
    {"select": select, "compare": comparisons, "export": export}[args.action]()
