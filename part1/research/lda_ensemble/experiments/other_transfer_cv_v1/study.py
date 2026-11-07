"""Transfer only Other scores to the frozen three-way ensemble.

Recipes are fixed before scoring. Reuses exact verified fold predictions; no
outer labels select the transfer amount or enter source model fitting.
"""
import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(variable, "1")
from pathlib import Path
import sys
import json
import datetime
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
GAP = OUT.parent / "feature_gap_cv_v1"
GOLD = OUT.parent / "gold_correction_cv_v1"
sys.path.insert(0, str(GAP))
import run as runner
import features as f

SOURCE = {
    "gold_targets": (GOLD, "ani__half_drug_targets"),
    "gold_embedding": (GOLD, "ani__half_none"),
    "pseudo_targets": (GAP, "ani__drug_targets"),
    "native_ani": (GOLD, "ani"),
}
BLOCKS = {source + "_" + amount: f"Transfer {amount} of {source} Other score only"
          for source in SOURCE for amount in ["half", "full"]}
BASE_COMPARE = runner.comparisons
BASE_REPORT = runner.report
BASE_PROTOCOL = runner.read(GAP / "protocol.json")


def transfer(original, source, block):
    amount = .5 if block.endswith("_half") else 1.
    result = original.copy()
    result[:, 4] = (1 - amount) * original[:, 4] + amount * source[:, 4]
    np.testing.assert_array_equal(result[:, :4], original[:, :4])
    return result


def sources():
    paths = {runner.PART1 / name for name in BASE_PROTOCOL["source_sha256"]}
    paths |= {Path(__file__), GAP / "protocol.json", GOLD / "protocol.json", GOLD / "study.py",
              GOLD / "verification.json", GOLD / "independent_verification.json"}
    for stage in runner.SPLITS:
        for folder, name in SOURCE.values():
            paths.add(folder / stage / (name + ".npz"))
        paths.add(GOLD / stage / "ensemble.npz")
    return {str(path.relative_to(runner.PART1)): runner.sha(path) for path in sorted(paths)}


def guard():
    protocol = runner.read(OUT / "protocol.json")
    for name, digest in protocol["source_sha256"].items():
        assert runner.sha(runner.PART1 / name) == digest, name
    assert protocol["registry_sha256"] == runner.sha(OUT / "registry.json")
    return protocol


def prepare():
    if (OUT / "protocol.json").exists():
        guard()
        return
    rows = [dict(name="ensemble__" + name, parent="ensemble", block=name, description=description)
            for name, description in BLOCKS.items()]
    runner.save(OUT / "registry.json", rows)
    protocol = dict(BASE_PROTOCOL)
    protocol.update(version="other-transfer-cv-v1", frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sources(), registry_sha256=runner.sha(OUT / "registry.json"),
        hypotheses=BLOCKS, candidates=len(rows), parent="Submitted three-way ensemble; all changes restricted to Other score",
        fixed_recipe="Only replace Other decision score, half or full, with a frozen source: gold-half-corrected Ani with targets; same without metadata; pseudo-trained target addition; or native Ani. Other four score columns, cutoff and nonempty fallback unchanged.",
        metadata_diagnostic="No metadata-only diagnostic. Matched gold-embedding and native-Ani transfers distinguish feature changes from source-model and transfer effects.",
        control_reuse="Source models and predictions were fitted on these exact grouped folds. Read their scores, IDs and truth; verify hashes and alignment. Elementwise score transfer never inspects truth. Only supplementary checks require new source fits.",
        controls="Gold-target transfer vs gold-embedding transfer at same amount; pseudo-target transfer vs native-Ani transfer. All advancement remains against original three-way parent; metadata attribution reported separately.",
        limitations="Follow-up motivated by source-level Other results already observed on reused labels/folds. Confirmation is development consistency, not a newly independent holdout.")
    runner.save(OUT / "protocol.json", protocol)
    print("Frozen eight Other-only transfers to the three-way parent.", flush=True)


def read_source(stage, source, gold):
    directory, name = SOURCE[source]
    with np.load(directory / stage / (name + ".npz")) as saved:
        assert saved["ids"].tolist() == gold.ID.tolist()
        np.testing.assert_array_equal(saved["truth"], gold[f.LABELS].to_numpy())
        return saved["scores"].copy()


def parity():
    guard()
    frame = runner.frame_for("pooled")
    plan, pos, groups, masked = runner.plan("pooled3", frame)
    refs = runner.controls("pooled3", frame.iloc[pos])
    for source in SOURCE:
        scores = read_source("pooled3", source, frame.iloc[pos])
        for amount in ["half", "full"]:
            for original, replacement in zip(refs["ensemble"], scores):
                changed = transfer(original, replacement, source + "_" + amount)
                assert np.isfinite(changed).all()
                np.testing.assert_array_equal(changed[:, :4], original[:, :4])
    runner.save(OUT / "parity.json", dict(passed=True, source_alignment=True,
        non_other_score_columns_unchanged=True, label_free_elementwise_transfer=True,
        source_models_previously_verified=True))
    print("Source alignment and Other-only score boundaries verified.", flush=True)


def run_stage(stage, blocks):
    guard()
    # Retain the matching controls in every stage for attribution.
    controls = ["gold_embedding_half", "gold_embedding_full", "native_ani_half", "native_ani_full"]
    blocks = list(dict.fromkeys([*blocks, *controls]))
    directory = OUT / stage
    directory.mkdir(exist_ok=True)
    frame = runner.frame_for("pooled" if stage.startswith("pooled") else "train")
    plan, pos, groups, masked = runner.plan(stage, frame)
    gold = frame.iloc[pos]
    refs = runner.controls(stage, gold)
    cached = {source: read_source(stage, source, gold) for source in SOURCE}
    scores = dict(refs)
    for block in blocks:
        source = block.rsplit("_", 1)[0]
        scores["ensemble__" + block] = np.array([transfer(a, b, block) for a, b in zip(refs["ensemble"], cached[source])])
    rows = []
    for name, value in scores.items():
        repeats = [runner.metrics(gold[f.LABELS].to_numpy(), v) for v in value]
        numbers = [v["sample_f1"] for v in repeats]
        rows.append(dict(candidate=name, mean=float(np.mean(numbers)), sd=float(np.std(numbers)), repeat_scores=repeats))
        np.savez_compressed(directory / (name + ".npz"), scores=value,
            predictions=np.array([runner.decide(v) for v in value]), ids=gold.ID.to_numpy(dtype=str), truth=gold[f.LABELS].to_numpy(int))
    for ri, seed in enumerate(plan["seeds"]):
        for fold in [v for v in plan["folds"] if v["seed"] == seed]:
            fit, query, held = runner.fold_inputs(frame, pos, groups, masked, fold)
            filename = f"seed{seed}_fold{fold['fold']}"
            np.savez_compressed(directory / (filename + ".npz"), ids=query.ID.to_numpy(dtype=str),
                                **{name: value[ri, held] for name, value in scores.items()})
            runner.save(directory / (filename + ".json"), dict(fit_ids=fit.ID.tolist(), query_ids=query.ID.tolist(),
                blocks=blocks, source_predictions_verified=True, control_max_difference=0.,
                new_model_fits=0, protocol_sha256=runner.sha(OUT / "protocol.json")))
    runner.save(directory / "index.json", rows)
    print(pd.DataFrame([dict(candidate=r["candidate"], mean=r["mean"]) for r in rows]).to_string(index=False), flush=True)


def comparisons(extra=False):
    rows = BASE_COMPARE(extra)
    for row in rows:
        source, amount = row["block"].rsplit("_", 1)
        control_source = "gold_embedding" if source == "gold_targets" else "native_ani" if source == "pseudo_targets" else source
        name = "ensemble__" + control_source + "_" + amount
        row["matched_transfer_control"] = name
        for kind in ["train", "pooled"]:
            values = []
            for stage in ([kind + "3", kind + "5"] if extra else [kind + "3"]):
                record = next(v for v in runner.read(OUT / stage / "index.json") if v["candidate"] == name)
                values.extend(v["sample_f1"] for v in record["repeat_scores"])
            delta = np.array(row[kind + "_repeats"]) - values
            row[kind + "_vs_matched_transfer_delta"] = float(delta.mean())
            row[kind + "_vs_matched_transfer_wins"] = int((delta > 1e-12).sum())
    return rows


def gold_source(packet, fit, query, block):
    zq = packet["scaler"].transform(query[f.EMB].to_numpy())
    x, q = packet["z"], zq
    if block == "drug_targets":
        transformer = f.Block(block).fit(fit)
        a, b = transformer.transform(fit), transformer.transform(query)
        weight = .3 * np.sqrt(np.mean(np.sum(x*x, axis=1))) / max(np.sqrt(np.mean(np.sum(a*a, axis=1))), 1e-12)
        x, q = np.column_stack([x, weight*a]), np.column_stack([q, weight*b])
    model = f.native.fit_ani_base(x[packet["gold"]], packet["y"])[0]
    new, old = np.zeros((len(query), 5)), np.zeros((len(query), 5))
    new[:, model.classes_] = model.predict_proba(q)
    native = packet["base"][0]
    old[:, native.classes_] = native.predict_proba(zq)
    raw = f.native.ani_base_predict(packet["base"], zq)
    return f.centered(np.clip(raw + .35*(new-old), 0, 1))


def supplementary():
    guard()
    names = runner.read(OUT / "confirmation.json")["supplementary_names"]
    if not names:
        runner.save(OUT / "supplementary/results.json", [])
        return
    source_names = {name.split("__", 1)[1].rsplit("_", 1)[0] for name in names}
    train = runner.frame_for("train")
    full = runner.frame_for("pooled").set_index("ID", drop=False)
    val = pd.read_csv(runner.PART1 / "data/part1_val.csv")
    windows = [dict(name="original_validation", training_ids=train.ID.tolist(), heldout_ids=val.ID.tolist())]
    windows += runner.read(runner.ROOT / "experiments/cross_repo_benchmark_v1/protocol.json")["chronological_splits"]
    historical = runner.read(runner.HISTORY / "supplementary/results.json")
    (OUT / "supplementary").mkdir(exist_ok=True)
    results = []
    for wi, window in enumerate(windows):
        fit = full.loc[window["training_ids"]].reset_index(drop=True)
        held = full.loc[window["heldout_ids"]].reset_index(drop=True)
        query = held.drop(columns=f.LABELS)
        assert not set(fit.ID) & set(query.ID)
        if wi:
            assert not set(pd.util.hash_pandas_object(fit[f.EMB].round(10), index=False)) & set(pd.util.hash_pandas_object(query[f.EMB].round(10), index=False))
        refs = {}
        for key, title in runner.CONTROL_NAMES.items():
            record = next(r for r in historical if r["candidate"] == title and r["window"] == wi)
            original = pd.read_csv(runner.ROOT / record["prediction_file"]).set_index("ID")
            refs[key] = original.loc[query.ID, [f"score_{label}" for label in f.LABELS]].to_numpy()
        packet = f.train_native(fit)
        np.testing.assert_allclose(f.native_prediction(packet, query), refs["ani"], rtol=0, atol=1e-10)
        generated = {"native_ani": refs["ani"]}
        if "gold_targets" in source_names:
            generated["gold_targets"] = gold_source(packet, fit, query, "drug_targets")
            with np.load(GOLD / "supplementary" / f"window{wi}_ani__half_drug_targets.npz") as saved:
                np.testing.assert_allclose(generated["gold_targets"], saved["scores"], rtol=0, atol=1e-10)
        if "gold_embedding" in source_names:
            generated["gold_embedding"] = gold_source(packet, fit, query, "none")
        if "pseudo_targets" in source_names:
            generated["pseudo_targets"] = f.fit_addition(packet, fit, query, "drug_targets")[0]
            with np.load(GAP / "supplementary" / f"window{wi}_ani__drug_targets.npz") as saved:
                np.testing.assert_allclose(generated["pseudo_targets"], saved["scores"], rtol=0, atol=1e-10)
        for name in [*runner.CONTROL_NAMES, *names]:
            if name in refs:
                score = refs[name]
            else:
                block = name.split("__", 1)[1]
                score = transfer(refs["ensemble"], generated[block.rsplit("_", 1)[0]], block)
            truth = held[f.LABELS].to_numpy(int)
            np.savez_compressed(OUT / "supplementary" / f"window{wi}_{name}.npz", scores=score, predictions=runner.decide(score), truth=truth, ids=query.ID.to_numpy(dtype=str))
            results.append(dict(candidate=name, window=wi, window_name=window["name"], **runner.metrics(truth, score)))
    runner.save(OUT / "supplementary/results.json", results)


def report():
    BASE_REPORT()
    path = OUT / "report.html"
    body = path.read_text().replace("Part 1 feature gap experiments", "Other-only corrective score transfer")
    body = body.replace("Part 1: new feature structure and earlier-feature retries", "Preserving the ensemble while correcting Other")
    start = body.index("<p>Nine blocks")
    stop = body.index("</p>", start) + 4
    body = body[:start] + '<p>Eight frozen Other-only transfers, using gold-corrected or pseudo-trained sources and matched embedding-only controls. Scores for Safety, Funding, Efficacy and Recruitment stay unchanged. The final nonempty-label fallback remains active. Sources use only Part 1 and the same saved grouped splits.</p>' + body[stop:]
    body = body.replace("<h2>Metadata-only diagnostics</h2>", "<h2>No new metadata-only fits</h2>")
    path.write_text(body)
    (OUT / "metadata_only.csv").unlink()


def independent_verify():
    import verify_saved
    verify_saved.HERE = OUT
    verify_saved.main()


runner.OUT = OUT
runner.prepare, runner.guard, runner.parity = prepare, guard, parity
runner.run_stage, runner.comparisons = run_stage, comparisons
runner.supplementary, runner.report = supplementary, report
# The inherited command dispatcher only reads this registry during screening.
runner.m = type("Adapter", (), dict(BLOCKS=BLOCKS, LABELS=f.LABELS, EMB=f.EMB))

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "independent_verify":
        independent_verify()
    else:
        runner.main()
