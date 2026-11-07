"""Gold-label corrective classifiers, with unchanged semi-supervised neighbors.

Reuses the prior frozen runner through in-memory adapters; writes only here.
Run with the recorded research environment: study.py prepare|parity|screen|...
"""
import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(variable, "1")
from pathlib import Path
import sys
import json
import hashlib
import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PRIOR = OUT.parent / "feature_gap_cv_v1"
sys.path.insert(0, str(PRIOR))
import run as runner
import features as features

ORIGINAL_COMPARE = runner.comparisons
ORIGINAL_STAGE = runner.run_stage
ORIGINAL_REPORT = runner.report
PRIOR_PROTOCOL = runner.read(PRIOR / "protocol.json")
BLOCKS = {f"{strength}_{block}": f"{strength} gold-only LR correction with {block}"
          for strength in ["half", "full"]
          for block in ["none", "drug_targets", "diseases", "criteria_text", "icd_full_category"]}


def fit_addition(packet, fit, query, name):
    strength, block_name = name.split("_", 1)
    blend = .5 if strength == "half" else 1.
    zq = packet["scaler"].transform(query[features.EMB].to_numpy())
    multiplier = 0.
    names = []
    if block_name == "none":
        x, q = packet["z"], zq
    else:
        block = features.Block(block_name).fit(fit)
        a, b = block.transform(fit), block.transform(query)
        rms = np.sqrt(np.mean(np.sum(a * a, axis=1)))
        multiplier = .3 * np.sqrt(np.mean(np.sum(packet["z"] ** 2, axis=1))) / max(rms, 1e-12)
        x, q = np.column_stack([packet["z"], a * multiplier]), np.column_stack([zq, b * multiplier])
        names = block.feature_names
    model = features.native.fit_ani_base(x[packet["gold"]], packet["y"])[0]
    gold_lr = np.zeros((len(query), 5))
    gold_lr[:, model.classes_] = model.predict_proba(q)
    native_model = packet["base"][0]
    old_lr = np.zeros_like(gold_lr)
    old_lr[:, native_model.classes_] = native_model.predict_proba(zq)
    raw = features.native.ani_base_predict(packet["base"], zq)
    result = features.centered(np.clip(raw + .7 * blend * (gold_lr - old_lr), 0, 1))
    audit = dict(block=name, dimensions=x.shape[1], multiplier=float(multiplier),
                 feature_names_sha256=hashlib.sha256(json.dumps(names).encode()).hexdigest(),
                 gold_rows=len(packet["gold"]), pseudo_rows_used_by_corrective_LR=0,
                 pseudo_labels_still_used_by_native_kNN=True, replacement_fraction=blend,
                 iterations=int(model.n_iter_.max()), query_rows=len(query))
    return result, features.centered(gold_lr), audit


def source_hashes():
    result = dict(PRIOR_PROTOCOL["source_sha256"])
    for path in [Path(__file__), PRIOR / "protocol.json", PRIOR / "verification.json", PRIOR / "independent_verification.json"]:
        result[str(path.relative_to(runner.PART1))] = runner.sha(path)
    return result


def guard():
    p = runner.read(OUT / "protocol.json")
    for name, digest in p["source_sha256"].items():
        assert runner.sha(runner.PART1 / name) == digest, name
    assert p["registry_sha256"] == runner.sha(OUT / "registry.json")
    return p


def prepare():
    if (OUT / "protocol.json").exists():
        guard()
        return
    for name, digest in PRIOR_PROTOCOL["source_sha256"].items():
        assert runner.sha(runner.PART1 / name) == digest
    registry = [dict(name=f"{parent}__{block}", parent=parent, block=block,
                     description=description, decoder="threshold .5 plus nonempty fallback")
                for block, description in BLOCKS.items() for parent in runner.CONTROL_NAMES]
    runner.save(OUT / "registry.json", registry)
    p = dict(PRIOR_PROTOCOL)
    p.update(version="gold-correction-cv-v1", frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             source_sha256=source_hashes(), registry_sha256=runner.sha(OUT / "registry.json"),
             hypotheses=BLOCKS, candidates=len(registry),
             parent="Original Ani and three-way parents; matched gold-only embedding heads quantify feature contribution.",
             fixed_recipe="Native embeddings, pseudo-label generation and augmented kNN unchanged. Replace half or all of Ani's final LR contribution with a C=.3 multinomial head trained only on gold rows, preserving native class balancing. Compare embeddings alone with individual metadata blocks at fixed .3 RMS weight.",
             metadata_diagnostic="Saved diagnostic is the gold-only multinomial head, including embeddings, centered at native .3 cutoff. It is not a metadata-only model and cannot select recipes.",
             controls="For each replacement amount, an embedding-only gold head is always run. Advance whole recipes against original parents by existing rules; report paired changes against matching gold-only control too. Claim metadata contribution only with positive means and required repeat wins in both pools against that matched control.",
             error_analysis=PRIOR_PROTOCOL["error_analysis"],
             control_reuse=PRIOR_PROTOCOL["control_reuse"],
             supplementary=PRIOR_PROTOCOL["supplementary"])
    runner.save(OUT / "protocol.json", p)
    print("Frozen 20 recipes: five feature choices x half/full gold correction x two original parents.", flush=True)


def run_stage(stage, blocks):
    # Always retain both gold-only controls, including in confirmation.
    blocks = list(dict.fromkeys([*blocks, "half_none", "full_none"]))
    return ORIGINAL_STAGE(stage, blocks)


def comparisons(extra=False):
    rows = ORIGINAL_COMPARE(extra)
    for row in rows:
        strength, block = row["block"].split("_", 1)
        matched = f"{row['parent']}__{strength}_none"
        for kind in ["train", "pooled"]:
            stages = [kind + "3", kind + "5"] if extra else [kind + "3"]
            control = []
            for stage in stages:
                entry = next(r for r in runner.read(OUT / stage / "index.json") if r["candidate"] == matched)
                control.extend(r["sample_f1"] for r in entry["repeat_scores"])
            deltas = np.array(row[kind + "_repeats"]) - control
            row[kind + "_vs_matched_gold_delta"] = float(deltas.mean())
            row[kind + "_vs_matched_gold_wins"] = int((deltas > 1e-12).sum())
        row["matched_gold_control"] = matched
        row["metadata_contribution_confirmed"] = bool(block != "none" and all(
            row[kind + "_vs_matched_gold_delta"] > 1e-12 and
            row[kind + "_vs_matched_gold_wins"] >= (5 if extra else 2) for kind in ["train", "pooled"]))
    return rows


def parity():
    guard()
    frame = runner.frame_for("pooled")
    p, positions, groups, masked = runner.plan("pooled3", frame)
    fit, query, h = runner.fold_inputs(frame, positions, groups, masked, p["folds"][0])
    packet = features.train_native(fit)
    old = features.native_prediction(packet, query)
    refs = runner.controls("pooled3", frame.iloc[positions])
    np.testing.assert_allclose(old, refs["ani"][0, h], rtol=0, atol=1e-10)
    checks = []
    for name in BLOCKS:
        all_scores, head, audit = fit_addition(packet, fit, query, name)
        single, single_head, _ = fit_addition(packet, fit, query.iloc[:1], name)
        np.testing.assert_allclose(all_scores[:1], single, rtol=0, atol=1e-10)
        np.testing.assert_allclose(head[:1], single_head, rtol=0, atol=1e-10)
        changed = dict(packet, targets=1-packet["targets"])
        unchanged, _, _ = fit_addition(changed, fit, query, name)
        np.testing.assert_array_equal(unchanged, all_scores)
        checks.append(dict(recipe=name, query_batch_invariance=True, corrective_head_ignores_generated_targets=True,
                           verified_fitting_rows=audit["gold_rows"]))
    runner.save(OUT / "parity.json", dict(passed=True, native_parent_reproduced=True, checks=checks))
    print("Native parity, query-batch invariance and gold-head label-source checks passed.", flush=True)


def report():
    ORIGINAL_REPORT()
    path = OUT / "report.html"
    body = path.read_text().replace("Part 1 feature gap experiments", "Part 1 gold-label correction")
    body = body.replace("Part 1: new feature structure and earlier-feature retries", "Gold-label corrective classifiers with matched controls")
    body = body.replace("Nine blocks added separately to Ani’s final classifier, evaluated alone and inside the fixed three-way ensemble. Four new representations; five retries. Pseudo-labeling, neighbors, regularization, feature weight and decoders stayed fixed. Only supplied Part 1 data were used.",
        "Half or full replacement of Ani's final logistic contribution with a gold-only head: embeddings alone, protein targets, diseases, criteria text, or richer ICD codes. Native pseudo-labeling and kNN remain intact. Each recipe is evaluated in Ani and the fixed three-way ensemble. Only Part 1 inputs are used.")
    body = body.replace("Metadata-only diagnostics", "Gold-only head diagnostics (embeddings included)")
    body += '<p>Paired metadata contribution against matching gold-only embedding controls is recorded in screen_comparison.csv and final_comparison.csv. A whole-recipe improvement does not establish that metadata caused it.</p>'
    path.write_text(body)
    (OUT / "metadata_only.csv").rename(OUT / "gold_head_diagnostics.csv")
    summary = runner.read(OUT / "summary.json")
    summary["study"] = "gold-only final-head corrections; includes matched embedding controls"
    runner.save(OUT / "summary.json", summary)


def independent_verify():
    import verify_saved
    verify_saved.HERE = OUT
    verify_saved.main()


runner.OUT = OUT
runner.m = SimpleNamespace(**{name: getattr(features, name) for name in ["PART1", "RESEARCH", "LABELS", "EMB", "train_native", "native_prediction"]},
                           BLOCKS=BLOCKS, fit_addition=fit_addition)
runner.prepare, runner.guard = prepare, guard
runner.parity, runner.run_stage = parity, run_stage
runner.comparisons, runner.report = comparisons, report

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "independent_verify":
        independent_verify()
    else:
        runner.main()
