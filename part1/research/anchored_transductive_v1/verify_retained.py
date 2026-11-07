"""Read-only verification of compact records; no fitting or discarded-cache rescoring."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[2]
RESULTS = STUDY/"results"


def main():
    manifest = json.loads((STUDY/"retention.json").read_text())
    protocol = json.loads((STUDY/"protocol.json").read_text())
    for relative, record in manifest["files"].items():
        data = (STUDY/relative).read_bytes()
        assert len(data) == record["bytes"]
        assert hashlib.sha256(data).hexdigest() == record["sha256"], relative
    for relative, digest in protocol["sources"].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == digest, relative
    check = json.loads((RESULTS/"verification.json").read_text())
    assert check["protocol_sha256"] == hashlib.sha256((STUDY/"protocol.json").read_bytes()).hexdigest()
    assert check["passed"] and check["full_cv_fits"] == 240 and check["supplementary_fits"] == 12
    repeats = pd.read_csv(RESULTS/"repeat_scores.csv")
    summary = pd.read_csv(RESULTS/"cv_summary.csv")
    pairs = pd.read_csv(RESULTS/"paired_summary.csv")
    deltas = pd.read_csv(RESULTS/"paired_repeat_deltas.csv")
    assert len(repeats) == 48 and len(summary) == 6 and len(pairs) == 4 and len(deltas) == 32
    label_f1 = ["f1_"+s for s in ["Safety", "Funding", "Efficacy", "Recruitment", "Other"]]
    np.testing.assert_allclose(repeats[label_f1].mean(axis=1), repeats.macro_f1, atol=1e-12, rtol=0)
    qualified = {m: True for m in ["geometry_only", "target_self_training"]}
    for row in summary.itertuples():
        group = repeats[(repeats.pool == row.pool) & (repeats["mode"] == row.mode)]
        assert len(group) == 8 and group.seed.is_unique
        np.testing.assert_allclose([group.sample_f1.mean(), group.sample_f1.std(ddof=0), group.macro_f1.mean()],
                                   [row.mean_sample_f1, row.population_sd, row.mean_macro_f1], atol=1e-12, rtol=0)
    for row in pairs.itertuples():
        p = repeats[repeats.pool == row.pool].pivot(index="seed", columns="mode", values="sample_f1")
        d = p[row.mode]-p.inductive
        saved = deltas[(deltas.pool == row.pool) & (deltas["mode"] == row.mode)].set_index("seed").loc[d.index]
        np.testing.assert_allclose(saved.sample_f1_delta, d, atol=1e-12, rtol=0)
        wins, ties, losses = (d > 1e-12).sum(), (d.abs() <= 1e-12).sum(), (d < -1e-12).sum()
        assert (wins, ties, losses) == (row.wins, row.ties, row.losses)
        np.testing.assert_allclose(d.mean(), row.mean_delta, atol=1e-12, rtol=0)
        passed = bool(d.mean() > 0 and wins >= 5)
        assert passed == row.meets_pool_rule
        qualified[row.mode] &= passed
    assert qualified == check["qualified_cv"] and not any(qualified.values())
    supplementary = pd.read_csv(RESULTS/"supplementary_scores.csv")
    assert len(supplementary) == 15
    np.testing.assert_allclose(supplementary[label_f1].mean(axis=1), supplementary.macro_f1, atol=1e-12, rtol=0)
    chronology = json.loads((ROOT/"part1/experiments/current_leaders_grouped_v1/snapshots/chronology.json").read_text())
    for mode in qualified.keys() | {"inductive"}:
        scores = supplementary[supplementary["mode"] == mode].set_index("case").sample_f1
        total = sum(w["heldout_rows"] for w in chronology["chronological_splits"])
        assert total == 98
        pooled = sum(scores[w["name"]]*w["heldout_rows"] for w in chronology["chronological_splits"])/total
        np.testing.assert_allclose(pooled, scores["chronology98"], atol=1e-12, rtol=0)
    result = dict(passed=True, retained_files_checked=len(manifest["files"]),
                  protected_sources_checked=len(protocol["sources"]), repeat_records=48,
                  supplementary_records=15, qualified_cv=qualified, models_fitted=0,
                  discarded_trial_predictions_rescored=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
