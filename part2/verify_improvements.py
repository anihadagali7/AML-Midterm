"""Independent artifact checks; no expensive model refits."""
import json
import numpy as np
import pandas as pd
from benchmark_protocol import initialize, resolve, digest
from experiments import ROOT
from improvement_models import RECIPES
from improvement_trends import TRENDS
from improvement_text import TEXT_RECIPES
from improvement_report import current, audit_track, REFERENCES


def verify():
    out = current()
    universe, protocol = initialize()
    manifest = json.loads((out / "manifest.json").read_text())
    for name, expected in manifest["source_sha256"].items():
        assert digest(ROOT / name) == expected, name
    for batch, filename in [("trend_followup", "improvement_trends.py"), ("text_followup", "improvement_text.py")]:
        assert digest(ROOT / filename) == json.loads((out / batch / "manifest.json").read_text())["source_sha256"]
    details = dict(run=out.name, recipe_count=len(RECIPES)+len(TRENDS)+len(TEXT_RECIPES),
                   tracks=2, splits_per_track=10, source_hashes_match=True,
                   original_protocol_unchanged=True, fitted_boundaries_verified=True)
    max_reproduction_error = 0.
    for track in ["strict", "published"]:
        audit_track(out / "text_followup" / track, universe, protocol, names=list(RECIPES)+list(TRENDS)+list(TEXT_RECIPES))
        for split in protocol["splits"]:
            train, ev = resolve(universe, split)
            bank = out / track / "bases" / f"{split['name']}.npz"
            arrays = np.load(bank)
            audit = json.loads(bank.with_name(split["name"]+"_audit.json").read_text())
            assert audit["training_ids"] == train.ID.tolist()
            assert audit["evaluation_ids"] == ev.ID.tolist()
            assert not set(train.trial_id) & set(ev.trial_id)
            for inner in audit["audit"]:
                assert not set(inner["fit_ids"]) & set(inner["heldout_ids"])
                if track == "strict":
                    assert set(inner["preprocessing_fit_ids"]) == set(inner["fit_ids"])
                    assert not set(inner["preprocessing_fit_ids"]) & set(ev.ID)
            extra = json.loads((out / "text_followup" / track / "bases" / f"{split['name']}_audit.json").read_text())
            for inner in extra:
                assert set(inner["outcome_ratio_fit_ids"]) == set(inner["fit_ids"])
                assert not set(inner["outcome_ratio_fit_ids"]) & (set(inner["eval_ids"]) | set(ev.ID))
                if track == "strict":
                    assert set(inner["feature_fit_ids"]) == set(inner["fit_ids"])
            archived = ROOT / "experiments/local_benchmark/runs/68db59271f5f0b9d"
            if track == "published":
                archived /= "published_recipe_track"
            old = pd.read_csv(archived / "predictions" / split["name"] / "ani_stack.csv")
            new = pd.read_csv(out / track / "predictions" / split["name"] / "ani_stack.csv")
            assert old.ID.equals(new.ID)
            error = float(np.abs(old.probability-new.probability).max())
            max_reproduction_error = max(max_reproduction_error, error)
            assert error < 1e-6, (track, split["name"], error)
    details["historical_baseline_max_probability_error"] = max_reproduction_error
    parity = json.loads((out / "final/parity.json").read_text())
    assert all(v["max_probability_error"] < 1e-6 for v in parity.values())
    details["both_kaggle_sources_reproduced"] = parity
    selection = json.loads((out / "selection.json").read_text())
    assert digest(ROOT / "IMPROVEMENT_PROTOCOL.md") == selection["protocol_document_sha256"]
    assert digest(ROOT / "improvement_text.py") == selection["selection_code_sha256"]
    for track, expected in selection["input_sha256"].items():
        assert digest(out / "text_followup" / track / "independent_summary.csv") == expected
    robustness = json.loads((out / "robustness/protocol.json").read_text())
    for track in ["strict", "published"]:
        audit_track(out / "robustness" / track, universe, robustness, names=REFERENCES+selection["shortlist"])
    details["gapped_windows_verified"] = 6
    details["shortlist"] = selection["shortlist"]
    export = json.loads((ROOT / "submissions/stack_improvements_manifest.json").read_text())
    for entry in export["submissions"]:
        assert digest(ROOT / "submissions" / entry["file"]) == entry["sha256"]
        assert entry["kaggle_score"] is None
    details["new_submissions"] = len(export["submissions"])
    details["independently_rescored_predictions"] = 2*10*details["recipe_count"]
    (out / "verification.json").write_text(json.dumps(details, indent=2))
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    verify()
