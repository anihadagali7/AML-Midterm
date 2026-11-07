"""Reproduce the original ensemble's full-data fit for sprint exports."""
import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(key, "4")
import json
import joblib
from all_submission_models import LegacySubmissions
from benchmark_protocol import initialize, digest
from experiments import ROOT
from sprint_export import LEGACY


def run():
    train, _ = initialize()
    sources = {name: digest(ROOT/name) for name in [
        "all_submission_models.py", "benchmark_models.py", "experiments.py", "boosted.py"]}
    if LEGACY.exists():
        packet = joblib.load(LEGACY)
        assert packet["source_sha256"] == sources
        assert packet["model"].training_ids == train.ID.tolist()
        action = "Verified existing full-data fit"
    else:
        model = LegacySubmissions().fit(train, progress=lambda msg: print(msg, flush=True))
        packet = dict(model=model, source_sha256=sources)
        LEGACY.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(packet, LEGACY, compress=3)
        action = "Fitted original ensemble on full labeled data"
    print(json.dumps(dict(action=action, training_rows=len(train),
                         path=str(LEGACY.relative_to(ROOT)), sha256=digest(LEGACY)), indent=2))


if __name__ == "__main__":
    run()
