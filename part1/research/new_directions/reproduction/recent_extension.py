"""Declare the fixed one-update candidate's supplementary diagnostic before fitting."""
from pathlib import Path
import json,time,sys
from threadpoolctl import threadpool_limits
import study as e,recent_checks,study_v10
ROOT=Path(__file__).resolve().parent;DEST=ROOT/'recent_followup_plan.json'
def freeze():
    assert not DEST.exists()
    e.save(DEST,dict(frozen_at=time.time(),source_sha256=e.sha(__file__),study='v10',recipe='hybrid_leaf_one_raw',
      evidence_sha256=e.sha(study_v10.OUT/'comparison_8.csv'),
      rationale='Best fixed follow-up improves both CV means, wins 6/8 train and 4/8 pooled, but fails the frozen two-pool gate. Evaluate fixed validation and chronology as a diagnostic; no promotion or threshold change.',
      general_protocol_sha256=e.sha(ROOT/'recent_diagnostic_plan.json'),no_promotion=True))
def run():
    plan=json.loads(DEST.read_text());assert e.sha(__file__)==plan['source_sha256']
    folder=study_v10.OUT/'supplementary'/plan['recipe'];folder.mkdir(parents=True,exist_ok=True)
    record=folder/'additional_diagnostic_plan.json'
    if record.exists():assert json.loads(record.read_text())==plan
    else:e.save(record,plan)
    with threadpool_limits(limits=1):recent_checks.supplementary(plan['study'],plan['recipe'])
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    else:run()
