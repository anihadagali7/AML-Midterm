"""Predeclared follow-up for the focused pairwise-discrimination study."""
import sys,time,json
from pathlib import Path
from threadpoolctl import threadpool_limits
import study as e,study_v14,models_v14,iteration_checks
BASE=Path(__file__).resolve().parent;PLAN=BASE/'next_diagnostics/pairwise_followup_plan.json'
def freeze():
    assert not PLAN.exists()
    e.save(PLAN,dict(frozen_at=time.time(),source_sha256=e.sha(__file__),study='v14',
       supplementary_runner_sha256=e.sha(BASE/'iteration_checks.py'),
       selection='Positive paired means and >=2/3 wins in both pools advances to eight. Positive means and >=5/8 wins in both pools then receives supplementary checks. No exceptions.',
       preceding_checkpoint_sha256=e.sha(BASE/'next_diagnostics/summary.json'),no_test_predictions=True,no_promotion=True))
def run():
    plan=json.loads(PLAN.read_text());assert plan['source_sha256']==e.sha(__file__) and plan['supplementary_runner_sha256']==e.sha(BASE/'iteration_checks.py')
    study_v14.configure();cs=study_v14.registry()['recipes']
    with threadpool_limits(1):
        names=e.summarize(3).query('qualifies').recipe.tolist()
        for c in cs:
            if c['name'] in names:
                for pool in ['train','pooled']:e.evaluate(c,pool,8)
        if names:
            qualified=e.summarize(8).query('qualifies').recipe.tolist();iteration_checks.PLAN=PLAN
            for c in cs:
                if c['name'] in qualified:iteration_checks.supplementary(study_v14,models_v14,c)
        print('Initial pairwise qualifiers:',names,flush=True)
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    else:run()
