"""Verify cached execution, native model replay, protocol and candidate artifacts."""
import argparse,json,subprocess,sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from benchmark_protocol import DIRECTORY,initialize,resolve,digest
from experiments import ROOT


def run(verify_cache=False):
    universe,protocol=initialize();out=ROOT/json.loads((DIRECTORY/'current_run.json').read_text())['directory']
    manifest=json.loads((out/'run_manifest.json').read_text())
    assert all(digest(ROOT/f)==sha for f,sha in manifest['source_sha256'].items())
    assert digest(DIRECTORY/'protocol.json')==manifest['protocol_sha256']
    for track,models in [(out,6),(out/'published_recipe_track',5),(out/'derived_candidates',3),(out/'published_recipe_track/derived_candidates',3)]:
        summary=pd.read_csv(track/'summary.csv');scores=pd.read_csv(track/'fold_scores.csv')
        assert len(summary)==models and summary.complete_comparison.all()
        assert len(scores)==10*models and not scores.duplicated(['model','split']).any()
    official=next(s for s in protocol['splits'] if s['name']=='official')
    _,ev=resolve(universe,official);errors={}
    for name in ['ani_stack','our_linear']:
        model=joblib.load(out/'models/official'/f'{name}.joblib')
        expected=pd.read_csv(out/'predictions/official'/f'{name}.csv')
        assert expected.ID.tolist()==ev.ID.tolist()
        error=float(np.max(np.abs(model.predict_proba(ev)-expected.probability.to_numpy())))
        assert error<1e-12;errors[name]=error
    checks={'splits':10,'strict_recipes':6,'published_recipe_variants':5,'complete_comparisons':True,
            'dataset_and_source_hashes_match':True,'native_probability_replay_max_errors':errors}
    sample=pd.read_csv(ROOT/'part2_sample_submission.csv')
    files=list((ROOT/'submissions').glob('0[1-8]_*.csv'))
    for f in files:
        d=pd.read_csv(f)
        assert d.columns.tolist()==['ID','label'] and len(d)==1500 and d.ID.is_unique and d.ID.equals(sample.ID)
        assert np.isfinite(d.label).all() and d.label.between(0,1,inclusive='neither').all()
    checks['valid_submission_files']=len(files)
    if verify_cache:
        before={str(p):p.stat().st_mtime_ns for p in (out/'models').glob('**/*.joblib')}
        old_summary=pd.read_csv(out/'summary.csv').set_index('model')
        with (DIRECTORY/'cached_replay.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'local_benchmark.py'),'--suite','full'],stdout=log,stderr=subprocess.STDOUT,check=True)
        after={str(p):p.stat().st_mtime_ns for p in (out/'models').glob('**/*.joblib')}
        assert before==after,'Cached replay unexpectedly refitted or rewrote models.'
        new_summary=pd.read_csv(out/'summary.csv').set_index('model')
        for col in ['official_logloss','temporal_mean_logloss','cv_mean_logloss','temporal_delta_lower','group_cv_delta_upper']:
            assert np.allclose(old_summary[col],new_summary[col].reindex(old_summary.index),atol=1e-12,rtol=0,equal_nan=True)
        checks['cached_run_refits_no_models']=True;checks['cached_metrics_and_intervals_repeat']=True
    (DIRECTORY/'verification.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify-cache',action='store_true')
    run(parser.parse_args().verify_cache)
