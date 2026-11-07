"""Prepare two optional, Ani-anchored CSVs after the complete benchmark.

The winning published prediction file remains the anchor. Refit our linear
component on train+validation consistently with Ani's final fitting population.
These are prospective candidates, not established leaderboard improvements.
"""
import hashlib,json
import joblib
import numpy as np
import pandas as pd
from benchmark_models import OurLinear
from benchmark_protocol import DIRECTORY,initialize,digest
from experiments import ROOT,load_data


def run():
    initialize();out=ROOT/json.loads((DIRECTORY/'current_run.json').read_text())['directory']
    for track in [out,out/'published_recipe_track']:
        for suffix in ['', 'derived_candidates/']:
            s=pd.read_csv(track/suffix/'summary.csv')
            assert s.complete_comparison.all(),'Finish all comparisons before generating candidates.'
    data=load_data();train=pd.concat([data['train'],data['val']],ignore_index=True)
    assert not set(train.trial_id)&set(data['test'].trial_id)
    source=ROOT/'experiments/ani_comparison/source'
    provenance=json.loads((source.parent/'provenance.json').read_text())
    for file in ['part2_best_submission.csv','part2_alt_phase3_trend_submission.csv']:
        assert digest(source/file)==provenance['files']['midterm/'+file]
    sample=pd.read_csv(ROOT/'part2_sample_submission.csv')
    def ordered(file):
        d=pd.read_csv(source/file)
        assert d.ID.is_unique and set(d.ID)==set(sample.ID)
        return d.set_index('ID').loc[sample.ID,'label'].to_numpy()
    ani=ordered('part2_best_submission.csv');trend=ordered('part2_alt_phase3_trend_submission.csv')
    models=out/'final_models';models.mkdir(exist_ok=True)
    path=models/'our_linear_trainval.joblib'
    if path.exists():linear=joblib.load(path)
    else:
        linear=OurLinear().fit(train);joblib.dump(linear,path,compress=3)
    assert linear.training_ids==train.ID.tolist()
    test=data['test'].set_index('ID',drop=False).loc[sample.ID].reset_index(drop=True)
    lp=linear.predict_proba(test)
    candidates={'07_ani_mild_phase3_trend':.75*ani+.25*trend,'08_ani_linear25':.75*ani+.25*lp}
    recipes={'07_ani_mild_phase3_trend':{'Ani 0.58688 published probabilities':.75,'Ani capped Phase III trend alternative':.25},
             '08_ani_linear25':{'Ani 0.58688 published probabilities':.75,'Our linear component refitted on train+validation':.25}}
    history={p.name:digest(p) for p in (ROOT/'submissions').glob('0[1-6]_*.csv')}
    rows=[]
    for name,p in candidates.items():
        assert np.isfinite(p).all() and ((p>0)&(p<1)).all()
        f=ROOT/'submissions'/f'{name}.csv'
        pd.DataFrame({'ID':sample.ID,'label':p}).to_csv(f,index=False,float_format='%.10f')
        disk=pd.read_csv(f)
        assert len(disk)==1500 and disk.ID.equals(sample.ID) and disk.ID.is_unique
        assert np.max(abs(disk.label-p))<5.1e-11
        rows.append({'submission':f.name,'mean_test_probability':float(p.mean()),'mean_abs_change_from_ani':float(np.abs(p-ani).mean()),
            'sha256':digest(f),'kaggle_score':None})
    for name,sha in history.items():assert digest(ROOT/'submissions'/name)==sha
    pd.DataFrame(rows).to_csv(ROOT/'submissions/ani_candidates_index.csv',index=False)
    (ROOT/'submissions/ani_candidates_manifest.json').write_text(json.dumps({
        'baseline_score':.58688,'baseline_source_commit':provenance['commit'],
        'reference_file_sha256':digest(source/'part2_best_submission.csv'),
        'trend_file_sha256':digest(source/'part2_alt_phase3_trend_submission.csv'),
        'benchmark_run':str(out.relative_to(ROOT)),'linear_final_training_rows':len(train),
        'recipe':recipes,'files':rows,'status':'Unsubmitted, optional experiments. Small local differences and overlapping paired intervals do not establish a Kaggle improvement.',
        'selection':'Compare both strict and faithful-published-recipe tracks. Prefer consistency across chronological windows and grouped CV over supplied-validation F1.'},indent=2))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':run()
