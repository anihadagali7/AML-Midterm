"""Frozen matched-parent study of improvements stacked onto leading ensembles."""
from pathlib import Path
import argparse,json,warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import run_ani_improvements as prior
import ani_combinations as models

r,b,e,pooled=prior.r,prior.b,prior.e,prior.pooled
OUT=r.ROOT/'experiments/ani_combination_v1'
ANI='Ani frozen control'
PAIR='Ani + tri 25%'
TRIPLE='Ani + tri/SVM 15% each'
CONTROLS=[ANI,PAIR,TRIPLE]


def sources():
    return {n:b.file_hash(r.ROOT/n) for n in ['ani_combinations.py','run_ani_combinations.py']}


def registry():return json.loads((OUT/'registry.json').read_text())


def guard():
    prior.guard()
    p=json.loads((OUT/'protocol.json').read_text())
    assert p['source_sha256']==sources()
    assert p['registry_sha256']==b.file_hash(OUT/'registry.json')
    assert p['protected_sha256']==prior.protected()
    assert p['prior_confirmation_sha256']==b.file_hash(prior.OUT/'confirmation.json')
    return p


def prepare():
    if (OUT/'protocol.json').exists():guard();print('Frozen combination protocol verified.');return
    prior.guard();rs=[]
    weights={ANI:dict(ani=1.),PAIR:dict(ani=.75,tri=.25),TRIPLE:dict(ani=.7,tri=.15,svm=.15)}
    for name,w in weights.items():
        rs.append(dict(name=name,parent=name,kind='control',config=dict(model='ani_combination',weights=w,source_sha256=sources()),rule='t0.5'))
    additions=[('pseudo weight 0.5',dict(variant=dict(pseudo_weight=.5))),
               ('three-seed average',dict(seeds=[0,1,2])),
               ('inner Other threshold',dict(other=dict(specialist_weight=0.))),
               ('Other specialist 50%',dict(other=dict(specialist_weight=.5)))]
    for parent,short in [(PAIR,'75/25'),(TRIPLE,'70/15/15')]:
        for name,change in additions:
            rs.append(dict(name=f'{short} + {name}',parent=parent,kind=name,
                config=dict(model='ani_combination',weights=weights[parent],**change,source_sha256=sources()),rule='t0.5'))
    r.save(OUT/'registry.json',rs)
    p=dict(version='ani-combinations-v1',created_at=b.now(),source_sha256=sources(),
        registry_sha256=b.file_hash(OUT/'registry.json'),protected_sha256=prior.protected(),
        prior_confirmation_sha256=b.file_hash(prior.OUT/'confirmation.json'),
        cohort='Eight single-change combinations plus the two unmodified ensemble parents and native Ani control.',
        controls='Compare every addition with its own parent. Keep the prior training-CV leader (75/25) and pooled-CV leader (70/15/15) as fixed frontier references.',
        initial='Same grouped train157 and pooled235 five-fold/three-repeat protocols. No new splits.',
        additions='Half pseudo-label trust in Ani only; or average raw Ani GMM seeds [0,1,2] before centering; or inner-CV Other threshold on the complete ensemble; or 50% binary Other specialist followed by inner-CV threshold. One addition at a time. Ensemble weights unchanged.',
        inner='Refit ALL ensemble members within each of three grouped inner folds (seed270117), excluding inner/outer held-out groups from masked pools. Same threshold grid [.35,.40,.45,.50,.55,.60] and .005*logit(threshold)^2 penalty as previous round. No outer labels influence selection.',
        shortlist='Require positive paired means and >=2/3 wins in BOTH pools versus matched parent. Keep at most four, ranked by minimum mean gain versus fixed frontier references. If none qualify, keep at most two explicitly diagnostic candidates whose matched-parent minimum delta>=-.003 and whose mean improves in one pool.',
        confirmation='All selected additions and all three controls use the same five previously frozen extra repeats. Report eight-repeat means, SD, paired wins versus parent and frontier.',
        supplementary='Only candidates with positive eight-repeat mean gains and >=5/8 wins in BOTH pools versus parent advance; at most three ranked by minimum gain versus frontier. Controls always retained. Freeze this list before original validation and unchanged chronology. No post-validation tuning.',
        control_supplementary='Reuse prior validation and chronology prediction files for the unchanged controls, with recorded file hashes and provenance. Parity is verified independently in CV.',
        limitations='Previous labels informed development and choice of combinations. Reused CV is not independent test data. More candidates create selection optimism. Scores do not establish Kaggle improvement.',
        submissions_created=False,full235_submission_refit=False,test_inputs_for_fit=False)
    r.save(OUT/'protocol.json',p)
    print('Frozen eight single-change combinations and three controls.',flush=True)


def parity():
    guard();checks=[]
    with pooled.pooled_context():
        f=pooled.protocol()['folds'][0];a=np.array(f['train_gold_positions']);h=np.array(f['heldout_gold_positions'])
        q=e.DATA.iloc[e.LAB[h]].drop(columns=e.LABELS);ex=e.GROUP[e.LAB[h]]
        for name in CONTROLS:
            original=next(c for c in prior.registry() if c['name']==name)
            current=next(c for c in registry() if c['name']==name)
            models.clear();reference=models.base.predict(a,q,original['config'],ex)
            models.clear();score=models.predict(a,q,current['config'],ex)
            np.testing.assert_allclose(score,reference,rtol=0,atol=1e-12)
            checks.append(dict(candidate=name,max_difference=float(np.max(np.abs(score-reference)))))
    r.save(OUT/'parity.json',dict(passed=True,checks=checks));print('All three unchanged controls reproduce their prior scores.',flush=True)


def run(kind,extra=False):
    guard();assert json.loads((OUT/'parity.json').read_text())['passed']
    names=json.loads((OUT/'shortlist.json').read_text())['names'] if extra else [c['name'] for c in registry()]
    folder=OUT/(kind+('5' if extra else '3'));folder.mkdir(exist_ok=True)
    def work():
        p=prior.stage_protocol(kind,extra);previous=b.OUT;rows=[]
        try:
            b.OUT=folder
            for c in registry():
                if c['name'] not in names:continue
                models.clear();result=b.evaluate(c,p)
                rows.append(dict(candidate=c['name'],parent=c['parent'],kind=c['kind'],result_key=result['key'],mean=result['cv_mean'],sd=result['repeat_std'],repeat_scores=result['repeat_scores']))
                r.save(folder/'index.json',rows)
                print(json.dumps(dict(stage=folder.name,candidate=c['name'],mean=result['cv_mean'])),flush=True)
        finally:b.OUT=previous;models.clear()
    if kind=='pooled':
        with pooled.pooled_context():work()
    else:work()
    guard()


def summarize(extra=False):
    tables={k:{v['candidate']:v for v in json.loads((OUT/(k+'3')/'index.json').read_text())} for k in ['train','pooled']}
    names=json.loads((OUT/'shortlist.json').read_text())['names'] if extra else [c['name'] for c in registry()]
    if extra:
        for kind in tables:
            more={v['candidate']:v for v in json.loads((OUT/(kind+'5')/'index.json').read_text())}
            assert set(more)==set(names)
            for name in names:tables[kind][name]['repeat_scores']+=more[name]['repeat_scores']
    rows=[]
    for c in registry():
        name=c['name'];parent=c['parent']
        if name not in names:continue
        row=dict(candidate=name,parent=parent,kind=c['kind'])
        for kind in ['train','pooled']:
            get=lambda n:np.array([v['sample_f1'] for v in tables[kind][n]['repeat_scores']])
            values=get(name);assert len(values)==(8 if extra else 3)
            delta=values-get(parent);frontier=values-get(PAIR if kind=='train' else TRIPLE)
            row.update({kind+'_mean':float(values.mean()),kind+'_sd':float(values.std()),kind+'_parent_delta':float(delta.mean()),kind+'_parent_wins':int((delta>1e-12).sum()),kind+'_frontier_delta':float(frontier.mean()),kind+'_frontier_wins':int((frontier>1e-12).sum())})
        row['minimum_parent_gain']=min(row['train_parent_delta'],row['pooled_parent_delta'])
        row['minimum_frontier_gain']=min(row['train_frontier_delta'],row['pooled_frontier_delta'])
        wins=5 if extra else 2
        row['qualifies']=bool(c['kind']!='control' and row['minimum_parent_gain']>1e-12 and row['train_parent_wins']>=wins and row['pooled_parent_wins']>=wins)
        rows.append(row)
    return pd.DataFrame(rows).sort_values('minimum_frontier_gain',ascending=False),tables


def shortlist():
    guard();table,_=summarize();passed=table[table.qualifies].head(4);diagnostic=False
    if passed.empty:
        passed=table[(table.kind!='control')&(table.minimum_parent_gain>=-.003)&((table.train_parent_delta>1e-12)|(table.pooled_parent_delta>1e-12))].head(2);diagnostic=True
    record=dict(created_at=b.now(),names=CONTROLS+passed.candidate.tolist(),diagnostic_only=diagnostic,rule='Frozen in protocol.json',initial_results=table.to_dict('records'))
    path=OUT/'shortlist.json'
    if path.exists():assert json.loads(path.read_text())['names']==record['names']
    else:r.save(path,record)
    table.to_csv(OUT/'screen_comparison.csv',index=False)
    print(table[['candidate','train_mean','pooled_mean','train_parent_delta','pooled_parent_delta','train_parent_wins','pooled_parent_wins','qualifies']].to_string(index=False),flush=True)
    print('Frozen shortlist:',record['names'],flush=True)


def confirm():
    guard();table,tables=summarize(True)
    chosen=table[table.qualifies].head(3).candidate.tolist()
    record=dict(created_at=b.now(),comparison=table.to_dict('records'),repeat_results=tables,supplementary_names=CONTROLS+chosen)
    path=OUT/'confirmation.json'
    if path.exists():assert json.loads(path.read_text())['supplementary_names']==record['supplementary_names']
    else:r.save(path,record)
    table.to_csv(OUT/'confirmation.csv',index=False)
    print(table[['candidate','train_mean','pooled_mean','train_parent_delta','pooled_parent_delta','train_parent_wins','pooled_parent_wins','minimum_frontier_gain','qualifies']].to_string(index=False),flush=True)
    print('Frozen supplementary list:',record['supplementary_names'],flush=True)


def supplementary():
    guard();names=json.loads((OUT/'confirmation.json').read_text())['supplementary_names']
    train,val,full=r.frames();rows=[]
    old_results=json.loads((prior.OUT/'supplementary/results.json').read_text())
    windows=[dict(name='original_validation',training_ids=train.ID.tolist(),heldout_ids=val.ID.tolist())]+json.loads((r.OUT/'protocol.json').read_text())['chronological_splits']
    for wi,window in enumerate(windows):
        fit=full.set_index('ID').loc[window['training_ids']].reset_index();query=full.set_index('ID').loc[window['heldout_ids']].reset_index()
        with r.data_context(fit):
            ex=None if wi==0 else pd.util.hash_pandas_object(query[e.EMB].round(10),index=False).to_numpy()
            for i,c in enumerate(registry()):
                if c['name'] not in names:continue
                if c['name'] in CONTROLS:
                    old=next(x for x in old_results if x['candidate']==c['name'] and x['window']==wi)
                    row=dict(old,parent=c['parent'],reused_from=str((prior.OUT/'supplementary/results.json').relative_to(r.ROOT)),prediction_sha256=b.file_hash(r.ROOT/old['prediction_file']))
                    rows.append(row);r.save(OUT/'supplementary/results.json',rows)
                    print(json.dumps(dict(stage='supplementary_reused',candidate=c['name'],window=wi,sample_f1=row['sample_f1'])),flush=True)
                    continue
                models.clear();p=e.predict(np.arange(len(e.Y)),query.drop(columns=e.LABELS),c['config'],ex)
                y=query[e.LABELS].to_numpy(int);path=OUT/'supplementary'/f'window{wi}_model{i}.csv'
                r.persist_predictions(path,query,y,p,'t0.5')
                row=dict(candidate=c['name'],parent=c['parent'],window=wi,window_name=window['name'],prediction_file=str(path.relative_to(r.ROOT)),**r.detailed(y,p,'t0.5'))
                rows.append(row);r.save(OUT/'supplementary/results.json',rows)
                print(json.dumps(dict(stage='supplementary',candidate=c['name'],window=wi,sample_f1=row['sample_f1'])),flush=True)
        models.clear()
    guard()


def report():
    guard();conf=json.loads((OUT/'confirmation.json').read_text());supp=json.loads((OUT/'supplementary/results.json').read_text());rows=[];temporal=[]
    for name in conf['supplementary_names']:
        row=next(x.copy() for x in conf['comparison'] if x['candidate']==name)
        row['validation']=next(x['sample_f1'] for x in supp if x['candidate']==name and x['window']==0)
        d=pd.concat([pd.read_csv(r.ROOT/x['prediction_file']) for x in supp if x['candidate']==name and x['window']>0],ignore_index=True)
        assert d.ID.is_unique and len(d)==98
        y=d[[f'actual_{l}' for l in e.LABELS]].to_numpy(int);p=d[[f'score_{l}' for l in e.LABELS]].to_numpy()
        score=r.detailed(y,p,'t0.5');row['chronological']=score['sample_f1'];rows.append(row);temporal.append(dict(candidate=name,**score))
    table=pd.DataFrame(rows).sort_values('minimum_frontier_gain',ascending=False)
    table.to_csv(OUT/'final_comparison.csv',index=False)
    r.save(OUT/'summary.json',dict(created_at=b.now(),comparison=rows,protocol_sha256=b.file_hash(OUT/'protocol.json'),submissions_created=False,final235_submission_refit=False))
    r.save(OUT/'chronological_summary.json',temporal)
    display=['candidate','train_mean','pooled_mean','validation','chronological','train_parent_delta','pooled_parent_delta']
    screen=pd.read_csv(OUT/'screen_comparison.csv')
    body='<!doctype html><meta charset="utf-8"><title>Ensemble combinations</title><style>body{font:16px system-ui;max-width:1600px;margin:36px auto;padding:0 24px;color:#172331}table{border-collapse:collapse;width:100%;margin:24px 0}td,th{padding:9px;border-bottom:1px solid #d9e0e7;text-align:left}th{background:#eef3f8}p{max-width:1100px;line-height:1.55}</style><h1>Combining improvements with the leading ensembles</h1>'
    body+='<p>Eight additions, one at a time, tested against their exact unchanged parents. All settings were frozen before scoring. Every initial model used the same grouped five-fold x three-repeat protocols in train157 and pooled235. Candidates selected by the frozen paired-CV rule received five additional saved repeats. Validation and chronology were consulted only after the supplementary list was frozen.</p>'
    body+='<h2>Eight-repeat results and supplementary checks</h2>'+table[display].to_html(index=False,float_format=lambda x:f'{x:.5f}',border=0)
    body+='<p>Parent deltas compare each addition with the ensemble it modifies. The two frontier references are the previous training-CV leader (75/25) and pooled-CV leader (70/15/15). All CV estimates reuse development labels; they are not independent tests or new Kaggle scores. No Kaggle submission, final235 submission refit, or test-input fitting occurred.</p>'
    body+='<h2>Complete initial screen</h2>'+screen[['candidate','parent','train_mean','pooled_mean','train_parent_delta','pooled_parent_delta','qualifies']].to_html(index=False,float_format=lambda x:f'{x:.5f}',border=0)
    body+='<p><a href="protocol.json">Protocol</a> · <a href="registry.json">Recipes</a> · <a href="confirmation.json">All repeat comparisons</a> · <a href="verification.json">Verification</a> · <a href="boundary_audit.json">Fitting-boundary audit</a></p>'
    (OUT/'report.html').write_text(body)
    print(table[display].to_string(index=False),flush=True)


def verify():
    guard();checks=[]
    assert json.loads((OUT/'boundary_audit.json').read_text())['passed']
    for kind in ['train','pooled']:
        frame=r.frames()[0 if kind=='train' else 2];gold=frame[frame[e.LABELS].ge(0).all(axis=1)]
        for count,n in [('3',3),('5',5)]:
            folder=OUT/(kind+count)
            for row in json.loads((folder/'index.json').read_text()):
                a=np.load(folder/(row['result_key']+'.npz'),allow_pickle=True)
                assert a['ids'].tolist()==gold.ID.tolist();np.testing.assert_array_equal(a['truth'],gold[e.LABELS].to_numpy(int));assert len(a['scores'])==n
                for i,p in enumerate(a['scores']):
                    np.testing.assert_array_equal(e.decisions(p,'t0.5'),a['predictions'][i])
                    assert abs(e.metrics(a['truth'],p,'t0.5')['sample_f1']-row['repeat_scores'][i]['sample_f1'])<1e-12
                if row['candidate'] in CONTROLS:
                    ref=next(x for x in json.loads((prior.OUT/(kind+count)/'index.json').read_text()) if x['candidate']==row['candidate'])
                    old=np.load(prior.OUT/(kind+count)/(ref['result_key']+'.npz'),allow_pickle=True)
                    np.testing.assert_allclose(a['scores'],old['scores'],rtol=0,atol=1e-12)
                    np.testing.assert_array_equal(a['predictions'],old['predictions'])
                checks.append(dict(stage=kind+count,candidate=row['candidate'],repeats_rescored=n))
    for row in json.loads((OUT/'supplementary/results.json').read_text()):
        if 'prediction_sha256' in row:assert b.file_hash(r.ROOT/row['prediction_file'])==row['prediction_sha256']
        d=pd.read_csv(r.ROOT/row['prediction_file']);y=d[[f'actual_{l}' for l in e.LABELS]].to_numpy(int);h=d[[f'predicted_{l}' for l in e.LABELS]].to_numpy(int)
        assert abs(e.metrics(y,h,'t0.5')['sample_f1']-row['sample_f1'])<1e-12
    r.save(OUT/'verification.json',dict(passed=True,checks=checks,unchanged_parents_reproduced=True,source_sha256=sources(),protected_sha256=prior.protected()))
    print('Predictions, truth, IDs and parent parity verified; protected artifacts unchanged.',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','parity','train','pooled','shortlist','extra_train','extra_pooled','confirm','supplementary','report','verify']);a=parser.parse_args()
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        if a.stage in ['train','pooled']:run(a.stage)
        elif a.stage.startswith('extra_'):run(a.stage[6:],True)
        else:globals()[a.stage]()


if __name__=='__main__':main()
