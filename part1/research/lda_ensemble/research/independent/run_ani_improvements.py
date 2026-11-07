"""Frozen staged comparison for four targeted Ani improvement directions.

Stages: prepare, parity, train, pooled, shortlist, extra_train, extra_pooled,
confirm, supplementary, report, verify. Original benchmarks remain immutable.
"""
from pathlib import Path
import argparse
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import ani_improvements as models
import confirm_cross_repo_cv as extension

r = models.common
b, e, pooled = r.b, r.e, r.pooled
OUT = r.ROOT / 'experiments/ani_improvement_v1'
CONTROL = 'Ani frozen control'


def own_sources():
    return {name:b.file_hash(r.ROOT/name) for name in ['ani_improvements.py','run_ani_improvements.py']}


def protected():
    files = r.protected()
    for name in ['predictions/part1_tri_training_trainval235.csv',
                 'predictions/tri_training_trainval235/kaggle_result.json']:
        files[name] = b.file_hash(r.ROOT/name)
    return files


def registry():
    return json.loads((OUT/'registry.json').read_text())


def guard():
    r.check_contract()
    p = json.loads((OUT/'protocol.json').read_text())
    assert p['source_sha256'] == own_sources()
    assert p['protected_sha256'] == protected()
    assert p['registry_sha256'] == b.file_hash(OUT/'registry.json')
    return p


def prepare():
    if (OUT/'protocol.json').exists():
        guard(); print('Existing frozen experiment contract verified.'); return
    r.check_contract()
    configs = [(CONTROL, dict(mode='variant',variant={}), 'control')]
    for weight in [.10,.25,.40]:
        configs.append((f'Ani + tri {weight:.0%}',dict(mode='blend',weights=dict(ani=1-weight,tri=weight)), 'ensemble'))
    for weight in [.10,.25]:
        configs.append((f'Ani + SVM {weight:.0%}',dict(mode='blend',weights=dict(ani=1-weight,svm=weight)), 'ensemble'))
    configs.append(('Ani + tri/SVM 15% each',dict(mode='blend',weights=dict(ani=.7,tri=.15,svm=.15)), 'ensemble'))
    for weight in [.5,.25]:
        configs.append((f'Ani pseudo weight {weight}',dict(mode='variant',variant=dict(pseudo_weight=weight)), 'pseudo_trust'))
    for weight in [1.,.5]:
        configs.append((f'Ani agreement, pseudo weight {weight}',dict(mode='variant',variant=dict(pseudo_weight=weight,agreement=True)), 'pseudo_agreement'))
    configs.append(('Ani three-GMM-seed average',dict(mode='seed_average',seeds=[0,1,2]), 'seed_average'))
    for weight in [0.,.25,.5]:
        name = 'Ani inner-CV Other threshold' if not weight else f'Ani Other specialist {weight:.0%}'
        configs.append((name,dict(mode='other',specialist_weight=weight), 'other'))
    rs = [dict(name=name, family=family, config=dict(model='ani_improvement',**config,source_sha256=own_sources()),rule='t0.5',incumbent=name==CONTROL) for name,config,family in configs]
    assert len(rs) == 15
    r.save(OUT/'registry.json', rs)
    p = dict(version='ani-targeted-improvements-v1', created_at=b.now(), source_sha256=own_sources(),
        registry_sha256=b.file_hash(OUT/'registry.json'), protected_sha256=protected(),
        initial_candidates=len(rs), initial_screen='All 15 recipes on the existing train157 and pooled235 grouped five-fold splits, three repeats each.',
        control='Native Ani with unchanged fitting access, C, kNN/GMM, two self-training rounds and threshold; only monotone decision-score centering changes its output representation.',
        ensemble='Center Ani scores from cutoff .3 to .5 using log-odds shift, then fixed convex mixtures. Tri scores stay native. Local SVM hybrid contributes hard votes. These are decision scores, not calibrated probabilities.',
        pseudo_weights='Apply fixed trust to generated rows in multinomial LR, cosine kNN and GMM label mapping. Normalize the LR weighted loss total to its native row-count scale. Preserve exact native arithmetic for weight=1.',
        agreement='Require LR and kNN top-label agreement in addition to native confidence>=.8; always two rounds.',
        other='Three grouped inner folds at seed270117. Binary balanced LR C=.3 on fitting gold only. Fixed specialist mixing weight. Select Other threshold from [.35,.40,.45,.50,.55,.60] by inner sample F1 minus .005*logit(threshold)^2; tie-break toward .5. No outer labels enter selection.',
        seeds='Average GMM seeds [0,1,2]; PCA remains seed0. No best-seed selection.',
        shortlist='Require positive paired mean differences in BOTH three-repeat CV pools and wins in >=2/3 repeats in each. Keep at most four candidates by largest minimum of the two mean differences. If none qualify, retain at most two diagnostic candidates with positive gain in one pool and minimum delta>=-.003; explicitly no improvement claim.',
        confirmation='Use the five already frozen additional repeat assignments for all shortlisted candidates and control; concatenate with original three repeats. No new seeds.',
        supplementary='After confirmation freeze, evaluate at most three strongest candidates by minimum paired eight-repeat mean gain, plus control, on original validation and the same three chronological windows. No post-validation tuning.',
        improvement_evidence='Positive paired means in both eight-repeat pools and >=5/8 repeat wins in each support a CV improvement, not a new Kaggle result. Chronology and original validation remain supplementary.',
        caveat='All are reused development labels. Multiple tested candidates and historical tuning create selection optimism. No independent future-performance guarantee.',
        new_submissions=False, full235_submission_refit=False, test_inputs_used=False)
    r.save(OUT/'protocol.json',p)
    print('Frozen 14 targeted variants plus unchanged Ani control.',flush=True)


def parity():
    guard()
    with pooled.pooled_context():
        p=pooled.protocol();f=p['folds'][0]
        a=np.array(f['train_gold_positions']);h=np.array(f['heldout_gold_positions'])
        q=e.DATA.iloc[e.LAB[h]].drop(columns=e.LABELS);excluded=e.GROUP[e.LAB[h]]
        reference=models.LEGACY(a,q,dict(model='repo_ani'),excluded)
        candidate=models.native_variant(a,q,{},excluded)
        np.testing.assert_allclose(candidate,reference,rtol=0,atol=1e-12)
        np.testing.assert_array_equal(e.decisions(models.centered(candidate),'t0.5'),e.decisions(reference,'t0.3'))
    r.save(OUT/'parity.json',dict(passed=True,max_probability_difference=float(np.max(np.abs(candidate-reference))),native_decisions_identical=True))
    print('Native Ani parity and decision-centering checks passed.',flush=True)


def stage_protocol(kind,extra=False):
    if extra:return extension.protocol(kind)
    return b.protocol() if kind=='train' else pooled.protocol()


def run(kind,extra=False):
    guard();assert json.loads((OUT/'parity.json').read_text())['passed']
    names = json.loads((OUT/'shortlist.json').read_text())['names'] if extra else [r['name'] for r in registry()]
    chosen=[c for c in registry() if c['name'] in names]
    folder=OUT/(kind+('5' if extra else '3'));folder.mkdir(exist_ok=True)
    def work():
        # Read/verify original split artifacts before temporarily redirecting output.
        p=stage_protocol(kind,extra)
        old=b.OUT;rows=[]
        try:
            b.OUT=folder
            for c in chosen:
                models.clear()
                result=b.evaluate(c,p)
                row=dict(candidate=c['name'],family=c['family'],result_key=result['key'],mean=result['cv_mean'],sd=result['repeat_std'],repeat_scores=result['repeat_scores'])
                rows.append(row);r.save(folder/'index.json',rows)
                print(json.dumps(dict(stage=folder.name,candidate=c['name'],mean=result['cv_mean'])),flush=True)
        finally:
            b.OUT=old;models.clear()
    if kind=='pooled':
        with pooled.pooled_context():work()
    else:work()
    guard()


def shortlist():
    guard();tables={kind:{x['candidate']:x for x in json.loads((OUT/(kind+'3')/'index.json').read_text())} for kind in ['train','pooled']}
    assert all(len(t)==15 for t in tables.values())
    rows=[]
    for c in registry():
        name=c['name'];row=dict(candidate=name,family=c['family'])
        for kind,t in tables.items():
            delta=np.array([s['sample_f1'] for s in t[name]['repeat_scores']])-np.array([s['sample_f1'] for s in t[CONTROL]['repeat_scores']])
            row.update({kind+'_mean':t[name]['mean'],kind+'_delta':float(delta.mean()),kind+'_wins':int((delta>1e-12).sum())})
        row['minimum_gain']=min(row['train_delta'],row['pooled_delta'])
        row['qualifies']=bool(row['minimum_gain']>1e-12 and row['train_wins']>=2 and row['pooled_wins']>=2)
        rows.append(row)
    table=pd.DataFrame(rows).sort_values('minimum_gain',ascending=False)
    passed=table[table.qualifies].head(4)
    diagnostic=False
    if passed.empty:
        passed=table[(table.minimum_gain>=-.003)&((table.train_delta>1e-12)|(table.pooled_delta>1e-12))].head(2)
        diagnostic=True
    selection=dict(created_at=b.now(),names=[CONTROL]+passed.candidate.tolist(),diagnostic_only=diagnostic,
        selection_rule='Frozen in protocol.json before initial fits',screen_results=rows)
    path=OUT/'shortlist.json'
    if path.exists():assert json.loads(path.read_text())['names']==selection['names']
    else:r.save(path,selection)
    table.to_csv(OUT/'screen_comparison.csv',index=False)
    print(table.to_string(index=False),flush=True)
    print('FROZEN SHORTLIST',selection['names'],flush=True)


def confirm():
    guard();names=json.loads((OUT/'shortlist.json').read_text())['names'];scores={}
    for kind in ['train','pooled']:
        old={x['candidate']:x for x in json.loads((OUT/(kind+'3')/'index.json').read_text())}
        new={x['candidate']:x for x in json.loads((OUT/(kind+'5')/'index.json').read_text())}
        assert set(new)==set(names)
        scores[kind]={}
        for name in names:
            values=np.array([v['sample_f1'] for v in old[name]['repeat_scores']+new[name]['repeat_scores']])
            assert len(values)==8
            scores[kind][name]=dict(mean=float(values.mean()),sd=float(values.std()),repeats=values.tolist())
    rows=[]
    for name in names:
        row=dict(candidate=name)
        for kind in ['train','pooled']:
            delta=np.array(scores[kind][name]['repeats'])-np.array(scores[kind][CONTROL]['repeats'])
            row.update({kind+'_cv8':scores[kind][name]['mean'],kind+'_sd8':scores[kind][name]['sd'],kind+'_delta':float(delta.mean()),kind+'_wins':int((delta>1e-12).sum())})
        row['minimum_gain']=min(row['train_delta'],row['pooled_delta'])
        row['cv_improvement']=bool(row['minimum_gain']>1e-12 and row['train_wins']>=5 and row['pooled_wins']>=5)
        rows.append(row)
    table=pd.DataFrame(rows).sort_values('minimum_gain',ascending=False)
    selected=table[table.candidate!=CONTROL].head(3).candidate.tolist()
    record=dict(created_at=b.now(),scores=scores,comparison=rows,supplementary_names=[CONTROL]+selected)
    path=OUT/'confirmation.json'
    if path.exists():assert json.loads(path.read_text())['supplementary_names']==record['supplementary_names']
    else:r.save(path,record)
    table.to_csv(OUT/'confirmation.csv',index=False)
    print(table.to_string(index=False),flush=True)


def supplementary():
    guard();names=json.loads((OUT/'confirmation.json').read_text())['supplementary_names']
    chosen=[c for c in registry() if c['name'] in names]
    train,val,full=r.frames();rows=[]
    windows=[dict(name='original_validation',training_ids=train.ID.tolist(),heldout_ids=val.ID.tolist())]
    windows+=json.loads((r.OUT/'protocol.json').read_text())['chronological_splits']
    for wi,window in enumerate(windows):
        fit=full.set_index('ID').loc[window['training_ids']].reset_index()
        query=full.set_index('ID').loc[window['heldout_ids']].reset_index()
        with r.data_context(fit):
            ex=None if wi==0 else pd.util.hash_pandas_object(query[e.EMB].round(10),index=False).to_numpy()
            for c in chosen:
                models.clear()
                probability=e.predict(np.arange(len(e.Y)),query.drop(columns=e.LABELS),c['config'],ex)
                truth=query[e.LABELS].to_numpy(int)
                index=next(i for i,x in enumerate(registry()) if x['name']==c['name'])
                path=OUT/'supplementary'/f'window{wi}_model{index}.csv'
                r.persist_predictions(path,query,truth,probability,c['rule'])
                row=dict(candidate=c['name'],window=wi,window_name=window['name'],prediction_file=str(path.relative_to(r.ROOT)),**r.detailed(truth,probability,c['rule']))
                rows.append(row);r.save(OUT/'supplementary/results.json',rows)
                print(json.dumps(dict(stage='supplementary',window=window['name'],candidate=c['name'],sample_f1=row['sample_f1'])),flush=True)
        models.clear()
    guard()


def report():
    guard();confirmed=json.loads((OUT/'confirmation.json').read_text())
    supp=json.loads((OUT/'supplementary/results.json').read_text());rows=[];temporal=[]
    for name in confirmed['supplementary_names']:
        row=next(x.copy() for x in confirmed['comparison'] if x['candidate']==name)
        row['validation']=next(x['sample_f1'] for x in supp if x['candidate']==name and x['window']==0)
        chunks=[pd.read_csv(r.ROOT/x['prediction_file']) for x in supp if x['candidate']==name and x['window']>0]
        data=pd.concat(chunks,ignore_index=True)
        assert len(data)==98 and data.ID.is_unique
        truth=data[[f'actual_{l}' for l in e.LABELS]].to_numpy(int)
        p=data[[f'score_{l}' for l in e.LABELS]].to_numpy()
        metrics=r.detailed(truth,p,'t0.5');row['chronological']=metrics['sample_f1']
        temporal.append(dict(candidate=name,**metrics));rows.append(row)
    table=pd.DataFrame(rows).sort_values('minimum_gain',ascending=False)
    table.to_csv(OUT/'final_comparison.csv',index=False)
    r.save(OUT/'chronological_summary.json',temporal)
    r.save(OUT/'summary.json',dict(created_at=b.now(),protocol_sha256=b.file_hash(OUT/'protocol.json'),
        comparison=rows,submissions_created=False,full235_submission_refit=False,validation_tuning=False))
    screen=pd.read_csv(OUT/'screen_comparison.csv')
    body='<!doctype html><meta charset="utf-8"><title>Ani improvement experiments</title><style>body{font:16px system-ui;max-width:1500px;margin:36px auto;padding:0 24px;color:#172331}table{border-collapse:collapse;width:100%;margin:24px 0}td,th{padding:9px;border-bottom:1px solid #d9e0e7;text-align:left}th{background:#eef3f8}p{max-width:1100px;line-height:1.55}</style><h1>Targeted Ani improvements</h1>'
    body+='<p>Four directions: fixed score ensembles, pseudo-label trust and agreement, averaging fixed clustering seeds, and an Other specialist with nested threshold selection. Fifteen initial candidates include the unchanged Ani control. Recipes were frozen before scoring. Shared grouped folds exclude held-out embedding groups and matching masked duplicates. Test inputs never enter fitting.</p>'
    body+='<h2>Eight-repeat confirmation and supplementary checks</h2>'+table.to_html(index=False,float_format=lambda x:f'{x:.5f}',border=0)
    body+='<p>Both CV pools use eight repeats of five folds: 157 gold labels for training CV, 235 for pooled CV. Validation remains a separate 78-trial measurement of a 157-label fit. Chronology pools 98 distinct trials across three fixed later-year windows. Repeated CV shares data; these are development estimates, not independent tests. No Kaggle score is implied by a local improvement.</p>'
    body+='<h2>All initial three-repeat results, including unsuccessful variants</h2>'+screen.to_html(index=False,float_format=lambda x:f'{x:.5f}',border=0)
    details=pd.DataFrame([dict(candidate=x['candidate'],reason=l,**v) for x in temporal for l,v in x['reasons'].items()])
    body+='<h2>Chronological per-reason results</h2>'+details.to_html(index=False,float_format=lambda x:f'{x:.5f}',border=0)
    body+='<p><a href="protocol.json">Frozen protocol</a> · <a href="registry.json">Recipes</a> · <a href="shortlist.json">CV shortlist</a> · <a href="confirmation.json">All eight repeat scores</a> · <a href="verification.json">Verification</a></p>'
    (OUT/'report.html').write_text(body)
    print(table.to_string(index=False),flush=True)


def verify():
    guard();checks=[]
    rs={c['name']:c for c in registry()}
    for kind in ['train','pooled']:
        frame=r.frames()[0 if kind=='train' else 2]
        gold=frame[frame[e.LABELS].ge(0).all(axis=1)]
        for count,repeat_count in [('3',3),('5',5)]:
            folder=OUT/(kind+count)
            for row in json.loads((folder/'index.json').read_text()):
                a=np.load(folder/(row['result_key']+'.npz'),allow_pickle=True)
                assert a['ids'].tolist()==gold.ID.tolist()
                np.testing.assert_array_equal(a['truth'],gold[e.LABELS].to_numpy(int))
                assert len(a['scores'])==repeat_count
                for i,p in enumerate(a['scores']):
                    np.testing.assert_array_equal(e.decisions(p,'t0.5'),a['predictions'][i])
                    assert abs(e.metrics(a['truth'],p,'t0.5')['sample_f1']-row['repeat_scores'][i]['sample_f1'])<1e-12
                if row['candidate']==CONTROL:
                    old_dir=r.OUT/(kind+'_cv') if count=='3' else extension.OUT/kind
                    ref=next(x for x in json.loads((old_dir/'index.json').read_text()) if x['candidate']=='Ani training-only adaptation')
                    old=np.load(old_dir/(ref['result_key']+'.npz'),allow_pickle=True)
                    np.testing.assert_array_equal(a['predictions'],old['predictions'])
                checks.append(dict(stage=kind+count,candidate=row['candidate'],repeats_rescored=repeat_count))
    for row in json.loads((OUT/'supplementary/results.json').read_text()):
        d=pd.read_csv(r.ROOT/row['prediction_file']);y=d[[f'actual_{l}' for l in e.LABELS]].to_numpy(int);h=d[[f'predicted_{l}' for l in e.LABELS]].to_numpy(int)
        assert abs(e.metrics(y,h,'t0.5')['sample_f1']-row['sample_f1'])<1e-12
    r.save(OUT/'verification.json',dict(passed=True,checks=checks,baseline_predictions_identical_to_prior_benchmark=True,
        all_gold_labels_and_ids_verified=True,protected_sha256=protected(),source_sha256=own_sources()))
    print('All CV/validation/chronology predictions rescored; Ani control exactly matches prior benchmark; protected artifacts unchanged.',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','parity','train','pooled','shortlist','extra_train','extra_pooled','confirm','supplementary','report','verify']);args=parser.parse_args()
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        if args.stage in ['train','pooled']:run(args.stage)
        elif args.stage.startswith('extra_'):run(args.stage[6:],True)
        else:globals()[args.stage]()


if __name__=='__main__':main()
