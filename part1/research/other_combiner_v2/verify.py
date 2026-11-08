"""Independent rescore, fitting-boundary checks and frozen-parent comparison."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run as study


def independent(y,p):
    tp=(y*p).sum(0);fp=((1-y)*p).sum(0);fn=(y*(1-p)).sum(0)
    precision=np.divide(tp,tp+fp,out=np.zeros(5),where=(tp+fp)>0)
    recall=np.divide(tp,tp+fn,out=np.zeros(5),where=(tp+fn)>0)
    f=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(5),where=(2*tp+fp+fn)>0)
    sample=(2*(y*p).sum(1)/(y.sum(1)+p.sum(1))).mean()
    return sample,f.mean(),precision,recall,f,y.sum(0)


def verify(output):
    study.guard();output=Path(output)
    recorded=pd.read_csv(output/'summary/repeat_scores.csv')
    labels=pd.read_csv(output/'summary/per_label_scores.csv')
    pairs=pd.read_csv(output/'summary/paired_summary.csv')
    groups={};outer_count=0;inner_count=0;coefs=[]
    for c in study.cases():
        path=output/'predictions'/(c['key']+'.npz')
        z=study.loadnp(path);meta=study.read(path.with_suffix('.json'))
        assert study.sha(path)==meta['sha256']
        assert z['names'].tolist()==study.NAMES
        np.testing.assert_array_equal(z['ids'],c['query'].ID)
        np.testing.assert_array_equal(z['truth'],c['query'][study.LABELS])
        a,cc=study.refs(c);native=.8*a+.2*cc
        parent=(native>=.3).astype(int);parent[np.arange(len(native)),native.argmax(1)]=1
        np.testing.assert_array_equal(parent,z['predictions'][0])
        assert np.isfinite(z['scores']).all() and (z['predictions'].sum(2)>0).all()
        assert meta['fitting_ids_sha256']==study.idhash(c['fit'].ID)
        assert set(c['fit'].ID).isdisjoint(c['query'].ID)
        if c['key']!='validation':
            assert not set(study.benchmark.groups(c['fit']))&set(study.benchmark.groups(c['query']))
        gold=c['fit'][c['fit'][study.LABELS].ge(0).all(axis=1)]
        held=[]
        for inner in meta['inner']:
            fitted=c['fit'].set_index('ID',drop=False).loc[inner['train_ids']]
            query=gold.set_index('ID',drop=False).loc[inner['query_ids']]
            assert set(fitted.ID)<=set(c['fit'].ID)
            assert set(fitted.ID).isdisjoint(c['query'].ID)
            assert set(fitted.ID).isdisjoint(query.ID)
            assert not set(study.benchmark.groups(fitted))&set(study.benchmark.groups(query))
            assert set(fitted[study.LABELS].min(axis=1).unique())<=set([-100,0,1])
            held+=inner['query_ids'];inner_count+=1
        assert sorted(held)==sorted(gold.ID) and len(held)==len(set(held))
        np.testing.assert_array_equal(z['inner_gold_ids'],gold.ID)
        for n in ['other_rbf','other_conditional_rbf','combiner_other_only']:
            j=study.NAMES.index(n)
            np.testing.assert_array_equal(z['scores'][j,:,:4],z['scores'][0,:,:4])
        # Reconstruct every fitted meta-head from saved genuine inner-OOF predictions.
        heads,coefficients=study.learned_heads(z['inner_anchored'],z['inner_cand2'],z['inner_specialists'],
            a,cc,z['outer_specialists'],gold[study.LABELS].to_numpy(int))
        assert coefficients==meta['coefficients']
        for name,s in heads.items():np.testing.assert_allclose(s,z['scores'][study.NAMES.index(name)],atol=1e-12,rtol=0)
        for name,coef in coefficients.items():
            coefs.append(dict(case=c['key'],pool=c['pool'],recipe=name,coefficients=coef))
        key=(c['pool'],c['seed']) if c['pool']!='supplementary' else (c['key'],-1)
        groups.setdefault(key,[]).append(z)
        if c['key'].startswith('through_'):groups.setdefault(('chronology98',-1),[]).append(z)
        outer_count+=1
    checked=0
    for (pool,seed),zs in groups.items():
        y=np.concatenate([z['truth'] for z in zs]);ids=np.concatenate([z['ids'] for z in zs])
        assert len(ids)==len(set(ids))
        if pool in ['train','pooled']:assert len(zs)==5 and len(y)==(157 if pool=='train' else 235)
        for j,name in enumerate(study.NAMES):
            p=np.concatenate([z['predictions'][j] for z in zs])
            sample,macro,pr,re,f,support=independent(y,p)
            saved=recorded[(recorded.pool==pool)&(recorded.seed==seed)&(recorded.recipe==name)].iloc[0]
            np.testing.assert_allclose([sample,macro],[saved.sample_f1,saved.macro_f1],atol=1e-12,rtol=0)
            lab=labels[(labels.pool==pool)&(labels.seed==seed)&(labels.recipe==name)].set_index('label').loc[study.LABELS]
            np.testing.assert_allclose(np.c_[pr,re,f,support],lab[['precision','recall','f1','support']],atol=1e-12,rtol=0)
            checked+=1
    for _,r in pairs.iterrows():
        d=recorded[recorded.pool==r.pool].pivot(index='seed',columns='recipe',values='sample_f1')
        v=d[r.recipe]-d[r.control]
        np.testing.assert_allclose(v.mean(),r.mean_delta,atol=1e-12,rtol=0)
        assert int((v>1e-12).sum())==r.wins and int((v.abs()<=1e-12).sum())==r.ties
    expected={'train':.935244161358811,'pooled':.9216312056737589,'validation':.8910256410256411,'chronology98':.8877551020408163}
    for pool,value in expected.items():
        np.testing.assert_allclose(recorded[(recorded.pool==pool)&(recorded.recipe=='parent')].sample_f1.mean(),value,atol=1e-12,rtol=0)
    study.save(output/'summary/coefficients.json',coefs)
    result=dict(passed=True,outer_cases=outer_count,inner_folds=inner_count,metric_rows_checked=checked,
        independent_sample_and_per_label_arithmetic=True,parent_four_headlines_reproduced=True,
        ids_groups_and_nested_coverage_checked=True,all_learned_heads_reconstructed=True,
        all_source_hashes_unchanged=True,protocol_sha256=study.sha(study.HERE/'protocol.json'))
    study.save(output/'summary/verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':verify(Path(sys.argv[1]) if len(sys.argv)>1 else study.HERE/'results')
