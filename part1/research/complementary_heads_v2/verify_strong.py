"""Independent metric arithmetic and reconstruction of every nested boundary."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import expit, logit
import strong_study as wrapper
study=wrapper.study


def metrics(y,p):
    tp=(y*p).sum(0);fp=((1-y)*p).sum(0);fn=(y*(1-p)).sum(0)
    pr=np.divide(tp,tp+fp,out=np.zeros(5),where=tp+fp>0)
    re=np.divide(tp,tp+fn,out=np.zeros(5),where=tp+fn>0)
    f=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(5),where=2*tp+fp+fn>0)
    sample=np.mean(2*(y*p).sum(1)/(y.sum(1)+p.sum(1)))
    return sample,f.mean(),np.c_[pr,re,f,y.sum(0)]


def verify(output):
    study.guard();output=Path(output)
    table=pd.read_csv(output/'summary/repeat_scores.csv')
    label_table=pd.read_csv(output/'summary/per_label_scores.csv')
    pairs=pd.read_csv(output/'summary/paired_summary.csv')
    grouped={};boundaries=0;cases=study.cases()
    earlier=study.loadnp(wrapper.V1/'results/matched_predictions.npz')
    for case in cases:
        path=output/'predictions'/(case['key']+'.npz')
        z=study.loadnp(path);audit=study.read(path.with_suffix('.json'))
        assert audit['sha256']==study.sha(path)
        np.testing.assert_array_equal(z['ids'],case['query'].ID)
        np.testing.assert_array_equal(z['truth'],case['query'][study.LABELS])
        np.testing.assert_array_equal(z['names'],study.NAMES)
        old=earlier['case']==case['key']
        np.testing.assert_array_equal(earlier['ids'][old],z['ids'])
        np.testing.assert_allclose(z['scores'][study.NAMES.index('calibration')],
            earlier['scores'][earlier['names'].tolist().index('calibration'),old],atol=1e-12,rtol=0)
        assert study.idhash(case['fit'].ID)==audit['fit_ids_sha256']
        assert study.idhash(case['query'].ID)==audit['query_ids_sha256']
        assert set(case['fit'].ID).isdisjoint(case['query'].ID)
        if case['key']!='validation':
            assert not set(study.benchmark.groups(case['fit']))&set(study.benchmark.groups(case['query']))
        for step,(_,_,part,held,seed) in enumerate(study.inner_inputs(case['fit'],270301)):
            ai=audit['inner'][step]
            assert ai['fitting_ids_sha256']==study.idhash(part.ID)
            assert ai['query_ids_sha256']==study.idhash(held.ID) and ai['seed']==seed
            boundaries+=1
            for k,(_,_,sub,sh,ss) in enumerate(study.inner_inputs(part,270302)):
                aj=ai['error_training_subfolds'][k]
                assert aj['fitting_ids_sha256']==study.idhash(sub.ID)
                assert aj['query_ids_sha256']==study.idhash(sh.ID) and aj['seed']==ss
                assert set(sub.ID).isdisjoint(held.ID) and set(sh.ID).isdisjoint(held.ID)
                boundaries+=1
        a,c=study.reference(case)
        expected=expit(logit(np.clip(.8*a+.2*c,1e-6,1-1e-6))-logit(.3))
        np.testing.assert_allclose(z['scores'][0],expected,atol=1e-12,rtol=0)
        np.testing.assert_array_equal(z['predictions'][1],study.select_reasons(a))
        for name in study.STRENGTHS:
            np.testing.assert_allclose(z['scores'][study.NAMES.index('blend_'+name)],
                expit(logit(np.clip(.8*a+.2*expit(logit(z['scores'][study.NAMES.index('head_'+name)])+logit(.3)),1e-6,1-1e-6))-logit(.3)),atol=1e-12,rtol=0)
        for j in range(len(study.NAMES)):
            p=(z['scores'][j]>=.5).astype(int)
            p[np.arange(len(p)),z['scores'][j].argmax(1)]=1
            np.testing.assert_array_equal(p,z['predictions'][j])
        key=(case['pool'],case['seed']) if case['pool']!='supplementary' else (case['key'],-1)
        grouped.setdefault(key,[]).append(z)
        if case['key'].startswith('through_'):grouped.setdefault(('chronology98',-1),[]).append(z)
    checked=0
    for (pool,seed),packets in grouped.items():
        y=np.concatenate([z['truth'] for z in packets]);ids=np.concatenate([z['ids'] for z in packets])
        assert len(ids)==len(set(ids))
        if pool in ['train','pooled']:assert len(packets)==5 and len(y)==(157 if pool=='train' else 235)
        if pool=='chronology98':assert len(y)==98
        for j,name in enumerate(study.NAMES):
            p=np.concatenate([z['predictions'][j] for z in packets]);s,m,l=metrics(y,p)
            recorded=table[(table.pool==pool)&(table.seed==seed)&(table.recipe==name)].iloc[0]
            np.testing.assert_allclose([s,m],[recorded.sample_f1,recorded.macro_f1],atol=1e-12,rtol=0)
            labels=label_table[(label_table.pool==pool)&(label_table.seed==seed)&(label_table.recipe==name)].set_index('label').loc[study.LABELS]
            np.testing.assert_allclose(l,labels[['precision','recall','f1','support']],atol=1e-12,rtol=0)
            checked+=1
    for _,row in pairs.iterrows():
        pivot=table[table.pool==row.pool].pivot(index='seed',columns='recipe',values='sample_f1')
        delta=pivot[row.recipe]-pivot[row.control]
        np.testing.assert_allclose(delta.mean(),row.mean_delta,atol=1e-12,rtol=0)
        assert int((delta>1e-12).sum())==row.wins
        assert int((delta.abs()<=1e-12).sum())==row.ties
    expected={'train':.935244161358811,'pooled':.9216312056737589,'validation':.8910256410256411,'chronology98':.8877551020408163}
    for pool,value in expected.items():
        np.testing.assert_allclose(table[(table.pool==pool)&(table.recipe=='parent')].sample_f1.mean(),value,atol=1e-12,rtol=0)
    receipt=dict(passed=True,outer_cases=len(cases),nested_boundaries_checked=boundaries,
        independent_metric_rows=checked,independent_label_metric_rows=checked*5,
        all_saved_ids_and_groups_checked=True,protected_source_hashes_unchanged=True,
        all_four_parent_headlines_reproduced=True,v1_calibration_reproduced=True,
        protocol_sha256=study.sha(study.HERE/'protocol.json'))
    study.save(output/'summary/verification.json',receipt);print(receipt)


if __name__=='__main__':verify(Path(sys.argv[1]) if len(sys.argv)>1 else study.HERE/'results')
