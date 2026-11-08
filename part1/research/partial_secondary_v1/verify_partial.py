"""Independent arithmetic, boundary and partial-label contract verification."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import study_run as study


def metrics(y,p):
    tp=(y*p).sum(0);fp=((1-y)*p).sum(0);fn=(y*(1-p)).sum(0)
    pr=np.divide(tp,tp+fp,out=np.zeros(5),where=(tp+fp)>0)
    re=np.divide(tp,tp+fn,out=np.zeros(5),where=(tp+fn)>0)
    f=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(5),where=(2*tp+fp+fn)>0)
    sample=(2*(y*p).sum(1)/(y.sum(1)+p.sum(1))).mean()
    return float(sample),float(f.mean()),pr,re,f,y.sum(0)


def verify(output):
    study.guard();output=Path(output)
    recorded=pd.read_csv(output/'summary/repeat_scores.csv')
    labs=pd.read_csv(output/'summary/per_label_scores.csv');pairs=pd.read_csv(output/'summary/paired_summary.csv')
    grouped={};n=0;max_error=0.;unknown_rounds=0
    for c in study.cases():
        path=output/'predictions'/(c['key']+'.npz');z=study.loadnp(path);audit=study.read(path.with_suffix('.json'))
        assert study.sha(path)==audit['sha256']
        np.testing.assert_array_equal(z['ids'],c['query'].ID);np.testing.assert_array_equal(z['truth'],c['query'][study.LABELS])
        np.testing.assert_array_equal(z['names'],study.NAMES)
        assert study.idhash(c['fit'].ID)==audit['fit_ids_sha256'] and study.idhash(c['query'].ID)==audit['query_ids_sha256']
        assert set(c['fit'].ID).isdisjoint(c['query'].ID)
        if c['key']!='validation':assert not set(study.benchmark.groups(c['fit']))&set(study.benchmark.groups(c['query']))
        Y=c['fit'][study.LABELS].to_numpy(int);gold=(Y>=0).all(1);neg=int((Y[gold,4]==0).sum())
        for mode,rounds in audit['rounds'].items():
            assert len(rounds)==3
            for r in rounds:
                assert r['gold']==int(gold.sum())
                assert r['observed_other_positive']+r['observed_other_negative']+r['unknown_other']==r['gold']+r['generated']
                if mode in ['neighbor_unknown','binary_unknown']:
                    assert r['observed_other_negative']==neg
                    assert r['unknown_other']>=0
                    unknown_rounds+=1
                else:assert r['unknown_other']==0
                if mode.startswith('binary_'):
                    assert r['binary_training_rows']==r['observed_other_positive']+r['observed_other_negative']
                    np.testing.assert_allclose(r['binary_weight_by_class'],np.array([.8,.2])*r['gold'],atol=1e-10,rtol=0)
        a,cc=study.reference(c)
        np.testing.assert_array_equal(z['scores'][0],.8*a+.2*cc)
        for name in ['binary_complete','binary_unknown']:
            np.testing.assert_allclose(z['scores'][study.NAMES.index(name+'_half')],
                .5*z['scores'][0]+.5*z['scores'][study.NAMES.index(name)],atol=1e-12,rtol=0)
        for j in range(len(study.NAMES)):
            p=(z['scores'][j]>=.3).astype(int);p[np.arange(len(p)),z['scores'][j].argmax(1)]=1
            np.testing.assert_array_equal(p,z['predictions'][j])
        key=(c['pool'],c['seed']) if c['pool']!='supplementary' else (c['key'],-1)
        grouped.setdefault(key,[]).append(z)
        if c['key'].startswith('through_'):grouped.setdefault(('chronology98',-1),[]).append(z)
        n+=1;max_error=max(max_error,audit['parent_max_error'])
    checked=0
    for (pool,seed),zs in grouped.items():
        y=np.concatenate([z['truth'] for z in zs]);ids=np.concatenate([z['ids'] for z in zs])
        assert len(ids)==len(set(ids))
        if pool in ['train','pooled']:assert len(zs)==5 and len(y)==(157 if pool=='train' else 235)
        if pool=='chronology98':assert len(y)==98
        for j,name in enumerate(study.NAMES):
            p=np.concatenate([z['predictions'][j] for z in zs]);s,m,pr,re,f,support=metrics(y,p)
            r=recorded[(recorded.pool==pool)&(recorded.seed==seed)&(recorded.recipe==name)].iloc[0]
            np.testing.assert_allclose([s,m],[r.sample_f1,r.macro_f1],atol=1e-12,rtol=0)
            l=labs[(labs.pool==pool)&(labs.seed==seed)&(labs.recipe==name)].set_index('label').loc[study.LABELS]
            np.testing.assert_allclose(np.c_[pr,re,f,support],l[['precision','recall','f1','support']],atol=1e-12,rtol=0)
            checked+=1
    for _,r in pairs.iterrows():
        d=recorded[recorded.pool==r.pool].pivot(index='seed',columns='recipe',values='sample_f1')
        v=d[r.recipe]-d[r.control]
        np.testing.assert_allclose(v.mean(),r.mean_delta,atol=1e-12,rtol=0)
        assert int((v>1e-12).sum())==r.wins and int((v.abs()<=1e-12).sum())==r.ties
    expected={'train':.935244161358811,'pooled':.9216312056737589,'validation':.8910256410256411,'chronology98':.8877551020408163}
    for pool,value in expected.items():
        np.testing.assert_allclose(recorded[(recorded.pool==pool)&(recorded.recipe=='parent')].sample_f1.mean(),value,atol=1e-12,rtol=0)
    receipt=dict(passed=True,outer_cases=n,metric_rows_independently_checked=checked,
        partial_label_rounds_checked=unknown_rounds,gold_negative_labels_preserved=True,
        class_prior_weight_mass_fixed=True,parent_max_score_error=max_error,
        parent_four_headlines_reproduced=True,ids_groups_and_coverage_checked=True,
        half_blend_formulas_checked=True,all_source_hashes_unchanged=True,
        protocol_sha256=study.sha(study.HERE/'protocol.json'))
    study.save(output/'summary/verification.json',receipt);print(receipt)


if __name__=='__main__':verify(Path(sys.argv[1]) if len(sys.argv)>1 else study.HERE/'results')
