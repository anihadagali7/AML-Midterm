"""Export the authorized exploratory fixed blend; no tuning or upload."""
import os
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[v]='1'
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
P1=HERE.parents[1]
STUDY=HERE.parent/'complementary_heads_v2'
sys.path.insert(0,str(STUDY))
import strong_study as wrapper
study=wrapper.study
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits

CSV=P1/'submissions/candidates/anchored80_diverse_kernel20_trainval235.csv'
PARENT_SHA='c20aa937fa8b1e5aea55bcd6564c6acd952f0d74030cc7c1a437eac54a1ee257'
COMPONENT_SHA='640ca62408f5dfc9e92ba54fafce500b94387e3576ae5b5db6becb3f6f136c8a'


def frame(ids,pred):
    return pd.DataFrame(dict(ID=ids,Predicted=[' '.join(label for label,on in zip(study.LABELS,row) if on) for row in pred]))


def sources():
    files=[Path(__file__),HERE/'verify_fixed.py',HERE/'parent_final_components.npz',
           STUDY/'protocol.json',STUDY/'results/summary/verification.json',STUDY/'results/summary/overview.csv']
    return {str(p.relative_to(P1)):study.sha(p) for p in files}


def main():
    study.guard();assert not CSV.exists() and not (HERE/'export_protocol.json').exists()
    assert study.sha(HERE/'parent_final_components.npz')==COMPONENT_SHA
    overview=pd.read_csv(STUDY/'results/summary/overview.csv')
    metrics=overview[overview.recipe=='blend_diverse30'].set_index('pool').mean_sample_f1
    study.save(HERE/'export_protocol.json',dict(registered_at=datetime.now(timezone.utc).isoformat(),
        recipe='complementary-heads-v2/blend_diverse30: raw .8*anchored+.2*diverse_kernel_probability; .3 cutoff+argmax',
        source_benchmark_commit='735ad7a3',source_benchmark_protocol_sha256=study.sha(STUDY/'protocol.json'),
        trigger='User requested the best candidate to try despite the negative advancement result, then explicitly chose fixed rather than learned combination: nah do fixed.',
        selection='Select stronger-head v2 fixed blend with diversity penalty .3 before fitting or inspecting new test predictions. It has the highest pooled CV among v2 combined candidates and ties the best Train157. Do not select by test changes.',
        exception='User-authorized exploratory final refit after failed both-pool advancement. Does not change original benchmark status or promote this recipe. No upload is performed; best reported Kaggle remains .89935.',
        settings='Unchanged frozen v2 source. Kernel nonintercept L2 .00005,128 Nyström RBF components,gamma1/150,seed270303,diversity .3,ordinary warm start. Full235 kernel uses anchor crossfit seed270301 with group purges. Gold-only kernel supervised loss, eligible gold+masked feature geometry. Anchor unchanged.',
        fitting='235 supplied gold +1014 originally masked rows, original train-then-validation order. Exact benchmark fit_pair entrypoint refits all nested training stages; only the fixed blend is exported. No learned combiner is applied to the submission. Query inputs prediction-only.',
        known_metrics={k:float(metrics[k]) for k in ['train','pooled','validation','chronology98']},
        parent_reference=dict(score=.89935,source_commit='cde3d500',
            original_components='part1/experiments/cand2_followup_v1/blend_final_predictions.npz',
            components_sha256=COMPONENT_SHA,submission_sha256=PARENT_SHA,
            access='Cached final test scores only validate unchanged anchored and count differences; never enter candidate training or selection.'),
        output=str(CSV.relative_to(P1)),files=sources()))
    train=pd.read_csv(P1/'data/part1_train.csv');val=pd.read_csv(P1/'data/part1_val.csv')
    fit=pd.concat([train,val],ignore_index=True)
    test=pd.read_csv(P1/'data/part1_test.csv');sample=pd.read_csv(P1/'data/part1_sample_submission.csv')
    y=fit[study.LABELS].to_numpy(int)
    assert int((y>=0).all(1).sum())==235 and int((y==-100).all(1).sum())==1014
    assert set(fit.ID).isdisjoint(test.ID)
    prior=study.loadnp(HERE/'parent_final_components.npz')
    np.testing.assert_array_equal(prior['ids'],test.ID)
    with threadpool_limits(limits=1):
        scores,audit,anchored=wrapper.fit_pair(fit,test[study.EMB].to_numpy(float))
    np.testing.assert_allclose(anchored,prior['anchored_scores'],atol=1e-7,rtol=0)
    parent_pred=study.select_reasons(.8*anchored+.2*prior['cand2_scores'])
    parent=frame(test.ID,parent_pred)
    assert hashlib.sha256(parent.to_csv(index=False,lineterminator='\n').encode()).hexdigest()==PARENT_SHA
    second=expit(logit(np.clip(scores['head_diverse30'],1e-12,1-1e-12))+logit(.3))
    raw=.8*anchored+.2*second
    pred=study.select_reasons(raw)
    np.testing.assert_array_equal(pred,study.decide(scores['blend_diverse30']))
    assert np.isfinite(raw).all() and (pred.sum(1)>0).all()
    out=frame(test.ID,pred)
    assert len(out)==312 and out.ID.is_unique and out.ID.tolist()==sample.ID.tolist()
    CSV.parent.mkdir(parents=True,exist_ok=True);out.to_csv(CSV,index=False,lineterminator='\n')
    np.savez_compressed(HERE/'final_predictions.npz',ids=test.ID.to_numpy(str),anchored_scores=anchored,
        diverse_scores=second,scores=raw,decision_scores=scores['blend_diverse30'],predictions=pred,parent_predictions=parent_pred)
    changed=(pred!=parent_pred).any(1)
    pd.DataFrame(dict(ID=test.ID[changed],parent=parent.Predicted[changed],candidate=out.Predicted[changed])).to_csv(HERE/'changed_predictions.csv',index=False)
    study.save(HERE/'fit_audit.json',dict(fit_ids_sha256=study.idhash(fit.ID),query_ids_sha256=study.idhash(test.ID),
        fit_gold_ids_sha256=study.idhash(fit.loc[(y>=0).all(1),'ID']),**audit))
    record=dict(status='Prepared for manual exploratory submission; not uploaded; score unknown',
        recipe='blend_diverse30',csv=str(CSV.relative_to(P1)),sha256=study.sha(CSV),rows=312,gold=235,masked=1014,
        source_benchmark_commit='735ad7a3',source_benchmark_protocol_sha256=study.sha(STUDY/'protocol.json'),
        export_protocol_sha256=study.sha(HERE/'export_protocol.json'),
        prediction_sha256=study.sha(HERE/'final_predictions.npz'),fit_audit_sha256=study.sha(HERE/'fit_audit.json'),
        test_inputs_used_for_fit=False,learned_combiner_used=False,parent_scored_csv_reproduced=True,
        parent_max_score_error=float(np.abs(anchored-prior['anchored_scores']).max()),changed_label_sets=int(changed.sum()),
        added_labels={name:int(((pred[:,j]==1)&(parent_pred[:,j]==0)).sum()) for j,name in enumerate(study.LABELS)},
        removed_labels={name:int(((pred[:,j]==0)&(parent_pred[:,j]==1)).sum()) for j,name in enumerate(study.LABELS)})
    study.save(HERE/'prepared_submission.json',record)
    assert study.read(HERE/'export_protocol.json')['files']==sources();study.guard()
    print(record)


if __name__=='__main__':main()
