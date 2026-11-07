"""Audit reference parity, inner-fold decision selection, and immutable outputs."""
import json
import numpy as np
import pandas as pd
from simple_benchmark import ROOT, digest, load_data
from simple_models import REASONS, decide
from trusted_benchmark import OUT
from trusted_models import sample_f1


def main():
    known, unknown = load_data()
    old = np.load(ROOT/'experiments/simple_methodical/07_final_neighbor_check/probabilities.npz')['final_control_r3']
    control = np.load(OUT/'01_trust_and_anchoring/probabilities.npz')['trust_augmented_w1']
    np.testing.assert_allclose(old,control,atol=1e-10,rtol=1e-8)
    inner_checks=0;fit_checks=0;stage_count=0
    for path in sorted(OUT.glob('*/fit_audits.json')):
        stage_count+=1
        audits=json.loads(path.read_text())
        probs=np.load(path.parent/'probabilities.npz');predictions=np.load(path.parent/'predictions.npz')
        for audit in audits:
            fitting,held=audit['fit_ids'],audit['held_ids']
            assert len(fitting)==188 and len(held)==47 and set(fitting).isdisjoint(held)
            assert set(fitting)|set(held)==set(known.ID)
            assert set(fitting).isdisjoint(unknown.ID)
            fit_checks+=1
            inner=audit.get('inner',{})
            if not inner:
                continue
            fold=np.array(inner['inner_fold']);P=np.array(inner['inner_probabilities'])
            assert P.shape==(188,5) and np.isfinite(P).all() and set(fold)=={0,1,2}
            Y=known.set_index('ID').loc[fitting,REASONS].to_numpy(int)
            scores=[]
            for threshold in inner['candidates']:
                thresholds=np.array([.3,.3,.3,.3,threshold])
                scores.append(float(sample_f1(Y,decide(P,thresholds))))
            np.testing.assert_allclose(scores,inner['scores'],atol=1e-14)
            best=next(i for i,x in enumerate(scores) if x>=max(scores)-1e-12)
            assert inner['threshold']==inner['candidates'][best]
            repeat=audit['repeat'];outerfold=audit['fold'];name=audit['name']
            held_mask=predictions['folds'][repeat]==outerfold
            np.testing.assert_array_equal(decide(probs[name][repeat,held_mask],audit['cutoffs']),predictions[name][repeat,held_mask])
            inner_checks+=1
    assert inner_checks==80
    manifest=json.loads((OUT/'candidate/manifest.json').read_text())
    for name,expected in manifest['source_sha256'].items():
        assert digest(ROOT/name)==expected,name
    for name,expected in manifest['artifact_sha256'].items():
        assert digest(OUT/'candidate'/name)==expected,name
    predictions=pd.read_csv(OUT/'candidate/part1_trusted_submission.csv')
    template=pd.read_csv(ROOT/'data/part1_sample_submission.csv')
    assert predictions.columns.tolist()==['ID','Predicted'] and predictions.ID.tolist()==template.ID.tolist()
    assert predictions.Predicted.map(lambda x:bool(x.split()) and set(x.split())<=set(REASONS)).all()
    result=dict(reference_control_all40folds_reproduced=True,outer_fit_role_audits=fit_checks,
                inner_threshold_selections_rescored=inner_checks,inner_fit_count=3*inner_checks,
                inner_labels_aligned_with_saved_outer_training_ids=True,
                candidate_fold_replays=manifest['independent_fold_replays'],candidate_diagnostic_replays=manifest['independent_diagnostic_replays'],
                candidate_artifact_and_source_checksums_valid=True,submission_rows=len(predictions),
                sample_submission_order_and_valid_labels=True,protected_originals_unchanged=manifest['protected_originals_unchanged'])
    (OUT/'final_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
