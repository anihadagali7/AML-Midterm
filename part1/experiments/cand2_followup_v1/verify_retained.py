"""Check retained matched comparisons and independently rescore them; no model fits."""
import json
import csv
import tempfile
from pathlib import Path
import shutil
import numpy as np
from run import HERE,P1,LABELS,read,sha,freeze,collect,save
from analyze import independent


def main():
    freeze()
    retained=read(HERE/'retention.json')
    for name,digest in retained['sha256'].items():assert sha(HERE/name)==digest,name
    for name,meta in read(HERE/'input_provenance.json').items():assert sha(HERE/'inputs'/name)==meta['subset_sha256'],name
    for name,digest in read(HERE/'provenance.json')['copied_sha256'].items():assert sha(HERE/name)==digest,name
    export=read(HERE/'blend_prepared_submission.json');protocol=read(HERE/'blend_export_protocol.json')
    assert sha(P1/export['csv'])==export['sha256']
    assert sha(HERE/'export_blend.py')==protocol['builder_sha256']
    with np.load(HERE/'blend_final_predictions.npz') as z:
        scores=(4*z['anchored_scores']+z['cand2_scores'])/5
        np.testing.assert_allclose(scores,z['scores'],atol=1e-15,rtol=0)
        pred=(scores>=.3).astype(int);pred[np.arange(len(pred)),scores.argmax(1)]=1
        np.testing.assert_array_equal(pred,z['predictions'])
        with (P1/export['csv']).open(newline='') as f:rows=list(csv.DictReader(f))
        assert [r['ID'] for r in rows]==z['ids'].tolist()
        assert [r['Predicted'] for r in rows]==[' '.join(l for l,v in zip(LABELS,p) if v) for p in pred]
    output=Path(tempfile.mkdtemp(prefix='cand2-followup-verify-'))
    try:
        for p in (HERE/'results/J_packets').iterdir():
            if p.is_file():shutil.copyfile(p,output/p.name)
        metrics,labels,audits,_=collect(output)
        metrics.to_csv(output/'repeat_metrics.csv',index=False);labels.to_csv(output/'label_metrics.csv',index=False)
        independent(output)
        for name in ['summary.csv','matched_J.csv','rank_agreement.csv','expanded_panel.csv','pairwise_orderings.csv','repeat_metrics.csv','label_metrics.csv','complementarity.csv']:
            assert (output/'analysis'/name).read_bytes()==(HERE/'results'/name).read_bytes(),name
        with np.load(output/'native_dylan__validation.npz') as new,np.load(HERE/'inputs/strict_inductive__validation.npz') as old:
            np.testing.assert_array_equal(new['cand2_predictions'],old['cand2_predictions'])
            np.testing.assert_allclose(new['cand2_scores'],old['cand2_scores'],atol=1e-12,rtol=0)
        print('Retained hashes, repeated aggregation, independent rescoring and exact-recipe validation control passed.')
    finally:shutil.rmtree(output)


if __name__=='__main__':main()
