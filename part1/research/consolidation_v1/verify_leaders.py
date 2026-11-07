"""Rescore retained leader/reference arrays without requiring failed-run caches."""
import json
import re
import numpy as np
from evidence import ROOT, HERE, sha


def verify():
    record=json.loads((HERE/'retention.json').read_text())
    counts=dict(arrays=0,repeat_scores=0,supplementary_scores=0,final_submissions=0)
    for row in record['retained_files']:
        p=ROOT/row['path']
        if p.suffix!='.npz':continue
        assert sha(p)==row['sha256'],str(p)
        with np.load(p,allow_pickle=True) as z:
            scores=z['scores'];pred=z['predictions'];ids=z['ids']
            assert np.isfinite(scores).all() and len(set(ids))==len(ids),str(p)
            s=scores[None] if scores.ndim==2 else scores
            q=pred[None] if pred.ndim==2 else pred
            decided=(s>=.5).astype(int)
            empty=decided.sum(axis=2)==0
            ri,pi=np.where(empty);decided[ri,pi,s.argmax(axis=2)[ri,pi]]=1
            np.testing.assert_array_equal(decided,q,err_msg=str(p))
            counts['arrays']+=1
            if 'truth' not in z:
                counts['final_submissions']+=1
                continue
            truth=z['truth']
            assert truth.shape==(len(ids),5) and np.isin(truth,[0,1]).all()
            values=(2*(q*truth).sum(axis=2)/(q.sum(axis=2)+truth.sum(axis=1))).mean(axis=1)
            index=p.parent/'index.json'
            if index.exists():
                rows=json.loads(index.read_text())
                entry=next(r for r in rows if r.get('candidate')==p.stem or r.get('result_key')==p.stem)
                np.testing.assert_allclose(values,[r['sample_f1'] for r in entry['repeat_scores']],atol=1e-12,rtol=0)
                counts['repeat_scores']+=len(values)
            elif p.parent.name=='supplementary':
                wi,name=re.fullmatch(r'window(\d+)_(.+)',p.stem).groups()
                rows=json.loads((p.parent/'results.json').read_text())
                entry=next(r for r in rows if r['window']==int(wi) and r['candidate']==name)
                assert abs(float(values[0])-entry['sample_f1'])<1e-12,str(p)
                counts['supplementary_scores']+=1
            else:
                # The unsubmitted cutoff lead keeps combined CV summaries.
                rows=json.loads((p.parent.parent/'summary.json').read_text())['final_comparison']
                entry=next(r for r in rows if r['candidate']==p.stem)
                pool='pooled' if p.parent.name.startswith('pooled') else 'train'
                expected=entry[pool+'_repeats'][:3] if p.parent.name.endswith('3') else entry[pool+'_repeats'][3:]
                np.testing.assert_allclose(values,expected,atol=1e-12,rtol=0)
                counts['repeat_scores']+=len(values)
    return dict(passed=True,**counts,models_fitted=0)


if __name__=='__main__':
    print(json.dumps(verify(),indent=2))
