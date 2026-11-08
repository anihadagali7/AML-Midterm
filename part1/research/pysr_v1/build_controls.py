"""Snapshot matched controls from a frozen Git commit, never a moving worktree."""
import io
import importlib.util
import json
import subprocess
import sys
import hashlib
import numpy as np
from common import HERE,PART1,ROOT,benchmark,sha,save

COMMIT='cde3d500e1304f2afeeb6dc4c7f50e49190c6a4b'
BASE='part1/experiments/cand2_followup_v1/'
blob=lambda path:subprocess.check_output(['git','-C',str(ROOT),'show',COMMIT+':'+path])
prior=PART1/'research/anchored_neighbor_blend_v1'
sys.path.insert(0,str(prior));spec=importlib.util.spec_from_file_location('pysr_lda_control',prior/'study.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
plan={c['key']:c for c in json.loads(blob(BASE+'case_plan.json'))}
ids,truth,keys,pools,seeds,scores=[],[],[],[],[],[];origins={}
for c in benchmark.cases():
    key='strict_inductive__'+('grouped_' if c['pool']!='supplementary' else '')+c['key']
    old=plan[key];q=c['query'];assert old['query_ids']==q.ID.tolist()
    assert set(old['fit_ids'])==set(c['fit'].ID)
    path=BASE+'inputs/'+key+'.npz';data=blob(path);origins[path]=hashlib.sha256(data).hexdigest()
    with np.load(io.BytesIO(data)) as z:
        np.testing.assert_array_equal(z['ids'],q.ID);np.testing.assert_array_equal(z['truth'],q[benchmark.LABELS])
        anchor=z['anchored_scores'];cand=z['cand2_scores']
        center=lambda s:.7*s/(.7*s+.3*(1-s))
        a=center(anchor);lda=ref.lda_for(c)
        scores.append(np.stack([a,.75*a+.25*lda,center(.8*anchor+.2*cand)]))
    ids.extend(q.ID);truth.extend(q[benchmark.LABELS].to_numpy(int));keys.extend([c['key']]*len(q));pools.extend([c['pool']]*len(q));seeds.extend([c['seed']]*len(q))
np.savez_compressed(HERE/'controls.npz',modes=np.array(['anchored','current75_25','anchored80_cand2_20']),scores=np.concatenate(scores,axis=1),ids=np.array(ids),truth=np.array(truth),case=np.array(keys),pool=np.array(pools),seed=np.array(seeds))
result_path='part1/submissions/results/anchored80-cand2-inductive20-trainval235.json'
(HERE/'best_control_result.json').write_bytes(blob(result_path))
save(HERE/'control_provenance.json',dict(source_commit=COMMIT,source_sha256=origins,source_result_sha256=sha(HERE/'best_control_result.json'),retained_sha256=sha(HERE/'controls.npz'),cases=84,all_case_keys_fit_ids_query_ids_truth_matched=True,score_convention='All .5-centered; new best is .8 raw anchored + .2 Cand2 then centered from cutoff .3.',lda_source_helper='part1/research/anchored_neighbor_blend_v1/study.py::lda_for',lda_helper_sha256=sha(prior/'study.py')))
print('Frozen controls: all 84 exact case keys, fitting populations, query IDs and truth match.')
