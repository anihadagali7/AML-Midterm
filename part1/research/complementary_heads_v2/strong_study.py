"""Reuse v1's frozen nested runner; version the head and combination changes."""
import os
for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[variable]='1'
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
import numpy as np
from scipy.special import expit,logit

HERE=Path(__file__).resolve().parent
V1=HERE.parent/'complementary_heads_v1'
sys.path.insert(0,str(V1))
import head_study as study
from complement import centered_logit
from strong_kernel import StrongKernel

original_fit_pair=study.fit_pair
study.HERE=HERE
study.ComplementaryKernel=StrongKernel


def fit_pair(fit,query_x):
    scores,audit,anchor=original_fit_pair(fit,query_x)
    for name in study.STRENGTHS:
        raw_second=expit(logit(np.clip(scores['head_'+name],1e-12,1-1e-12))+logit(.3))
        scores['blend_'+name]=expit(centered_logit(.8*anchor+.2*raw_second))
    return scores,audit,anchor


def reference(case):
    z=study.loadnp(V1/'parent_inputs.npz');keep=z['case']==case['reference_key']
    np.testing.assert_array_equal(z['ids'][keep],case['query'].ID)
    np.testing.assert_array_equal(z['truth'][keep],case['query'][study.LABELS])
    return z['anchored'][keep],z['cand2'][keep]


def sources():
    old=study.read(V1/'protocol.json')['sources']
    # The original implementation and evidence remain protected, byte for byte.
    result={name:study.sha(study.ROOT/name) for name in old}
    assert result==old
    extra=[HERE/'strong_kernel.py',HERE/'strong_study.py',HERE/'check_strong.py',V1/'protocol.json']
    result.update({str(p.relative_to(study.ROOT)):study.sha(p) for p in extra})
    return result


def register():
    path=HERE/'protocol.json';assert not path.exists()
    protocol=study.read(V1/'protocol.json')
    protocol.update(version='complementary-heads-v2',registered_at=datetime.now(timezone.utc).isoformat(),
        preceding_result='v1 completed all84cases. Ordinary standalone Train.712049/Pooled.719858/Chronology.408163. No combined candidate passed. This adaptive followup strengthens the weak control before using it to judge the diversity hypothesis; it is not independent confirmation.',
        hypothesis='The v1 second head was too weak before adding diversity. Lower kernel weight regularization by100x and align its decision threshold to.3, then repeat matched ordinary and two diversity strengths.',
        head_changes='Identical supervised BCE and error-correlation objective, map/scaler/landmarks and seed. Nonintercept L2 coefficient .00005 instead of .005, intercept remains1e-6. Still lambda0/.1/.3, same optimizer and ordinary warm start. Raw head probabilities decoded at.3+argmax. Corresponding centered logits replace the old zero-centered second-head logits in fusion.',
        head_loss='Same exact v1 loss_gradient with explicit ridge=.00005. Correlation uses sigmoid(centered anchored logits)-y versus raw kernel sigmoid(logits)-y; raw kernel probabilities are used in the training loss, regardless of its .3 final decision cutoff. No gradient reversal of label loss.',
        fixed_blends='Raw .8*anchored_score+.2*kernel_probability, decoded at.3+argmax. Internally mapped monotonically to .5 for the shared evaluator. Unlike v1, this uses the same raw arithmetic score space as the current Cand2 blend.',
        learned_fusion='Unchanged nested cross-label offset logistic, except kernel logits now subtract logit(.3) before clipping/scaling. Calibration-only control must reproduce v1 exactly.',
        followup_scope='One stronger fixed kernel recipe, same three diversity strengths, six combined candidates. No further parameter grid or outcome-selected cutoff. Full84cases again; all negatives retained.',
        reference='Same unchanged anchor and scored .89935 controls. Fresh anchor must match retained outputs within1e-7. Every nested fit is fresh; only current Cand2 comparison is cached.',
        sources=sources())
    study.save(path,protocol);print('Registered stronger head v2; six combined candidates, all84cases.',flush=True)


study.fit_pair=fit_pair
study.reference=reference
study.sources=sources


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['register','run','collect'])
    parser.add_argument('--output',type=Path,default=HERE/'results');parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.command=='register':register()
    elif args.command=='run':study.run(args.output,args.workers)
    else:study.collect(args.output)
