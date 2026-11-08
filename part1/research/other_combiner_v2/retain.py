"""Compact checked results; discard unsuccessful inner caches after rescoring."""
from pathlib import Path
import importlib.util
import shutil
import numpy as np
import run as study


def main():
    here=study.HERE;output=here/'results';summary=output/'summary'
    receipt=study.read(summary/'verification.json');assert receipt['passed'] and receipt['outer_cases']==84
    keep=['parent','pseudo_lda_quarter','combiner_other_only'];indices=[study.NAMES.index(n) for n in keep]
    packets=[];meta=[];coefficients={};equivalent=True
    for c in study.cases():
        p=output/'predictions'/(c['key']+'.npz');z=study.loadnp(p);a=study.read(p.with_suffix('.json'))
        assert study.sha(p)==a['sha256']
        equivalent&=np.array_equal(z['predictions'][study.NAMES.index('pseudo_lda_quarter')],
                                   z['predictions'][study.NAMES.index('pseudo_lda_confidence')])
        packets.append(z);pool=c['pool'] if c['pool']!='supplementary' else c['key']
        meta.append(dict(pool=np.full(len(z['ids']),pool),seed=np.full(len(z['ids']),c['seed']),case=np.full(len(z['ids']),c['key'])))
        for name,v in a['coefficients'].items():
            b=v['beta'] if isinstance(v,dict) else v
            coefficients.setdefault((pool,name),[]).append(np.asarray(b))
    arrays={k:np.concatenate([z[k] for z in packets]) for k in ['ids','truth']}
    arrays.update({k:np.concatenate([m[k] for m in meta]) for k in ['pool','seed','case']})
    arrays.update(names=np.array(keep),scores=np.concatenate([z['scores'][indices] for z in packets],axis=1),
                  predictions=np.concatenate([z['predictions'][indices] for z in packets],axis=1))
    np.savez_compressed(output/'retained_comparison.npz',**arrays)
    study.save(summary/'coefficient_summary.json',[dict(pool=k[0],recipe=k[1],outer_cases=len(v),
        mean=np.mean(v,axis=0).tolist(),population_sd=np.std(v,axis=0).tolist()) for k,v in coefficients.items()])
    study.save(summary/'pseudo_variants_equivalence.json',dict(all_84_hard_prediction_arrays_identical=bool(equivalent),
        note='Raw scores differ; no equality of fitted models is claimed.'))
    assert equivalent
    # Per-case trained head coefficients and inner datasets are unsuccessful fit caches.
    (summary/'coefficients.json').unlink()
    files=list(summary.glob('*'))+[output/'retained_comparison.npz']
    manifest=dict(kept_comparisons=keep,
        reasons={'parent':'Scored .89935 reference.',
                 'pseudo_lda_quarter':'Useful matched diagnosis: recovers more Recruitment+Other but regresses chronology.',
                 'combiner_other_only':'Useful matched diagnosis: tied CV averages with a small chronology improvement.'},
        full_checks_before_cleanup=receipt,
        discard='84 per-case npz and 84 fit-audit JSON files, containing unsuccessful inner predictions/features and per-case fitted coefficients; per-case coefficient aggregate removed. Source, compact metrics for all seven variants, negative results and reproduction code retained.',
        retained_sha256={str(p.relative_to(here)):study.sha(p) for p in files})
    study.save(here/'retention.json',manifest)
    spec=importlib.util.spec_from_file_location('this_study_retained_check',here/'verify_retained.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.main()
    shutil.rmtree(output/'predictions')
    print('Removed checked unsuccessful fold caches; retained three useful matched comparisons.')


if __name__=='__main__':main()
