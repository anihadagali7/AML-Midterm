"""Retrospective error complementarity review; no new fits or model selection."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

HERE=Path(__file__).resolve().parent
PART=HERE.parents[1]
MODELS=['cand1','cand2','cand4','cand5','lda_ensemble']
LABELS=['Safety','Funding','Efficacy','Recruitment','Other']


def main():
    cases=json.loads((HERE/'case_plan.json').read_text())
    records=[]; label_records=[]; inputs=[Path(__file__),HERE/'case_plan.json']
    for pool in ['train','pooled']:
        selected=[c for c in cases if c['access']=='strict_inductive' and c['kind']==pool]
        seeds=sorted({c['repeat'] for c in selected})
        refpath=PART/'experiments/current_leaders_grouped_v1/snapshots'/f'lda_ensemble_{pool}.npz'
        inputs.append(refpath)
        with np.load(refpath,allow_pickle=True) as z:
            ref={k:z[k].copy() for k in z.files}
        lookup={v:i for i,v in enumerate(ref['ids'])}
        for case in selected:
            path=HERE/'five_seed/results'/(case['key']+'.npz');inputs.append(path)
            with np.load(path) as z:
                y=z['truth'];a=z['anchored_predictions'];ae=(a==y).all(1)
                af=2*(a*y).sum(1)/(a.sum(1)+y.sum(1))
                ii=[lookup[v] for v in z['ids']]
                np.testing.assert_array_equal(y,ref['truth'][ii])
                old=ref['predictions'][seeds.index(case['repeat']),ii]
                for model in MODELS:
                    p=old if model=='lda_ensemble' else z[model+'_predictions']
                    f=2*(p*y).sum(1)/(p.sum(1)+y.sum(1));ex=(p==y).all(1)
                    assert abs(f.mean()-f1_score(y,p,average='samples'))<1e-12
                    for j,ident in enumerate(z['ids']):
                        records.append(dict(pool=pool,repeat=case['repeat'],model=model,ID=ident,
                             anchor_f1=af[j],candidate_f1=f[j],disagreement=int((p[j]!=a[j]).any()),
                             fixes_anchor_exact=int(ex[j] and not ae[j]),breaks_anchor_exact=int(ae[j] and not ex[j]),
                             fixes_both_anchor_and_lda=int(ex[j] and not ae[j] and (old[j]!=y[j]).any())))
                    for j,label in enumerate(LABELS):
                        label_records.append(dict(pool=pool,model=model,label=label,
                            corrected=int(((p[:,j]==y[:,j])&(a[:,j]!=y[:,j])).sum()),
                            broken=int(((p[:,j]!=y[:,j])&(a[:,j]==y[:,j])).sum())))
    rf=pd.DataFrame(records); summary=[]; repeat=[]
    for (pool,model),g in rf.groupby(['pool','model']):
        n=157 if pool=='train' else 235
        assert len(g)==8*n
        for seed,b in g.groupby('repeat'):
            assert b.ID.is_unique and len(b)==n
            repeat.append(dict(pool=pool,model=model,repeat=seed,rows=n,
                 candidate_f1=b.candidate_f1.mean(),anchor_f1=b.anchor_f1.mean(),
                 disagreement_fraction=b.disagreement.mean(),fixed_exact_rows=int(b.fixes_anchor_exact.sum()),
                 broken_exact_rows=int(b.breaks_anchor_exact.sum()),
                 fixed_both_anchor_and_lda=int(b.fixes_both_anchor_and_lda.sum()),
                 oracle_upper_bound=np.maximum(b.anchor_f1,b.candidate_f1).mean()))
        summary.append(dict(pool=pool,model=model,rows_per_repeat=n,repeats=8,
             candidate_f1=g.candidate_f1.mean(),anchor_f1=g.anchor_f1.mean(),
             disagreement_fraction=g.disagreement.mean(),fixed_exact_rows_per_repeat=g.fixes_anchor_exact.sum()/8,
             broken_exact_rows_per_repeat=g.breaks_anchor_exact.sum()/8,
             fixed_both_anchor_and_lda_per_repeat=g.fixes_both_anchor_and_lda.sum()/8,
             row_f1_correlation=g.anchor_f1.corr(g.candidate_f1),
             oracle_upper_bound=np.maximum(g.anchor_f1,g.candidate_f1).mean()))
    summary=pd.DataFrame(summary)
    summary.to_csv(HERE/'diversity_summary.csv',index=False)
    pd.DataFrame(repeat).to_csv(HERE/'diversity_by_repeat.csv',index=False)
    labels=pd.DataFrame(label_records).groupby(['pool','model','label'])[['corrected','broken']].sum()/8
    labels.to_csv(HERE/'diversity_labels.csv')
    manifest=dict(purpose='Choose research mechanisms to investigate, not promote a submission.',
        retrospective=True,model_fits=0,source='Five-inner-seed strict benchmark; exact retained eight-repeat LDA ensemble control.',
        method_reference='https://arxiv.org/abs/2010.13993 ; supports graph residual correction and prediction smoothing. The paper studies transductive node classification; it does not establish gains for our proposed inductive multilabel adaptation.',
        definitions={'fixed':'Candidate exactly matches every gold label while anchored does not.',
                     'broken':'Anchored exactly matches every gold label while candidate does not.',
                     'oracle':'Mean of the better row-F1 from anchored and candidate, choosing with the true labels. Unattainable diagnostic upper bound, not a fitted selector or forecast.',
                     'counts':'Averages over eight repeated evaluations of the same157/235 rows; not independent fresh examples.'},
        source_hashes={str(p.relative_to(PART.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        conclusion='Cand1/cand2 supply more complementary correct predictions than the existing LDA ensemble, but introduce more errors. Prioritize a bounded component-level graph-labeling/error-correction test against anchored; no full candidate is promoted.',
        history='Graphs, prior correction, expected-F1 decoding, metadata features and Other-specific residual models already exist in the archive. No prior anchored + Correct-and-Smooth OOF residual correction was found in the inspected code/registries. This is a specific untested combination, not a claim that graph learning is new.',
        next_tests_not_run=['Replace only anchored pseudo-label generation with the fixed C&S generator, holding classifier/decoder choices fixed.',
                            'Apply grouped inner-OOF residual propagation to anchored label scores, with multilabel-safe clipping and unchanged decoder. No outer-label-based gate or oracle selection.'])
    (HERE/'diversity_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    tab=summary[summary.pool=='pooled'][['model','candidate_f1','disagreement_fraction','fixed_exact_rows_per_repeat','broken_exact_rows_per_repeat','fixed_both_anchor_and_lda_per_repeat']]
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dylan candidates: research value beyond the mean score</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#18313b}table{border-collapse:collapse;font-size:14px}th,td{padding:10px;border-bottom:1px solid #d8e2e6;text-align:right}th:first-child,td:first-child{text-align:left}a{color:#0b7074}.scroll{overflow:auto}</style>
<h1>Which different mechanism is worth pursuing?</h1>
<p><strong>Cand1's Correct &amp; Smooth graph mechanism is worth a bounded component experiment against anchored.</strong> Cand2 is the closest standalone challenger, but its additional clustering/neighbor features are familiar territory. The graphs in cand1/cand2 supply complementary correct predictions that the overall score hides.</p>
<div class="scroll">'''+tab.to_html(index=False,border=0,float_format=lambda v:f'{v:.4f}')+'''</div>
<p>These are the five-seed models on all eight saved Pooled235 repeats. “Fixed” means a completely correct label set where anchored was not completely correct. “Broken” means the reverse. Counts are averages per 235-row repeat, not eight independent datasets.</p>
<p>Cand1 fixes 8 anchored mistakes per repeat, versus 4.5 for the existing LDA ensemble. About 4.75 of cand1's fixes are rows both anchored and LDA get wrong. But cand1 also breaks 14.125 of anchored's correct rows, so replacing or simply averaging entire models is not supported.</p>
<p>For diagnosis only, an all-knowing row selector between anchored and cand1 would score .93651, compared with anchored's .92137 and .92979 for the corresponding anchored/LDA selector. This uses the answers to choose the model; it is an unattainable upper bound without a successful selector, not a prediction of future performance.</p>
<p>The suggested change is in how the 1,014 masked examples and neighborhood structure are used: graph propagation can spread label information and out-of-fold error corrections through connected examples. Anchored currently uses GMM-assisted pseudo-labels and direct nearest-neighbor votes. This remains the same supplied feature information, so large gains are not guaranteed.</p>
<p>The <a href="https://arxiv.org/abs/2010.13993">original C&amp;S paper</a> supports the general residual-correction and smoothing mechanism. Its transductive node-classification results do not establish gains for our proposed inductive, multilabel adaptation.</p>
<ol><li>Replace only anchored's pseudo-label generator with the fixed C&amp;S generator. Hold its classifier blend and decision rule fixed.</li>
<li>Separately test C&amp;S-style correction of anchored's grouped inner-OOF errors. Adapt it to independent label scores; do not normalize multilabel scores as if the labels were mutually exclusive. Preserve outer fitting boundaries and the decoder.</li></ol>
<p>Both are proposed follow-ups, not measured improvements. Start with the saved three repeats in both pools and the existing paired advancement rule, then confirm promising changes on all eight. This review did not tune a gate, blend weight or threshold on outer predictions.</p>
<p>Prior research already covers Poisson/duplicate-aware graphs, Bayesian neighbor propagation, expected-F1 decisions, source-prior corrections, metadata representations and Other residual models. The old metric_graph “residual” parameter concatenates embedding coordinates; it is not C&amp;S correction of anchored's OOF errors. I found no prior evaluation of the two specific anchored/C&amp;S combinations above.</p>
<p>Cand4 is more different in its errors, but considerably weaker: 6 fixes versus 19 broken rows. Cand5 has only 3.25 fixes versus 14 broken rows and depends more on label-frequency adaptation. I would defer both as whole-model research branches.</p>
<p><a href="diversity_summary.csv">Summary</a> · <a href="diversity_by_repeat.csv">Every repeat</a> · <a href="diversity_labels.csv">Per-label corrections</a> · <a href="diversity_manifest.json">Definitions and provenance</a> · <a href="report.html">Full candidate benchmark</a></p></html>'''
    (HERE/'diversity_report.html').write_text(html)
    print(tab.to_string(index=False))


if __name__=='__main__':main()
