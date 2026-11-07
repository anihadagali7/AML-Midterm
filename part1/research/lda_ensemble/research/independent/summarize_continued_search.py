"""Build a single review surface for the two continued-search studies."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/continued_search_v1'
STUDIES=['ani_combination_v1','ani_bagging_v1']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    initial=[];confirmed=[];final=[];provenance=[];recipes=[]
    for study in STUDIES:
        folder=ROOT/'experiments'/study
        for file in ['verification.json','boundary_audit.json']:
            assert json.loads((folder/file).read_text())['passed']
        screen=pd.read_csv(folder/'screen_comparison.csv');screen['study']=study
        initial.append(screen[screen.kind!='control'])
        extension=pd.read_csv(folder/'confirmation.csv');extension['study']=study
        confirmed.append(extension)
        final.append(pd.read_csv(folder/'final_comparison.csv'))
        registry=json.loads((folder/'registry.json').read_text())
        names=set(extension.loc[extension.qualifies,'candidate'])
        recipes.extend([dict(c,study=study,status='Passed frozen paired-CV rule; no Kaggle result') for c in registry if c['name'] in names])
        provenance.append(dict(study=study,protocol_sha256=sha(folder/'protocol.json'),
            registry_sha256=sha(folder/'registry.json'),verification_sha256=sha(folder/'verification.json'),
            boundary_audit_sha256=sha(folder/'boundary_audit.json')))
    initial=pd.concat(initial,ignore_index=True)
    confirmed=pd.concat(confirmed,ignore_index=True).drop_duplicates('candidate')
    final=pd.concat(final,ignore_index=True).drop_duplicates('candidate')
    final=final.sort_values('minimum_frontier_gain',ascending=False)
    assert len(initial)==12
    OUT.mkdir(exist_ok=True)
    initial.to_csv(OUT/'all_initial_trials.csv',index=False)
    confirmed.to_csv(OUT/'all_eight_repeat_results.csv',index=False)
    final.to_csv(OUT/'final_comparison.csv',index=False)
    old=json.loads((ROOT/'experiments/ani_improvement_v1/registry.json').read_text())
    controls=[c for c in old if c['name'] in ['Ani + tri 25%','Ani + tri/SVM 15% each']]
    assessment=[]
    for c in recipes:
        row=final[final.candidate==c['name']].iloc[0]
        parent=final[final.candidate==c['parent']].iloc[0]
        assessment.append(dict(candidate=c['name'],parent=c['parent'],
            validation_delta=float(row.validation-parent.validation),
            chronological_delta=float(row.chronological-parent.chronological),
            interpretation='CV-qualified challenger; retain prior leaders. Both supplementary scores decreased.'
                if row.validation<parent.validation and row.chronological<parent.chronological
                else 'CV-qualified challenger; assess supplementary evidence separately.'))
    data=dict(new_recipes_tested=12,cv_rule_qualifiers=[c['name'] for c in recipes],
        new_candidates=recipes,retained_previous_leaders=controls,
        supplementary_assessment=assessment,previous_leaders_replaced=False,
        final_comparison=final.to_dict('records'),provenance=provenance,
        submissions_created=0,full235_submission_refits=0,
        note='Keep prior artifacts unchanged. Local CV improvements do not establish improved Kaggle performance. Eight-repeat CV reuses development labels and is not an independent test.')
    result_path=ROOT/'predictions/three_way_trainval235/kaggle_result.json'
    kaggle=None
    if result_path.exists():
        kaggle=json.loads(result_path.read_text())
        assert sha(ROOT/kaggle['submission_file'])==kaggle['submission_sha256']
        data['subsequent_submission_result']=kaggle
        data['subsequent_submission_result_sha256']=sha(result_path)
        data['scope']='The two development studies created no submissions. The subsequent three-way 235-label refit and reported Kaggle result are recorded separately.'
    refit_path=ROOT/'experiments/refit_diagnostic_v1/summary.json'
    refit=None
    if refit_path.exists():
        refit=json.loads(refit_path.read_text());assert refit['verification_passed']
        data['subsequent_refit_diagnostic']=dict(path=str(refit_path.relative_to(ROOT)),sha256=sha(refit_path),comparison=refit['comparison'],test_sensitivity=refit['test_sensitivity'])
    (OUT/'summary.json').write_text(json.dumps(data,indent=2)+'\n')
    cols=['candidate','train_mean','pooled_mean','validation','chronological']
    extended=['candidate','train_mean','train_sd','pooled_mean','pooled_sd','train_parent_delta','pooled_parent_delta','train_parent_wins','pooled_parent_wins','qualifies']
    intro=f'<p>Tested 12 additional recipes in two frozen studies. {len(recipes)} passed the rule requiring higher eight-repeat mean F1 and wins on at least five of eight repeats in both CV pools versus the unchanged parent. The preceding eight single-change combinations produced no qualifying improvement; the second study tested averaging Ani fits over grouped gold subsets.</p>'
    for item in assessment:
        intro+=f'<p><strong>{item["candidate"]}: {item["interpretation"]}</strong> Relative to its parent, validation changed by {item["validation_delta"]:+.5f} and chronological F1 by {item["chronological_delta"]:+.5f}. Higher CV alone does not establish an overall upgrade.</p>'
    body='<!doctype html><meta charset="utf-8"><title>Continued Part 1 model search</title><style>body{font:16px system-ui;max-width:1700px;margin:36px auto;padding:0 24px;color:#172331}table{border-collapse:collapse;width:100%;margin:24px 0}th,td{padding:9px;border-bottom:1px solid #d9e0e7;text-align:left}th{background:#eef3f8}p{max-width:1150px;line-height:1.55}.scroll{overflow-x:auto}</style><h1>Continued Part 1 model search</h1>'+intro
    body+='<p>Primary: train157 grouped five-fold CV. Supplemental: pooled235 grouped five-fold CV. Both use the same eight saved repeats across candidates. F1 is sample-wise; each repeat is scored on its complete out-of-fold predictions. Original validation uses train157 only. Chronology pools 98 later trials across the same three frozen earlier-to-later windows. Unchanged supplementary control scores are reused with provenance.</p>'
    body+='<h2>Qualified candidates and retained controls</h2><div class="scroll">'+final[cols].to_html(index=False,float_format=lambda x:f'{x:.5f}',border=0)+'</div>'
    body+='<p>Ani here is the inductive adaptation under the common fitting policy. Its local scores are not a reproduction of the original transductive Kaggle submission. The two development studies above created no submission or final train-plus-validation fit.</p>'
    if kaggle:
        body+='<h2>Subsequent Kaggle evidence</h2><p>The frozen 70% Ani / 15% tri-training / 15% SVM ensemble was subsequently refitted on all 235 gold labels and submitted manually. The user reported <strong>'+f'{kaggle["kaggle_f1"]:.5f}'+'</strong>, compared with Ani’s <strong>0.89423</strong> and standalone tri-training’s <strong>0.88012</strong>. The three-way score is 0.00022 below Ani and 0.01389 above standalone tri-training. Ani remains the highest reported submission.</p>'
        body+='<p>The local CV advantage did not yield a higher Kaggle score than the original Ani submission. This is a comparison of complete pipelines: original Ani used transductive fitting, whereas our ensemble excluded test inputs from fitting. The aggregate score cannot identify which component helped or which examples failed. There are two submissions remaining from the user’s stated budget. The 75/25 and subset-averaging candidates remain unsubmitted.</p><p><a href="../../predictions/three_way_trainval235/kaggle_result.json">User-reported result and exact submission hash</a> · <a href="../../predictions/three_way_trainval235/manifest.json">Preparation manifest and training policy</a></p>'
    else:
        body+='<p>The reported Kaggle results remain 0.89423 for Ani and 0.88012 for standalone tri-training. No new ensemble Kaggle result has been recorded.</p>'
    if refit:
        row=next(x for x in refit['comparison'] if x['candidate']=='Ani + tri/SVM 15% each' and x['scope']=='all235')
        sensitivity=refit['test_sensitivity']
        body+='<h2>Subsequent check of adding validation labels</h2><p>On identical held-out groups, the frozen three-way recipe scored '+f'{row["reduced_mean"]:.5f}'+' using only eligible original-training gold and '+f'{row["expanded_mean"]:.5f}'+' when eligible validation-source gold was added. The expanded arm won '+str(row['expanded_wins'])+' of eight repeats. Gains were concentrated in held-out validation-source rows; held-out original-training rows declined slightly. This does not show overall refit harm on these development data.</p>'
        body+='<p>The actual157-label versus235-label fits changed '+str(sensitivity['changed_label_sets'])+' of312 test label sets. That measures prediction sensitivity, not which fit is more accurate. The matched CV diagnostic uses roughly125 versus188 fitting labels inside each fold, and cannot determine the unavailable Kaggle score of the157-label model. No new submission was exported. <a href="../refit_diagnostic_v1/report.html">Full paired diagnostic and verification</a>.</p>'
    body+='<h2>All eight-repeat results</h2><p>Parent deltas and win counts compare each changed recipe with its own unmodified ensemble. Repeat SD is descriptive; the repeats share trials and are not independent datasets. A promising initial result that failed the longer check remains visible here.</p><div class="scroll">'+confirmed[extended].to_html(index=False,float_format=lambda x:f'{x:.5f}',border=0)+'</div>'
    body+='<h2>All 12 initial trials: three repeats only</h2><p>These values are a separate screening stage and must not be compared directly with the eight-repeat means above. All settings and advancement rules were frozen before each study.</p><div class="scroll">'+initial[['candidate','parent','train_mean','pooled_mean','train_parent_delta','pooled_parent_delta','qualifies']].to_html(index=False,float_format=lambda x:f'{x:.5f}',border=0)+'</div>'
    body+='<h2>Audit and reproducibility</h2><p>Both studies passed held-out-label/feature mutation and query-batch checks, including matching masked duplicates. The subset study also checks omitted gold-group exclusion for every member. Saved scores, decisions, IDs and truth were rescored; all unchanged control CV scores reproduce prior outputs exactly. Protected raw data and earlier submissions remain unchanged.</p><p>Repeated development on these folds can still favor the selected recipes. Validation and chronological checks are supplementary and were consulted only for a CV-qualified frozen shortlist. These are local improvements, not guarantees about later Kaggle trials.</p><p><a href="../ani_combination_v1/report.html">Eight single-change combinations</a> · <a href="../ani_bagging_v1/report.html">Four subset-averaging recipes</a> · <a href="summary.json">Recipes, results and provenance</a></p>'
    (OUT/'report.html').write_text(body)
    print(final[cols].to_string(index=False))
    print('Qualified additions:',data['cv_rule_qualifiers'])


if __name__=='__main__':main()
