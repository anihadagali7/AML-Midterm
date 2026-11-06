"""Audit prior model/feature coverage and persistent errors before a new search."""
from pathlib import Path
import collections,json,re,hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,precision_score,recall_score
import run_ani_improvements as study

ROOT=study.r.ROOT;OUT=ROOT/'experiments/architecture_review_v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    study.guard();before=study.protected();OUT.mkdir(exist_ok=True)
    trial_path=ROOT/'experiments/trials.jsonl'
    trials=[json.loads(s) for s in trial_path.read_text().splitlines() if s.strip()]
    families=collections.defaultdict(list);blocks=collections.Counter()
    for row in trials:
        c=row.get('config',{});families[c.get('model','unknown')].append(row)
        blocks.update(c.get('features',[]))
    inventory=[]
    for name,rows in families.items():
        inventory.append(dict(family=name,logged_evaluations=len(rows),successful_evaluations=sum(x.get('status')=='ok' for x in rows),distinct_configs=len({json.dumps(x.get('config',{}),sort_keys=True) for x in rows})))
    inventory=pd.DataFrame(inventory).sort_values('logged_evaluations',ascending=False)
    inventory.to_csv(OUT/'model_inventory.csv',index=False)
    pd.DataFrame([dict(block=k,logged_uses=v) for k,v in blocks.most_common()]).to_csv(OUT/'feature_inventory.csv',index=False)
    train,val,full=study.r.frames();test=pd.read_csv(ROOT/'data/part1_test.csv');labels=study.e.LABELS
    gold=full[full[labels].ge(0).all(axis=1)].reset_index(drop=True);y=gold[labels].to_numpy(int)
    original=set(train.ID);source=np.where(gold.ID.isin(original),'original_train','original_validation')
    names=['Ani frozen control','Ani + tri/SVM 15% each'];scores={};provenance=[]
    for name in names:
        arrays=[]
        for stage in ['pooled3','pooled5']:
            folder=study.OUT/stage
            row=next(x for x in json.loads((folder/'index.json').read_text()) if x['candidate']==name)
            p=folder/(row['result_key']+'.npz');a=np.load(p,allow_pickle=True)
            assert a['ids'].tolist()==gold.ID.tolist();np.testing.assert_array_equal(a['truth'],y)
            arrays.extend(a['scores']);provenance.append(dict(candidate=name,path=str(p.relative_to(ROOT)),sha256=sha(p)))
        scores[name]=np.array(arrays)
    metrics=[];persistent=gold[['ID','start_year']].copy();persistent['source']=source
    persistent['actual_labels']=[' '.join(l for l,flag in zip(labels,row) if flag) for row in y]
    hard={}
    for name,ps in scores.items():
        pred=np.array([study.e.decisions(p,'t0.5') for p in ps]);hard[name]=pred
        sample=2*(pred*y[None,:,:]).sum(axis=2)/(pred.sum(axis=2)+y.sum(axis=1)[None,:])
        prefix='ani' if name==names[0] else 'three_way'
        persistent[prefix+'_mean_f1']=sample.mean(axis=0)
        persistent[prefix+'_exact_repeats']=(pred==y[None,:,:]).all(axis=2).sum(axis=0)
        for j,label in enumerate(labels):
            metrics.append(dict(candidate=name,label=label,gold_positives=int(y[:,j].sum()),
                mean_f1=float(np.mean([f1_score(y[:,j],h[:,j],zero_division=0) for h in pred])),
                mean_precision=float(np.mean([precision_score(y[:,j],h[:,j],zero_division=0) for h in pred])),
                mean_recall=float(np.mean([recall_score(y[:,j],h[:,j],zero_division=0) for h in pred])),
                mean_false_positives=float(((pred[:,:,j]==1)&(y[None,:,j]==0)).sum(axis=1).mean()),
                mean_false_negatives=float(((pred[:,:,j]==0)&(y[None,:,j]==1)).sum(axis=1).mean())))
    persistent['shared_persistent_error']=(persistent.ani_exact_repeats<=2)&(persistent.three_way_exact_repeats<=2)
    persistent.sort_values(['shared_persistent_error','three_way_mean_f1'],ascending=[False,True]).to_csv(OUT/'persistent_oof_errors.csv',index=False)
    metric_table=pd.DataFrame(metrics);metric_table.to_csv(OUT/'label_error_profile.csv',index=False)
    coverage=[]
    for cohort,frame in [('train157',train[train[labels].ge(0).all(axis=1)]),('masked1014',train[train[labels].eq(-100).all(axis=1)]),('validation78',val),('test312',test)]:
        for col in ['criteria','icdcodes','matched_drugs','drug_targets','drug_actions','drug_pathways','drug_smiles']:
            text=frame[col].fillna('').astype(str)
            coverage.append(dict(cohort=cohort,field=col,rows=len(frame),nonempty_rate=float(text.str.strip().ne('').mean()),median_characters=float(text.str.len().median())))
    pd.DataFrame(coverage).to_csv(OUT/'input_coverage.csv',index=False)
    groups=pd.util.hash_pandas_object(gold[study.e.EMB].round(10),index=False)
    inconsistent=[]
    for group,ix in pd.Series(np.arange(len(gold))).groupby(groups):
        ids=ix.to_numpy()
        if len(ids)>1 and len(np.unique(y[ids],axis=0))>1:
            inconsistent.append(dict(ids=gold.iloc[ids].ID.tolist(),labels=y[ids].tolist()))
    # Only count readily checkable structural properties here. Detailed clinical
    # bounds/negation extraction must be validated before model fitting.
    structure=[]
    catalog=pd.read_csv(ROOT/'data/pathway_catalog.csv');broad=set(catalog.loc[catalog.pathway_level==0,'reactome_id'])
    for cohort,frame in [('train',train),('validation',val),('test',test)]:
        criteria=frame.criteria.fillna('').astype(str)
        codes=frame.icdcodes.fillna('').map(lambda value:sorted(set(re.findall(r'\b[A-Z]\d{2}(?:\.[A-Z0-9]{1,4})?\b',str(value)))))
        paths=frame.drug_pathways.fillna('').map(lambda value:set(x.strip() for x in value.split(';') if x.strip()))
        structure.append(dict(cohort=cohort,rows=len(frame),explicit_inclusion_heading=int(criteria.str.contains(r'inclusion\s+criteria',case=False,regex=True).sum()),explicit_exclusion_heading=int(criteria.str.contains(r'exclusion\s+criteria',case=False,regex=True).sum()),rows_with_parseable_icd=int(codes.map(bool).sum()),median_icd_codes=float(codes.map(len).median()),rows_with_broad_pathway=int(paths.map(lambda x:bool(x&broad)).sum())))
    next_steps=[
        dict(direction='Structured eligibility and disease representation',
            evidence='Prior features treated criteria as whole-document word/character TF-IDF and ICD codes as generic text. ICD strings contain nested lists and dotted codes; generic tokenization separates decimal suffixes. No explicit inclusion/exclusion feature extractor was found in the phase scripts.',
            work='Build separate inclusion/exclusion text views, validated eligibility-burden features, and correctly parsed ICD code sets with parent categories. Keep unsupported clinical-bound extraction out until reviewed.',
            evaluation='Compare each view alone and added to the fixed reason-embedding model, using fold-local vocabulary/scaling and the common grouped CV. Reject a block without repeatable gain.'),
        dict(direction='Separate models for reason embeddings and structured metadata',
            evidence='Trees were tested mainly on raw/PCA reason coordinates; tree students and fixed-weight metadata co-training were also tried. The PDF says reason embeddings are randomly rotated, so coordinate splits have no inherent semantic meaning. Drug metadata are missing for a substantial fraction of trials.',
            work='Keep a rotation-tolerant embedding classifier; train regularized linear and shallow tree models on the new structured metadata. Evaluate metadata signal independently before any combination. If useful, test a strongly regularized gate learned from inner-fold predictions and explicit metadata coverage, rather than another global weight sweep.',
            evaluation='Use exact matched embedding-only, metadata-only and combined controls. Fit any gate only within outer-training data. This direction is not a claim that metadata will help.'),
        dict(direction='Joint multilabel prediction from representations',
            evidence='Powerset models, one classifier-chain recipe, and score-level joint decoders already exist. Repeating their names would not constitute a new architecture.',
            work='Only reopen joint modeling if the persistent-error review shows recurring co-occurrence mistakes. Compare a regularized joint label-set estimator on the new representations with independent heads, retaining an escape route for rare/unseen combinations. Prior semisupervised Gaussian mixtures and sparse prototypes should be controls, not advertised as untried methods.',
            evaluation='Use the unchanged sample-F1 target, per-label diagnostics, and nested decoder selection. Advance based on both common CV pools before validation.')]
    summary=dict(logged_evaluations=len(trials),families=len(families),distinct_configs=len({json.dumps(x.get('config',{}),sort_keys=True) for x in trials}),
        logged_count_note='These are evaluation records, including repeats, not independent datasets. Historical scores used different split/selection policies and are not directly ranked here.',
        common_persistent_error_rows=int(persistent.shared_persistent_error.sum()),gold_embedding_groups=int(groups.nunique()),conflicting_gold_embedding_groups=inconsistent,
        structural_input_audit=structure,next_steps=next_steps,new_model_fits=0,source_provenance=provenance,
        trial_log_sha256=sha(trial_path),script_sha256=sha(Path(__file__)),protected_sha256=before,
        constraints='No deep learning, external trial data, label edits or ID predictors. Read-only review; historical benchmark and model artifacts unchanged.')
    study.r.save(OUT/'summary.json',summary)
    body=f'<p>Reviewed {len(trials)} logged evaluation records across {len(families)} model-family identifiers. Records include repeats and incompatible historical selection policies; they are an inventory, not an independent leaderboard. The reset should change representations and test specific failure hypotheses, rather than re-run familiar classifier names.</p>'
    body+='<h2>Persistent error profile on the common pooled CV</h2>'+metric_table.to_html(index=False,float_format=lambda x:f'{x:.4f}',border=0)
    body+=f'<p>{int(persistent.shared_persistent_error.sum())} of235 gold trials fail exact label-set prediction on at least six of eight repeats for both Ani and the three-way ensemble. See <a href="persistent_oof_errors.csv">the per-trial audit</a>. These are repeated observations of the same trials, not independent failures.</p>'
    body+='<h2>Concrete directions for the next architecture/feature round</h2>'
    for item in next_steps:
        body+='<h3>'+item['direction']+'</h3><p>'+item['evidence']+'</p><p>'+item['work']+'</p><p>'+item['evaluation']+'</p>'
    body+='<h2>Structural input audit</h2>'+pd.DataFrame(structure).to_html(index=False,border=0)
    body+='<p>The assignment says drug actions and targets are pooled across drugs, so action-to-target pairs must not be invented. Pathway depths alone do not supply parent edges; use supplied broad-category membership. Existing criteria text does not contain the withheld stopping-reason text. Structured parsing is a hypothesis for adding information, not guaranteed extra signal.</p>'
    body+='<h2>Existing model-family coverage</h2>'+inventory.to_html(index=False,border=0)
    body+='<p><a href="feature_inventory.csv">Previously used feature blocks</a> · <a href="input_coverage.csv">Missingness by cohort</a> · <a href="summary.json">Review and provenance</a> · <a href="../three_way_transductive_v1/report.html">Separate transductive experiment</a></p>'
    (OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Architecture and feature review</title><style>body{font:16px system-ui;max-width:1500px;margin:40px auto;padding:0 24px}p{line-height:1.6;max-width:1150px}table{border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}</style><h1>Architecture and feature reset: evidence before more tuning</h1>'+body)
    assert before==study.protected();study.guard()
    print(json.dumps({k:summary[k] for k in ['logged_evaluations','families','distinct_configs','common_persistent_error_rows','gold_embedding_groups','conflicting_gold_embedding_groups','structural_input_audit']},indent=2))
    print(metric_table[metric_table.candidate==names[1]].to_string(index=False))


if __name__=='__main__':main()
