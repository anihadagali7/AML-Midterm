"""Evaluate fixed-weight ensembles on both local evaluation populations."""
import json
import numpy as np
import pandas as pd
from experiments import OUT, load_data, split_data, metrics

FAMILIES=['linear','refined','extra','boosted','robust','yearcap']


def collect():
    pred={s:{} for s in ['validation','temporal']}
    tests={s:{} for s in pred}
    configs={}
    for family in FAMILIES:
        c=OUT/f'{family}_configs.json'
        if c.exists():configs.update(json.loads(c.read_text()))
        for split in pred:
            p=OUT/f'{family}_{split}_predictions.npz'
            t=OUT/f'{family}_{split}_test_predictions.npz'
            if p.exists() and t.exists():
                with np.load(p) as f:pred[split].update({k:f[k] for k in f.files})
                with np.load(t) as f:tests[split].update({k:f[k] for k in f.files})
    return pred,tests,configs


def weighted(bank,weights,phases=None):
    if any(isinstance(v,dict) for v in weights.values()):
        assert phases is not None
        phases=np.asarray(phases)
        for p in ['I','II','III']:assert abs(sum(w[p] for w in weights.values())-1)<1e-9
        return sum(np.array([weight[p] for p in phases])*bank[name] for name,weight in weights.items())
    assert abs(sum(weights.values())-1)<1e-9
    return sum(weight*bank[name] for name,weight in weights.items())


def combine(a,b,fraction):
    keys=set(a)|set(b)
    return {k:(1-fraction)*a.get(k,0)+fraction*b.get(k,0) for k in keys if (1-fraction)*a.get(k,0)+fraction*b.get(k,0)>0}


def average(names):return {n:1/len(names) for n in names}


def run():
    data=load_data();pred,tests,configs=collect()
    common=set(pred['validation'])&set(pred['temporal'])
    groups={
      'linear_wordwide':{'text_wordwide_C2':1.},
      'linear_baseline':{'lr_criteria_none_C2_phase0':1.},
      'linear_clinical_baseline':{'lr_clinical_none_C0.5_phase0':1.},
      'linear_yearcap':{'lr_criteria_yearcap_C2_phase0':1.},
      'linear_text':average(['text_wordwide_C2','recency_criteria_C2_half6']),
      'linear_clinical':{'recency_clinical_C0.5_half3':1.},
      'catboost_seeds':average(['robust_catboost_seed43_n1500','robust_catboost_seed44_n1500']),
      'catboost_seeds_early':average(['robust_catboost_seed43_n800','robust_catboost_seed44_n800']),
      'catboost_standard':average(['catboost_d4_n1000_none','catboost_d6_n1000_none']),
      'catboost_early':average(['catboost_d4_n600_none','catboost_d6_n600_none']),
      'catboost_deep':average(['catboost_d6_n1000_none','catboost_d8_n1000_none']),
      'catboost_recency':{'robust_catboost_recency6_n1500':1.},
      'catboost_recency_early':{'robust_catboost_recency6_n800':1.},
      'catboost_phasebalanced':{'robust_catboost_phasebalanced_n1500':1.},
      'catboost_year':average(['catboost_d4_n1000_year','catboost_d6_n1000_year']),
      'catboost_year_early':average(['catboost_d4_n600_year','catboost_d6_n600_year']),
      'histgb':{'histgb_leaves31_none':1.},
      'extratrees':{'extratrees_leaf5_none':1.},
      'extratrees_year':{'extratrees_leaf5_year':1.},
      'lightgbm':{'lightgbm_leaves7_n1000_none':1.},
    }
    groups={k:v for k,v in groups.items() if set(v)<=common}
    candidates={k:v for k,v in groups.items()}
    if 'linear_text' in groups and 'linear_clinical' in groups:
        for w in [.25,.5]:
            groups[f'linear_mix_clinical{w:g}']=combine(groups['linear_text'],groups['linear_clinical'],w)
    linear={k:v for k,v in groups.items() if k.startswith('linear')}
    trees={k:v for k,v in groups.items() if not k.startswith('linear')}
    if 'catboost_seeds' in groups and 'catboost_standard' in groups:
        trees['catboost_bagged']=combine(groups['catboost_standard'],groups['catboost_seeds'],.5)
    if 'catboost_seeds' in groups and 'catboost_recency' in groups:
        trees['catboost_recentmix']=combine(groups['catboost_seeds'],groups['catboost_recency'],.5)
    for ln,lw in linear.items():
        for tn,tw in trees.items():
            for alpha in [.25,.5,.75]:
                candidates[f'{ln}__{tn}__tree{alpha:g}']=combine(lw,tw,alpha)
    for ln in ['linear_text','linear_mix_clinical0.25','linear_mix_clinical0.5']:
        if ln not in linear:continue
        for tn in ['catboost_standard','catboost_deep','catboost_seeds_early','catboost_year','histgb']:
            if tn not in trees:continue
            for alphas in [(.25,.5,.75),(.25,.25,.75)]:
                lw,tw=linear[ln],trees[tn]
                weights={k:{p:(1-a)*lw.get(k,0)+a*tw.get(k,0) for p,a in zip(['I','II','III'],alphas)} for k in set(lw)|set(tw)}
                candidates[f'{ln}__{tn}__phases'+','.join(f'{a:g}' for a in alphas)]=weights
    rows=[]
    for name,w in candidates.items():
        row={'name':name,'n_components':len(w)}
        for split in pred:
            evaluation=split_data(data,split)[1]
            m=metrics(evaluation,weighted(pred[split],w,evaluation.phase))
            for key,value in m.items():row[split+'_'+key]=value
        row['joint_logloss']=.5*(row['validation_phase_mean_logloss']+row['temporal_phase_mean_logloss'])
        rows.append(row)
    scores=pd.DataFrame(rows).sort_values('joint_logloss')
    scores.to_csv(OUT/'ensemble_scores.csv',index=False)
    (OUT/'ensemble_configs.json').write_text(json.dumps(candidates,indent=2))
    component_rows=[]
    for name in common:
        row={'name':name,**configs.get(name,{})}
        for split in pred:
            m=metrics(split_data(data,split)[1],pred[split][name])
            row[split+'_phase_mean_logloss']=m['phase_mean_logloss']
        row['joint_logloss']=.5*(row['validation_phase_mean_logloss']+row['temporal_phase_mean_logloss'])
        component_rows.append(row)
    pd.DataFrame(component_rows).sort_values('joint_logloss').to_csv(OUT/'all_component_scores.csv',index=False)
    print(scores[['name','validation_phase_mean_logloss','temporal_phase_mean_logloss','joint_logloss']].head(30).to_string(index=False))


def choose_four():
    scores=pd.read_csv(OUT/'ensemble_scores.csv')
    no_cap=~scores.name.str.startswith('linear_yearcap')
    filters={
      '01_balanced_ensemble':no_cap & scores.name.str.endswith('__tree0.5'),
      '02_text_heavy':no_cap & scores.name.str.endswith('__tree0.25'),
      '03_phase_specialized':scores.name.str.contains('__phases'),
      '04_capped_year':scores.name.str.startswith('linear_yearcap__'),
    }
    selected={name:scores[mask].iloc[0]['name'] for name,mask in filters.items()}
    assert len(set(selected.values()))==4
    (OUT/'selected_ensembles.json').write_text(json.dumps(selected,indent=2))
    (OUT/'selection_policy.json').write_text(json.dumps({
        'ranking':'Mean of equal-phase validation and temporal log loss',
        'directions':{
            '01_balanced_ensemble':'Best 50% text / 50% tree mixture without linear year effects',
            '02_text_heavy':'Best 75% text / 25% tree mixture without linear year effects',
            '03_phase_specialized':'Best mixture allowing different fixed weights in the three phases',
            '04_capped_year':'Best mixture using a linear year effect capped at the latest training year'},
        'interpretation':'Distinct strong candidates for four manual submissions, rather than the four nearly identical minimum-loss rows.'},indent=2))
    print('Selected:',json.dumps(selected,indent=2))


if __name__=='__main__':
    import sys
    run()
    if '--select' in sys.argv:choose_four()
