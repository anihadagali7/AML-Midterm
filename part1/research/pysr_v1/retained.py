"""Compact sufficient-statistic audit; raw candidate predictions can be discarded."""
import argparse
import json
import shutil
import numpy as np
import pandas as pd
from common import HERE,ROOT,benchmark,LABELS,sha,save
import study


def collect():
    study.report(HERE/'run',write=False)
    rows=[]
    for c in benchmark.cases():
        with np.load(HERE/'run/predictions'/(c['key']+'.npz')) as z:
            y=z['truth']
            for i,mode in enumerate(study.MODES):
                p=z['predictions'][i]
                row=dict(mode=mode,case=c['key'],pool=c['pool'],seed=c['seed'],rows=len(y),sample_f1_sum=float((2*(y*p).sum(1)/(y.sum(1)+p.sum(1))).sum()))
                for j,label in enumerate(LABELS):
                    row.update({f'tp_{label}':int((y[:,j]*p[:,j]).sum()),f'fp_{label}':int(((1-y[:,j])*p[:,j]).sum()),f'fn_{label}':int((y[:,j]*(1-p[:,j])).sum())})
                rows.append(row)
    assert len(rows)==84*len(study.MODES)
    dest=HERE/'results';assert not dest.exists();shutil.copytree(HERE/'run/results',dest)
    pd.DataFrame(rows).to_csv(dest/'sufficient_statistics.csv',index=False)
    verify()


def metric(frame):
    result=dict(sample_f1=float(frame.sample_f1_sum.sum()/frame.rows.sum()))
    fs=[]
    for label in LABELS:
        tp,fp,fn=[float(frame[f'{s}_{label}'].sum()) for s in ['tp','fp','fn']]
        pr=tp/(tp+fp) if tp+fp else 0.;re=tp/(tp+fn) if tp+fn else 0.;f=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
        result.update({f'precision_{label}':pr,f'recall_{label}':re,f'f1_{label}':f,f'support_{label}':tp+fn});fs.append(f)
    result['macro_f1']=float(np.mean(fs));return result


def verify():
    study.guard();out=HERE/'results';stats=pd.read_csv(out/'sufficient_statistics.csv');count=0
    assert len(stats)==504 and not stats.duplicated(['mode','case']).any()
    cases=benchmark.cases();assert set(stats['case'])=={c['key'] for c in cases}
    for c in cases:
        rows=stats[stats['case']==c['key']];assert len(rows)==6 and (rows.rows==len(c['query'])).all()
    for filename in ['repeat_scores','supplementary_scores']:
        table=pd.read_csv(out/(filename+'.csv'))
        for row in table.to_dict('records'):
            frame=stats[stats['mode']==row['mode']]
            if 'pool' in row:frame=frame[(frame.pool==row['pool'])&(frame.seed==row['seed'])]
            elif row['case']=='chronology98':frame=frame[(frame.pool=='supplementary')&(frame['case']!='validation')]
            else:frame=frame[frame['case']==row['case']]
            for name,value in metric(frame).items():assert abs(value-row[name])<1e-12,(filename,row['mode'],name);count+=1
    repeats=pd.read_csv(out/'repeat_scores.csv')
    for row in pd.read_csv(out/'cv_summary.csv').to_dict('records'):
        part=repeats[(repeats.pool==row['pool'])&(repeats['mode']==row['mode'])];assert len(part)==8
        for key,value in [('mean_sample_f1',part.sample_f1.mean()),('population_sd',part.sample_f1.std(ddof=0)),('mean_macro_f1',part.macro_f1.mean())]:assert abs(row[key]-value)<1e-12
    for row in pd.read_csv(out/'paired_summary.csv').to_dict('records'):
        p=repeats[repeats.pool==row['pool']].pivot(index='seed',columns='mode',values='sample_f1');d=p[row['mode']]-p[row['baseline']]
        assert abs(d.mean()-row['mean_delta'])<1e-12
        assert [int((d>1e-12).sum()),int((abs(d)<=1e-12).sum()),int((d< -1e-12).sum())]==[row[k] for k in ['wins','ties','losses']]
    # Matched controls remain detailed: independently check their sufficient stats.
    ref=study.controls()
    for mode in study.MODES[:3]:
        index=study.MODES.index(mode)
        for c in cases:
            mask=ref['case']==c['key'];values=study.arithmetic(ref['truth'][mask],study.decide(ref['scores'][index,mask],.5))
            stored=metric(stats[(stats['mode']==mode)&(stats['case']==c['key'])])
            for name,value in values.items():assert abs(value-stored[name])<1e-12
    if (HERE/'retention.json').exists():
        retained=json.loads((HERE/'retention.json').read_text())
        for path,digest in retained['files'].items():assert sha(ROOT/path)==digest,path
    print(json.dumps(dict(passed=True,matched_cases=84,metric_values_reconstructed=count,full_controls_rechecked=True,all_repeat_averages_and_paired_counts_match=True),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['collect','verify']);a=p.parse_args()
    collect() if a.command=='collect' else verify()
