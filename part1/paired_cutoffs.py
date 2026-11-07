"""Two fixed final-decision sensitivity checks; reuse all 55 fitted fold scores."""
import argparse
import json
from pathlib import Path

import pandas as pd

from simple_benchmark import digest,metrics
from simple_models import REASONS,decide
from paired_search import OUT,REFERENCE,summarize


def run(source_stage,source_name):
    parent=OUT/source_stage
    stage=OUT/'05_final_decision';stage.mkdir(exist_ok=True)
    protocol=dict(parent=source_stage,name=source_name,cutoffs=[.25,.3,.35],
                  parent_sha256={f:digest(parent/f) for f in ['protocol.json','predictions.csv','diagnostic_predictions.csv']},
                  policy='Only the final prediction cutoff changes. Fitting and pseudo-label cutoff stay fixed; reuse every saved fold score in both pools. These are development comparisons.')
    (stage/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (stage/'paired_cutoffs.py').write_bytes(Path(__file__).read_bytes())
    source=pd.read_csv(parent/'predictions.csv');source=source[source.name==source_name]
    baseline=pd.read_csv(REFERENCE/'predictions.csv');baseline=baseline[baseline.name.isin(['ani_current','trusted_two_pass'])]
    outputs=[];diagnostic_outputs=[];rows=[]
    diagnostics=pd.read_csv(parent/'diagnostic_predictions.csv');diagnostics=diagnostics[diagnostics.name==source_name]
    for cutoff in protocol['cutoffs']:
        for original,result in [(source,outputs),(diagnostics,diagnostic_outputs)]:
            df=original.copy();df['name']=f'cutoff_{cutoff}'
            pred=decide(df[[f'probability_{r}' for r in REASONS]].to_numpy(),cutoff)
            for j,r in enumerate(REASONS):df[f'predicted_{r}']=pred[:,j]
            result.append(df)
    df=pd.concat([baseline,*outputs],ignore_index=True);df.to_csv(stage/'predictions.csv',index=False)
    summarize(stage,df)
    dp=pd.concat(diagnostic_outputs,ignore_index=True);dp.to_csv(stage/'diagnostic_predictions.csv',index=False)
    for (split,name),g in dp.groupby(['split','name']):
        rows.append(dict(split=split,name=name,**metrics(g[[f'actual_{r}' for r in REASONS]].to_numpy(),g[[f'predicted_{r}' for r in REASONS]].to_numpy())))
    ref=pd.read_csv(REFERENCE/'current_diagnostic_scores.csv');ref=ref[ref.name.isin(['ani_current','trusted_two_pass'])]
    pd.concat([ref,pd.DataFrame(rows)]).to_csv(stage/'diagnostic_scores.csv',index=False)
    print(pd.DataFrame(rows).pivot(index='name',columns='split',values='sample_f1').to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('name');a=p.parse_args();run(a.stage,a.name)
