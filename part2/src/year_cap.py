"""Constrain linear year effects to the latest observed training year."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import json
import numpy as np
import pandas as pd
from experiments import OUT,load_data,prepare,matrix,fit_linear,predict_linear,metrics


def run():
    data=load_data();configs={}
    for split in ['temporal','validation']:
        tr,ev,f,bt,be,bx=prepare(data,split)
        pred={};test={};scores=[]
        for rep in ['criteria','clinical','rich']:
            for C in [.5,2.]:
                xs=[matrix(b,rep,'year_cap') for b in [bt,be,bx]]
                pair=[];pt=[]
                for separate in [False,True]:
                    models=fit_linear(xs[0],tr,C,separate)
                    pair.append(predict_linear(models,xs[1],ev));pt.append(predict_linear(models,xs[2],data['test']))
                for alpha in [0.,.25,.5]:
                    name=f'lr_{rep}_yearcap_C{C:g}_phase{alpha:g}'
                    p=(1-alpha)*pair[0]+alpha*pair[1]
                    pred[name]=p;test[name]=(1-alpha)*pt[0]+alpha*pt[1]
                    configs[name]=dict(family='lr',representation=rep,year='year_cap',C=C,phase_weight=alpha)
                    scores.append(dict(name=name,**metrics(ev,p)))
        np.savez_compressed(OUT/f'yearcap_{split}_predictions.npz',**pred)
        np.savez_compressed(OUT/f'yearcap_{split}_test_predictions.npz',**test)
        pd.DataFrame(scores).sort_values('phase_mean_logloss').to_csv(OUT/f'yearcap_{split}_scores.csv',index=False)
        print(split,pd.DataFrame(scores)[['name','phase_mean_logloss']].sort_values('phase_mean_logloss').head(8).to_string(index=False),flush=True)
    (OUT/'yearcap_configs.json').write_text(json.dumps(configs,indent=2))

if __name__=='__main__':run()
