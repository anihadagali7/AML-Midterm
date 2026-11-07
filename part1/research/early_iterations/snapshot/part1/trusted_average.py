"""Three predeclared equal-weight averages; reuse paired held-out scores."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from simple_benchmark import BASE, ROOT, load_data, report, metrics, digest
from simple_models import REASONS, decide
from trusted_benchmark import OUT


def main():
    stage = OUT/'07_fixed_equal_averages'
    stage.mkdir(exist_ok=True)
    source = OUT/'06_final_regularization_check'
    previous = OUT/'02_other_and_paired_labels'
    known, _ = load_data()
    folds = np.array([pd.read_csv(BASE/'splits'/f'repeat{r}.csv').set_index('ID').loc[known.ID,'fold'].to_numpy() for r in range(8)])
    combinations = {
        'average_C0.3_C0.1': ['final_r2_control','final_r2_C0.1'],
        'average_rounds2_3': ['final_r2_control','binary_neighbor_targets'],
        'average_trusted_ani': ['final_r2_control','ani_train_only'],
    }
    protocol = dict(combinations=combinations,weights=[.5,.5],cutoff=.3,
                    reason='Fixed equal averages check whether nearby simple fits reduce variability. No fitted ensemble or weight search.',
                    inputs={str(p.relative_to(ROOT)):digest(p) for p in [source/'probabilities.npz',previous/'probabilities.npz']},
                    fitting='Reuse strict held-out scores from the exact40folds; no new feature or label fitting.')
    (stage/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    data=np.load(source/'probabilities.npz')
    components={name:data[name] for name in ['final_r2_control','final_r2_C0.1']}
    components['binary_neighbor_targets']=np.load(previous/'probabilities.npz')['binary_neighbor_targets']
    ani=[]
    for repeat in range(8):
        df=pd.concat([pd.read_csv(BASE/'predictions'/f'repeat{repeat}_fold{f}__ani_train_only.csv') for f in range(5)]).set_index('ID').loc[known.ID]
        ani.append(df[[f'probability_{r}' for r in REASONS]].to_numpy())
    components['ani_train_only']=np.stack(ani)
    probabilities={name:.5*components[a]+.5*components[b] for name,(a,b) in combinations.items()}
    predictions={name:np.stack([decide(p,.3) for p in ps]) for name,ps in probabilities.items()}
    np.savez_compressed(stage/'probabilities.npz',**probabilities)
    np.savez_compressed(stage/'predictions.npz',ids=known.ID.to_numpy(str),truth=known[REASONS].to_numpy(int),folds=folds,**predictions)
    report(stage,known,folds,predictions)
    frame=pd.concat([pd.read_csv(source/'diagnostic_predictions.csv'),pd.read_csv(previous/'diagnostic_predictions.csv'),
                     pd.read_csv(ROOT/'experiments/simple_methodical/verification/diagnostic_predictions.csv')],ignore_index=True)
    rows,output=[],[]
    for split in ['original_validation','after_2003','after_2005','after_2007']:
        refs=frame[(frame.split==split)&(frame.name=='ani_train_only')].drop_duplicates('ID').set_index('ID')
        truth=refs[[f'actual_{r}' for r in REASONS]].to_numpy(int)
        for name,(a,b) in combinations.items():
            pair=[]
            for component in [a,b]:
                d=frame[(frame.split==split)&(frame.name==component)].drop_duplicates('ID').set_index('ID').loc[refs.index]
                np.testing.assert_array_equal(d[[f'actual_{r}' for r in REASONS]],truth)
                pair.append(d[[f'probability_{r}' for r in REASONS]].to_numpy())
            P=.5*pair[0]+.5*pair[1];pred=decide(P,.3)
            rows.append(dict(split=split,name=name,held=len(truth),**metrics(truth,pred)))
            out=pd.DataFrame(dict(ID=refs.index,split=split,name=name))
            for j,r in enumerate(REASONS):
                out[f'actual_{r}']=truth[:,j];out[f'probability_{r}']=P[:,j];out[f'predicted_{r}']=pred[:,j]
            output.append(out)
    pd.DataFrame(rows).to_csv(stage/'diagnostic_scores.csv',index=False)
    pd.concat(output).to_csv(stage/'diagnostic_predictions.csv',index=False)
    print(pd.DataFrame(rows)[['split','name','sample_f1','macro_f1']].to_string(index=False))


if __name__=='__main__':
    main()
