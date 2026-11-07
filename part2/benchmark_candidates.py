"""Two conservative, pre-specified blends scored on EVERY frozen split.

No new base-model fitting: use saved out-of-fold predictions from each track.
The 25% linear blend was already in the original batch. The mild year variant
is a conservative interpolation toward Ani's existing Phase III alternative.
"""
import json
import pandas as pd
from benchmark_protocol import initialize,digest
from local_benchmark import run_directory,predict_and_score,summarize
from experiments import ROOT

WEIGHTS={'ani_mild_phase3_trend':{'ani_stack':.75,'ani_phase3_trend':.25},
         'ani_anchored_linear25':{'ani_stack':.75,'our_linear':.25}}


def run():
    _,protocol=initialize();strict=run_directory(protocol)
    for parent in [strict,strict/'published_recipe_track']:
        for split in protocol['splits']:
            assert all((parent/'predictions'/split['name']/f'{model}.csv').exists() for model in ['ani_stack','ani_phase3_trend','our_linear']),f'Complete parent track first: {parent}/{split["name"]}'
        out=parent/'derived_candidates';out.mkdir(exist_ok=True)
        (out/'run_manifest.json').write_text(json.dumps({'parent':str(parent.relative_to(ROOT)),
            'parent_manifest_sha256':digest(parent/'run_manifest.json'),'protocol_sha256':digest(ROOT/'experiments/local_benchmark/protocol.json'),
            'weights':WEIGHTS,'selection_note':'Exploratory, fixed conservative blends after inspecting the initial batch; every saved split is evaluated, not just favorable periods.'},indent=2))
        for split in protocol['splits']:
            folder=parent/'predictions'/split['name']
            banks={name:pd.read_csv(folder/f'{name}.csv') for name in ['ani_stack','ani_phase3_trend','our_linear']}
            df=banks['ani_stack']
            assert all(frame.ID.equals(df.ID) for frame in banks.values())
            ps={'ani_stack':df.probability.to_numpy()}
            ps.update({name:sum(w*banks[key].probability.to_numpy() for key,w in weights.items()) for name,weights in WEIGHTS.items()})
            predict_and_score(out,split,df,ps)
        summarize(out,protocol)
        print('Derived candidates:',out,flush=True)

if __name__=='__main__':run()
