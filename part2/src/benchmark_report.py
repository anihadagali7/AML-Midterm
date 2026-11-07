"""Report and standalone plots for complete benchmark runs."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'experiments/.matplotlib'))
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,FormatStrFormatter
from benchmark_protocol import DIRECTORY,initialize
from experiments import ROOT

LABELS={'ani_stack':'Ani stack','ani_meta_C0.3':'Stronger meta L2','ani_phase3_trend':'Full Phase III trend',
        'our_linear':'Our linear component','blend_ani75_linear25':'75% Ani + 25% linear',
        'blend_ani50_linear50':'50% Ani + 50% linear','ani_mild_phase3_trend':'75% Ani + 25% trend',
        'ani_anchored_linear25':'75% Ani + 25% linear'}


def markdown(df):
    lines=['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):
        lines.append('| '+' | '.join('' if pd.isna(x) else f'{x:.6f}' if isinstance(x,float) else str(x) for x in row)+' |')
    return '\n'.join(lines)


def run():
    _,protocol=initialize();pointer=json.loads((DIRECTORY/'current_run.json').read_text());out=ROOT/pointer['directory']
    strict=pd.read_csv(out/'summary.csv');published=pd.read_csv(out/'published_recipe_track/summary.csv')
    assert strict.complete_comparison.all() and published.complete_comparison.all(),'Do not report incomplete comparisons as final.'
    figures=out/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    names=[x for x in LABELS if x in set(strict.model) and x!='ani_stack']
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),sharey=True)
    for ax,kind,title in zip(axes,['temporal','group_cv'],['Three chronological windows','Two repeats of grouped 3-fold CV']):
        rows=strict.set_index('model').loc[names]
        point=rows[kind+'_delta'].to_numpy();lo=rows[kind+'_delta_lower'].to_numpy();hi=rows[kind+'_delta_upper'].to_numpy()
        ax.errorbar(point,np.arange(len(names)),xerr=np.vstack([np.maximum(0,point-lo),np.maximum(0,hi-point)]),fmt='o',color='#087f8c',capsize=3)
        ax.axvline(0,color='.4',ls=':');ax.set_title(title);ax.set_xlabel('Log-loss difference from Ani (negative is better)')
        ax.set_yticks(np.arange(len(names)),[LABELS[n] for n in names])
    axes[0].invert_yaxis()
    fig.suptitle('Paired trial-bootstrap intervals; descriptive 95% ranges',fontsize=12)
    fig.tight_layout();fig.savefig(figures/'paired_comparisons.png');plt.close(fig)
    fold=pd.read_csv(out/'fold_scores.csv');temporal=fold[fold.kind.eq('temporal')]
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),sharey=True)
    limits=[]
    for ax,(name,g) in zip(axes,temporal.groupby('split',sort=True)):
        baseline=g[g.model.eq('ani_stack')].phase_mean_logloss.iloc[0]
        rows=g.set_index('model').loc[names]
        delta=rows.phase_mean_logloss.to_numpy()-baseline;limits.extend(delta)
        ax.scatter(delta,np.arange(len(names)),color=['#c36341' if d>0 else '#087f8c' for d in delta])
        ax.axvline(0,color='.4',ls=':');ax.set_title(name.replace('time_','').replace('_to_',' → ').replace('_','–'))
        ax.set_yticks(np.arange(len(names)),[LABELS[n] for n in names]);ax.set_xlabel('Log-loss difference from Ani')
    axes[0].invert_yaxis()
    for ax in axes:
        ax.set_xlim(min(limits)-.002,max(limits)+.002)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.tick_params(axis='x',labelsize=9)
    fig.tight_layout();fig.savefig(figures/'chronological_windows.png');plt.close(fig)
    columns=['model','official_logloss','temporal_mean_logloss','worst_temporal_logloss','cv_mean_logloss','temporal_mean_delta_vs_ani','cv_mean_delta_vs_ani']
    derived=[]
    for track in [out,out/'published_recipe_track']:
        path=track/'derived_candidates/summary.csv'
        if path.exists():
            d=pd.read_csv(path);d.insert(0,'track','Strict' if track==out else 'Published recipe');derived.append(d)
    exact=json.loads((ROOT/'experiments/ani_comparison/exact_reproduction.json').read_text())
    report=f'''# Part Two benchmark results

**Hold 05.** Its underlying linear component should be judged against Ani on the saved chronological and grouped splits, not inferred to win by extrapolating two leaderboard scores. The best confirmed reference remains Ani's **0.58688**; our reported scores are **02: 0.58993**, **03: 0.59562**.

Ani commit: `1b8bd11c47428fb5453e53cdc275542b92a3b347`. Exact final-submission reproduction has maximum probability difference **{exact[1]['max_submission_probability_difference']:.3g}** from the published CSV. The source notebook, CSVs and hashes are preserved in `experiments/ani_comparison`.

## Strict comparison on the same saved splits

{markdown(strict[columns])}

All ten splits are complete for every model. Features and stacking-inner preprocessing use fitting rows only. This adapts Ani's algorithm to a strict prospective evaluation. The three chronological windows are 2008–2009, 2010–2011 and 2012–2013, always fitting earlier years. Group CV uses two seeded repeats of three trial-grouped folds over the same labeled universe. Combined-phase rows stay together. Log loss is averaged equally across phases; CV is pooled across held-out rows within each repeat.

The 50/50 Ani/linear blend improves supplied validation and grouped CV, yet is worse on chronological mean loss. The linear endpoint is worse across time. This warns against using either validation F1 or random CV alone to choose a submission.

![Paired comparisons](runs/{pointer['run']}/figures/paired_comparisons.png)

Negative differences favor the candidate. Intervals spanning zero indicate inconclusive local evidence. The ranges are conditional, descriptive trial bootstraps, not guarantees for a future cohort or corrections for model selection.

![Chronological windows](runs/{pointer['run']}/figures/chronological_windows.png)

## Faithful published-recipe sensitivity track

{markdown(published[columns])}

This track preserves Ani's feature-only transduction: vocabulary/IDF sees all available train/validation/test text. Evaluation outcome labels never fit a model. These results are kept separate from strict scores. Both tracks reuse the exact same trial splits. Original Ani submission refits supervised models on 9,050 train+validation rows, whereas our initial submitted models fitted only 8,037 training rows. That difference also prevents attributing the leaderboard gap to the algorithm alone.

## Conservative follow-up comparisons

'''
    if derived:
        d=pd.concat(derived,ignore_index=True)
        report+=markdown(d[['track']+columns])+'\n\n'
    report+='''These are exploratory fixed blends, evaluated on every saved split. The mild trend interpolates 25% toward Ani's existing Phase III trend alternative; the linear blend puts 25% weight on our linear component. If using another exploratory submission, prefer optional **07_ani_mild_phase3_trend.csv** over 05: it remains anchored to Ani's known winning file and has favorable point differences in both tracks. Its temporal interval still includes zero. The linear blend improves grouped CV but is less consistent across time. Neither candidate has a known Kaggle score; preserve Ani's 0.58688 as the confirmed reference.

## Repeatability and future candidates

See [LOCAL_TESTING.md](../../LOCAL_TESTING.md) for protocol, commands and limitations. `protocol.json` contains immutable IDs, grouping, seeds, periods and data hashes. Each run records implementation hashes and package versions. Models and exact held-out predictions are cached. `fold_scores.csv`, `summary.csv`, per-fold `audit.json`, and prediction files make independent rescoring possible. Incomplete runs cannot be presented as completed comparisons.

Use this suite for the next candidate batch. Do not choose one favorable validation F1 or change boundaries to make a model win. Freeze the recipe before generating a final CSV; refit final components consistently on train plus validation, while keeping held-out benchmark metrics separate. Kaggle results remain the only direct evidence of performance on its newer cohort.
'''
    (DIRECTORY/'RESULTS.md').write_text(report)
    print('Wrote benchmark report and figures:',DIRECTORY/'RESULTS.md')

if __name__=='__main__':run()
