"""Export the assignment's per-phase comparisons and report-ready figures."""
import os
os.environ.setdefault('MPLCONFIGDIR',str(__import__('pathlib').Path(__file__).resolve().parent/'experiments/.matplotlib'))
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from experiments import ROOT,OUT,PHASES,load_data

FIGURES=OUT/'figures'


def mdtable(df):
    rows=['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):
        rows.append('| '+' | '.join('' if pd.isna(v) else f'{v:.5f}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(rows)


def run():
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    data=load_data();d=data['val'];selected=np.load(OUT/'selected_validation_predictions.npz')
    baseline=np.load(OUT/'linear_validation_predictions.npz')['lr_criteria_none_C2_phase0']
    improved=selected['03_phase_specialized']
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for j,phase in enumerate(PHASES):
        m=d.phase.eq(phase)
        for name,p,color in [('Text baseline',baseline,'#8495a6'),('Selected ensemble',improved,'#087f8c')]:
            RocCurveDisplay.from_predictions(d.label[m],p[m],ax=axes[0,j],name=name,curve_kwargs={'color':color})
            PrecisionRecallDisplay.from_predictions(d.label[m],p[m],ax=axes[1,j],name=name,curve_kwargs={'color':color})
        axes[0,j].set_title(f'Phase {phase} (n={m.sum()})')
        axes[0,j].plot([0,1],[0,1],ls=':',c='.6')
        axes[0,j].set_xlabel('False positive rate');axes[0,j].set_ylabel('True positive rate')
        axes[1,j].set_xlabel('Recall');axes[1,j].set_ylabel('Precision')
        axes[0,j].legend(fontsize=8,loc='lower right')
        axes[1,j].legend(fontsize=8,loc='lower left')
    fig.tight_layout();fig.savefig(FIGURES/'phase_roc_pr.png');plt.close(fig)
    comparison=pd.read_csv(OUT/'reason_error_comparison.csv')
    fig,axes=plt.subplots(1,3,figsize=(13,4.7),sharey=True)
    for ax,phase in zip(axes,PHASES):
        t=comparison[comparison.phase.eq(phase)].copy();x=np.arange(len(t))
        ax.bar(x-.18,t.false_success_rate_baseline,width=.36,color='#8495a6',label='Baseline')
        ax.bar(x+.18,t.false_success_rate_improved,width=.36,color='#087f8c',label='Ensemble')
        labels=[('No reason' if r.startswith('No recorded') else r)+f'\nn={n}' for r,n in zip(t.reason,t.n)]
        ax.set_xticks(x,labels,rotation=50,ha='right');ax.set_ylim(0,1.05);ax.set_title(f'Phase {phase}')
    axes[0].set_ylabel('Fraction of unsuccessful trials predicted successful')
    axes[-1].legend(loc='upper left',fontsize=8)
    fig.tight_layout();fig.savefig(FIGURES/'reason_errors.png');plt.close(fig)
    components=pd.read_csv(OUT/'all_component_scores.csv')
    manifest=pd.read_csv(ROOT/'submissions/index.csv')
    fig,ax=plt.subplots(figsize=(7.5,5))
    ax.scatter(components.validation_phase_mean_logloss,components.temporal_phase_mean_logloss,
        s=12,alpha=.22,c='#8495a6',label='Individual configurations')
    for row in manifest.itertuples():
        label=row.submission.removesuffix('.csv').replace('_',' ')
        ax.scatter(row.validation_logloss,row.temporal_logloss,s=110,marker='*',label=label)
    ax.set_xlabel('Supplied validation: phase-mean log loss (lower is better)')
    ax.set_ylabel('Later-year holdout: phase-mean log loss')
    ax.set_xlim(.54,.63);ax.set_ylim(.54,.64);ax.legend(loc='upper left',fontsize=8)
    fig.tight_layout();fig.savefig(FIGURES/'model_comparison.png');plt.close(fig)
    phases=pd.read_csv(OUT/'phase_metrics.csv')
    comparison_table=phases[phases.model.isin(['baseline','03_phase_specialized'])]
    category=comparison[['phase','reason','n','false_success_baseline','false_success_improved',
                         'true_reason_rows_baseline','predicted_reason_rows_baseline']]
    category.columns=['phase','reason','n','baseline false-success','ensemble false-success','true reason rows','predicted reason rows']
    total_models=len(components)
    total_ensembles=len(pd.read_csv(OUT/'ensemble_scores.csv'))
    best=manifest.iloc[0]
    report=f'''# Part Two: clinical-trial outcome prediction

**Latest recommendation: hold 05/06.** Ani's published 0.58688 model has now been reproduced. Use the [repeatable local benchmark](LOCAL_TESTING.md), [executed benchmark notebook](local_benchmark.ipynb), and [complete results](experiments/local_benchmark/RESULTS.md) for new comparisons. The original selection below is historical. Optional Ani-anchored candidates are documented in [ani_candidates_manifest.json](submissions/ani_candidates_manifest.json); their Kaggle scores remain unknown.

Prepared four initial manual Kaggle submissions and two follow-ups. **Lower log loss is better.** The best reported reference score is **0.58688**. Our `03_phase_specialized.csv` scored **0.59562**; `02_text_heavy.csv` improved to **0.58993**, still 0.00305 behind the reference. Initial candidates 01 and 04 have unknown Kaggle scores. Results are recorded against exact file hashes in [submissions/leaderboard_results.json](submissions/leaderboard_results.json).

## Evaluation protocol

Train: 8,037 rows / 7,535 trials. Supplied validation: 1,013 rows / 901 trials. Test: 1,500 rows, exactly 500 per phase. No trial crosses these files. Combined-phase rows stay together.

The chronological check trains on 5,660 rows starting through 2008 and evaluates on 2,377 training-file rows starting in 2009-2013. Every TF-IDF vocabulary, IDF, SVD, and metadata vocabulary is fitted only on that split's training population. Final models use **outcome_train.csv only**. No validation labels enter their fitting.

Selection compares the equal mean of the three per-phase log losses, matching Kaggle's phase proportions. The supplied validation set itself has unequal phase counts, so its ordinary row-weighted log loss is also saved but is not the selection objective. Both local splits are consulted for selection; neither is an untouched final assessment. Model selection across many settings can overfit these evaluations.

## Methods and comparisons

Evaluated {total_models} matching individual configurations (including pooled/separate-model blends) across both splits and {total_ensembles} fixed-weight ensemble configurations. Compare word and character TF-IDF, binary word features, clinical metadata, drug targets/actions/pathways, Morgan fingerprints and molecular descriptors, pooled and separate phase models, shared phase interactions, phase balancing, newer-trial weighting, raw/capped year effects, and NB log-ratio features. Classifiers include L2 logistic regression, CatBoost, histogram gradient boosting, sparse LightGBM, and Extra Trees. No deep learning or external outcome data is used.

The baseline is eligibility-criteria TF-IDF (unigrams/bigrams, at most 35,000 terms, minimum document frequency 3, sublinear TF), three phase indicators, and pooled L2 logistic regression with C=2. All F1 values below use cutoff 0.5. ROC-AUC and PR-AUC use probabilities, with success (label=1) as positive. PR-AUC means average precision.

Text models share signal across phases. Separate phase fits and phase interactions can overfit the smaller Phase I population. Molecular fingerprints do not consistently improve the text/clinical control. Raw year extrapolation is sensitive to the historical success trend; the capped-year alternative stops its effect at the latest observed training year. Drug annotations are less common in test (39.3%) than training (45.7%), so relying heavily on them risks a coverage shift. These patterns motivate mixed text/tree candidates and an explicit temporal comparison.

## Initial four submission candidates

{mdtable(manifest[['submission','validation_logloss','temporal_logloss','mean_test_probability','kaggle_score']])}

`03_phase_specialized.csv` was the lowest joint-loss candidate before Kaggle evaluation, but its local advantage did not transfer. 02 improved the leaderboard score by reducing tree influence in Phases II/III. That motivated diagnostic files 05/06, now **on hold**: our linear endpoint is worse than Ani in the new chronological comparisons. Their historical recipes remain in [submissions/followup_manifest.json](submissions/followup_manifest.json). The new benchmark compares strict fold-local preprocessing and Ani's faithful transductive recipe on the same ten saved splits. Optional 07 is a mild interpolation toward Ani's Phase III trend variant; 08 is an Ani-heavy linear blend. Read their paired uncertainty before submitting. Original files are preserved. Initial 01–06 use 8,037 supervised training rows; Ani's reference and the new final linear component use 9,050 train+validation rows after separate held-out evaluation. Reduced temporal fits are never submitted.

## Per-phase baseline and modification

{mdtable(comparison_table)}

![Per-phase ROC and precision-recall curves](experiments/model_search/figures/phase_roc_pr.png)

## Stopping-reason error analysis

Only unsuccessful validation rows are included. Their `trial_id` is matched to Part One's `ID` after outcome prediction. Use visible Part One train/validation labels when available; otherwise use the frozen original DBSCAN/logistic Part One model (157 verified training labels, C=1000, cutoff 0.5 with argmax fallback). The exact source model and checksum are recorded in `reason_model_provenance.json`. **Reason embeddings, reason labels, and membership in Part One never enter an outcome classifier.**

Rows absent from Part One form the separate no-recorded-stopping-reason group. Reasons overlap, so group counts need not sum to the number of failures. Each trial is counted at most once within a phase. Funding has no Phase III examples; its rate is undefined, not zero.

The baseline incorrectly predicts success for 8 of 9 Phase III efficacy-tagged failures. All nine tags are model predictions, so this is an uncertain subgroup signal. To address the baseline's restricted eligibility-only view, the comparisons add clinical and molecular context and nonlinear combinations, then blend probabilities to retain text generalization. The final table shows the modification's actual trade-offs; improvement in overall log loss need not improve every reason category or F1 at a fixed threshold.

{mdtable(category)}

![Baseline and modified false-success rates](experiments/model_search/figures/reason_errors.png)

Per-category rates, Wilson intervals, known/predicted-label counts, and all four candidates are saved in `experiments/model_search/*reason_errors.csv`. Small groups and predicted reason-label errors limit category-level conclusions. The paired trial-level bootstrap is descriptive and uses the same validation population used for selection.

## Reproduce

Open [part2.ipynb](part2.ipynb). It contains the complete module source plus the executed evaluation and figures. Set `RUN_TRAINING=True` to rerun all searches; the default reads the saved results. Install the pinned [requirements.txt](requirements.txt) in a virtual environment.

Run each search script, then `ensemble_search.py`, preserve the selected mapping in `experiments/model_search/selected_ensembles.json`, run `submission_models.py`, `error_analysis.py`, and `reporting.py`. `model_replay_verification.csv` checks independent linear refits and saved-model reloads against both stored validation and test probabilities. All exported files are checked for 1,500 unique IDs in sample-submission order, finite probabilities strictly between 0 and 1, exact column names, and CSV round-trip agreement.

For inference from the saved bundle: `python submission_models.py --input outcome_test.csv --output replayed_predictions`.

This directory supplies the Part Two implementation and report material. Integrate it with the team's Part One results in the required project report (maximum five pages, excluding references).
'''
    (ROOT/'README.md').write_text(report)
    short=manifest[['submission','validation_logloss','temporal_logloss','kaggle_score']]
    followup_path=ROOT/'submissions/followup_index.csv'
    followup_text='\n\n## Historical follow-ups — on hold\n\nHold **05/06** after the Ani comparison. The new chronological benchmark favors Ani over our linear endpoint.\n\n'+mdtable(pd.read_csv(followup_path)[['submission','validation_logloss','temporal_logloss','kaggle_score']]) if followup_path.exists() else ''
    ani_path=ROOT/'submissions/ani_candidates_index.csv'
    ani_text='\n\n## Optional Ani-anchored candidates\n\n07 blends 75% of the known 0.58688 reference with 25% of Ani\'s capped Phase III trend alternative. It has small favorable point differences across both benchmark tracks, but temporal intervals include zero. 08 is an Ani-heavy linear blend: it improves grouped CV but is less consistent across time. Neither has a known Kaggle score. Prefer 07 if spending another exploratory submission; retain Ani as the confirmed best. See [the benchmark report](../experiments/local_benchmark/RESULTS.md).\n\n'+mdtable(pd.read_csv(ani_path)[['submission','mean_abs_change_from_ani','kaggle_score']]) if ani_path.exists() else ''
    (ROOT/'submissions/README.md').write_text('# Part Two manual Kaggle submissions\n\nLower log loss is better. Best reported reference: **0.58688**. Our 03 scored **0.59562**, then 02 improved to **0.58993**. Initial 01/04 remain unscored. **Hold 05/06; use the new fixed benchmark for future comparisons.**\n\n'+mdtable(short)+followup_text+ani_text+'\n\nAll numbered files have exactly 1,500 rows and columns `ID,label`. Index CSVs are bookkeeping and should not be uploaded. Leaderboard results are tied to file hashes in `leaderboard_results.json`; exact recipes are in the corresponding manifest JSONs.\n')
    print('Wrote report and',len(list(FIGURES.glob('*.png'))),'figures')

if __name__=='__main__':run()
