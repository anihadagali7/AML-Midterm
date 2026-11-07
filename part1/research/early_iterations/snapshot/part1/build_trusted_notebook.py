"""Create the complete follow-up report and executable source notebook."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parent
cells=[]
def md(text):cells.append(nbf.v4.new_markdown_cell(text))
def code(text):cells.append(nbf.v4.new_code_cell(text))

md("""# Part 1: trust verified labels and keep multiple reasons

**Recommended next candidate:** the compact [trusted_solution.py](trusted_solution.py) model scores **0.915337 sample F1**, versus **0.908865** for the frozen training-only Ani reference. It improves all three chronological diagnostics. The old 78-row validation check remains weaker: **0.880342 versus 0.891026**. This is a stronger local candidate, not a confirmed Kaggle win; all local comparisons have influenced development.

[Checked submission CSV](experiments/trusted_iteration/candidate/part1_trusted_submission.csv) · [model manifest](experiments/trusted_iteration/candidate/manifest.json) · [all results](experiments/trusted_iteration/all_results.csv) · [audit](experiments/trusted_iteration/final_audit.json)

## What changed

The selected model keeps the original simple architecture: standardization, a joint logistic classifier (`C=0.3`, `tol=1e-4`, `max_iter=3000`), and ten cosine neighbors with squared positive-similarity weights. The classifier score has weight 0.7; neighbors have weight 0.3. The final cutoff stays 0.3 with an argmax fallback.

Four changes address specific problems:

1. Give each generated example **one quarter** of the weight of an equivalent verified example.
2. Compute inverse reason-frequency class weights **only from verified labels**, then hold them fixed through self-training. A verified two-reason trial still contributes half to each reason in the joint logistic loss. In every CV fit, verified examples retain total effective weight 188.
3. In the neighbor vote, a trial with two reasons contributes a full binary vote to **both**. Splitting its vote in half suppressed the second reason. The resulting selection scores need not sum to one and are not calibrated marginal probabilities.
4. Use **two** pseudo-labeling passes, rebuilding from verified labels each time. Accept a row only when its highest score reaches 0.8.

No extra “Other” classifier, adaptive cutoff, GMM, PCA, graph, or ensemble survives into the selected solution. The final fit uses 235 verified trials and accepts 698, then 766 of the 1,014 originally unlabeled trials. Test inputs are opened only after fitting completes.

## Same testing contract

Every trained candidate uses the exact saved 8 × 5 folds, with 188 known fitting trials and 47 held out. All learned preprocessing and pseudo-labeling exclude held-out and Kaggle test inputs. The 1,014 originally unlabeled training trials are the only unlabeled pool. Sample F1 is pooled over 235 out-of-fold predictions per repeat and then averaged across repeats. All original files, prior models and prior submissions are preserved.

Every candidate also has the old 157/78 split and chronological cutoffs 2003/2005/2007. The chronological checks exclude later-year unlabeled features too. **These checks are now reused development diagnostics, not independent confirmation.** The Kaggle score 0.89423 belongs to Ani's original transductive submission; the active comparison here is against Ani's training-only recipe on identical folds, not against that Kaggle number.
""")
code("""from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import display
PART1=Path.cwd()
if not (PART1/'data/part1_train.csv').exists():PART1=PART1/'part1'
OUT=PART1/'experiments/trusted_iteration'
comparison=pd.read_csv(OUT/'selected_comparison.csv').set_index('name')
display(comparison[['mean_f1','repeat_sd','macro_f1','delta_ani','repeat_wins_ani','fold_wins_ani','fold_ties_ani']])
display(comparison[[f'f1_{r}' for r in ['Safety','Funding','Efficacy','Recruitment','Other']]])
""")
md("""Against Ani, the candidate wins **25 folds**, ties **7**, and loses **8**. It wins **7 of 8 repeat averages**. Macro F1 improves from 0.822043 to 0.831522; repeat SD falls from 0.006631 to 0.004152. Four reasons improve; Efficacy is slightly lower (0.867622 versus 0.869526).

The descriptive paired trial-bootstrap interval for the +0.006472 advantage is **[+0.001241, +0.012234]**. Each trial's eight predictions stay together. The interval does **not** adjust for model selection, prior searches, or correlated training fits, so it is not a formal significance claim or a forecast of leaderboard improvement.
""")
code("""display(pd.read_csv(OUT/'paired_trial_bootstrap.csv'))
diagnostics=pd.read_csv(OUT/'selected_diagnostics.csv')
display(diagnostics[['split','name','verified_fit','unlabeled_fit','held','sample_f1','macro_f1']])
display(diagnostics.pivot(index='split',columns='name',values='sample_f1'))
""")
md("""The chronological gains are small but consistent: +0.003058 after 2003, +0.003876 after 2005, and +0.004505 after 2007. Those holdouts have 109, 86 and 37 known trials respectively. The old validation result is lower by 0.010684; it uses only the original 157 known fitting labels, while the eventual submission uses all 235. That difference in training population helps explain why the checks can disagree, but does not make the weaker result disappear.

## What was actually tested

The follow-up contains **66 named configurations, including repeated controls**, across seven stages. Every negative result is retained.

| Stage | Question | Finding |
|---|---|---|
| 1 | Lower pseudo-label weights; anchor class balancing; select an Other cutoff with inner CV | Lower weights alone are inconsistent. Anchored quarter weights improve later-year checks. Nested threshold tuning does not improve the original control. |
| 2 | Conditional/independent Other classifiers, extra weight on verified pairs, or binary neighbor targets | Extra classifiers and pair weights do not help. Full binary neighbor votes do. |
| 3 | Nearby C values, passes, class-balance strength, neighbor trust, and soft pseudo-labels | Two passes lead; soft pseudo-labels deteriorate strongly. |
| 4 | Relative cutoffs and expected-F1 decisions | None improves the fixed 0.3 policy. |
| 5 | Learn observed reason combinations with one multiclass logistic model | Best combination model scores 0.909273; below the selected model. |
| 6 | Combine stronger regularization with fewer passes; check confidence thresholds | Two passes, C0.3 and confidence0.8 remain the selected single model. |
| 7 | Three fixed equal-weight averages | Averaging C0.1/C0.3 scores 0.915514, only 0.000177 higher, with higher repeat SD and fewer repeat wins. Keep the single model. |

For Other-threshold selection, each outer training set is divided into three stratified inner folds (seed 8128). Inner held-out inputs are excluded from fitting, including scaling and pseudo-labeling. Select a cutoff from the fixed list 0.3/0.25/0.35/0.2/0.4/0.5/0.1 using total sample F1, with fixed tie preference. All 80 outer threshold selections (240 inner fits) are saved and independently rescored. Their choices vary considerably, which is a reason to retain the fixed policy.
""")
code("""results=pd.read_csv(OUT/'all_results.csv')
display(results[~results.name.isin(['ani_train_only','dbscan_train_only'])].sort_values('mean_f1',ascending=False).head(15)[['stage','name','mean_f1','macro_f1','repeat_sd','delta_ani']])
display(pd.read_csv(OUT/'error_analysis/prior_other_and_multilabel_errors.csv'))
""")
md("""## What the error analysis does and does not show

There are 19 two-reason trials among the 235 known trials. The prior simplified model scored only 0.716886 on that subgroup, versus Ani's 0.724781. The new selected model reaches **0.758772**. Its single-reason subgroup also improves over Ani: 0.929109 versus 0.925058.

However, the nine Recruitment+Other trials remain difficult. Their sample F1 is **0.662037 for both Ani and the selected model**. Other-only trial sample F1 also ties Ani at 0.701923. The improved overall Other F1 therefore should not be described as solving those missed second reasons; fewer false-positive Other assignments help. Targeted Other classifiers were tested and rejected. This remains a limitation.

## Artifacts and independent checks

- `trusted_solution.py` is the compact implementation with fixed settings, independent of the experimental model class.
- Its scores and labels reproduce all **40** selected CV fits and all **4** diagnostic fits. The raw weight calculation is independently checked: known effective mass stays 188 inside each CV fit, and each generated label gets its declared quarter weight before the fixed class multiplier.
- `candidate/trusted_two_pass.joblib` reloads with exactly identical test scores. The CSV has 312 unique IDs, correct order and nonempty valid labels.
- `candidate/generated_labels.csv` separates accepted fitting targets from final predictions for all unlabeled rows. `candidate/part1_train_augmented.csv` preserves every original verified label and marks generated labels. Part 2 data is unchanged.
- Original benchmark/data/model/submission checksums and the prior simple candidate's checksums still match. No upload, commit or push was performed.

The assignment's unsupervised label-generation comparisons remain in the prior notebooks, including the saved K-means augmented dataset. This selected recipe uses self-training; it should not be described as a clustering model. Submit the prior method comparisons alongside this follow-up if preparing the project report.
""")
code("""display(json.loads((OUT/'final_audit.json').read_text()))
manifest=json.loads((OUT/'candidate/manifest.json').read_text())
display({k:manifest[k] for k in ['status','known_labels','final_generated_counts','final_fit_weights',
                               'independent_fold_replays','independent_diagnostic_replays','kaggle_score','versions']})
""")
code("""# Standalone figure; higher F1 is better. These are development comparisons.
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(10,3.4))
models=['ani_train_only','final_r2_control'];labels=['Ani','Trusted two-pass']
axes[0].bar(labels,comparison.loc[models,'mean_f1'],color=['#718096','#167d8d'])
axes[0].set_ylim(.88,.93);axes[0].set_ylabel('Mean sample F1');axes[0].set_title('Fixed 40-fold benchmark')
for i,value in enumerate(comparison.loc[models,'mean_f1']):axes[0].text(i,value+.001,f'{value:.4f}',ha='center')
wide=diagnostics.pivot(index='split',columns='name',values='sample_f1').loc[['after_2003','after_2005','after_2007']]
x=np.arange(len(wide));w=.34
for i,(model,label,color) in enumerate(zip(models,labels,['#718096','#167d8d'])):
    axes[1].bar(x+(i-.5)*w,wide[model],w,label=label,color=color)
axes[1].set_xticks(x,['After 2003','After 2005','After 2007']);axes[1].set_ylim(.79,.92)
axes[1].set_title('Later-year diagnostics');axes[1].legend(frameon=False)
for ax in axes:ax.spines[['top','right']].set_visible(False)
fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.pdf');plt.show()
""")
md("""## Complete new source and reproduction

The following executable cells load the complete new source into separate modules. The earlier `simple_models.py` and `simple_benchmark.py` helpers are unchanged and are included in `simple_methodical.ipynb`; the supplied CSVs and saved reference folds/predictions remain required. Use the existing repository environment or `requirements-simple.txt`.

The last cell defaults to saved results. Set `RERUN_ALL=True` to rerun all seven frozen stages, the independent implementation checks and artifact generation. The stage folders also preserve the source snapshots used at the time of each experiment. The original reference never changes during this workflow.
""")
code("""import sys,types
from IPython.core.magic import register_cell_magic
sys.path.insert(0,str(PART1))
@register_cell_magic
def source_module(name,source):
    module=types.ModuleType(name.strip());module.__file__=str(PART1/(name.strip()+'.py'))
    sys.modules[name.strip()]=module
    exec(compile(source,module.__file__,'exec'),module.__dict__)
    print('Loaded',name.strip())
""")
for name in ['trusted_models','trusted_benchmark','trusted_average','trusted_solution','trusted_finalize','trusted_audit']:
    code(f'%%source_module {name}\n'+(ROOT/f'{name}.py').read_text())
code("""RERUN_ALL=False
if RERUN_ALL:
    import trusted_benchmark,trusted_average,trusted_finalize,trusted_audit
    for path in sorted(OUT.glob('*/protocol.json')):
        protocol=json.loads(path.read_text())
        if 'specs' in protocol:
            trusted_benchmark.run(path.parent.name,protocol['specs'])
            trusted_benchmark.diagnostics(path.parent.name)
    trusted_average.main();trusted_finalize.main();trusted_audit.main()
else:
    print('Saved, independently checked results. Set RERUN_ALL=True to refit.')
""")
md("""## Sources and remaining uncertainty

The local `Midterm_Project.pdf` supplies the feature definitions, time split, metrics, and method requirements. `experiments/paired_recipe_cv/BENCHMARK.md` supplies the fixed evaluation contract. Ani's archived notebook supplies the reference architecture and comparisons; this work modifies that existing recipe. The local scikit-learn implementations provide the estimators.

The new candidate beats Ani on the primary local benchmark and the three chronological diagnostics. It loses the original validation comparison. Repeated development testing can overfit, and the latest-year Kaggle labels are unavailable. **A manual Kaggle result is still required to establish an actual leaderboard win.** The requested roughly 0.05 gain is not demonstrated by these results.
""")
nb=nbf.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')))
nbf.write(nb,ROOT/'trusted_iteration.ipynb')
print(ROOT/'trusted_iteration.ipynb')
