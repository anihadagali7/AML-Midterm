"""Build a review notebook with complete executable source snapshots."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("""# Part 1: simple, methodical experiments

**Result:** remove PCA/GMM from Ani's recipe and use three self-training passes: **0.913191 sample F1**, versus **0.908865** for the frozen training-only Ani reference and **0.888316** for DBSCAN. This is a small development-fold improvement, **not a demonstrated Kaggle improvement**. Later-year checks do not confirm it, so the historical reference remains in place. The roughly 0.05 gain we wanted has **not** been achieved.

The selected implementation is [simple_solution.py](simple_solution.py). The separate [candidate CSV](experiments/simple_methodical/candidate/part1_simple_submission.csv) contains 312 checked predictions and has not been uploaded. It has no Kaggle score.

## Reasoning from the PDF and features

`Midterm_Project.pdf`, pages 4–5, says the 150 embeddings encode the actual stopping-reason text, are PCA-reduced and randomly rotated, and have small magnitudes. They directly describe the target. Eligibility criteria, diseases and drug biology describe the trial indirectly; matched drug features are absent for about 44% of training/validation rows. Start year identifies distribution shift but is not the meaning of a stopping reason. We first test simple models of the reason embeddings. IDs are identifiers only. No online data, external labels, reverse engineering of test labels, or deep learning is used.

There are 235 known trials: 216 have one reason and 19 have two. Label counts: Safety 25, Funding 12, Efficacy 31, Recruitment 151, Other 35. Global exploratory inspection found 82 labeled trials with a near-identical labeled embedding neighbor, all with agreeing labels. That motivated local methods; those inspection statistics are not fitted predictors.

## Unchanged benchmark

We reuse the exact eight saved repeats of five folds: 188 verified fitting rows, 47 held-out rows, and 1,014 originally unlabeled training rows. Every learned scaler, PCA, clusterer, graph and classifier excludes held-out and Kaggle test inputs. Pool all 235 out-of-fold predictions within each repeat, calculate sample F1, then average eight repeat scores. Also retain macro/per-reason F1, repeat SD and paired comparisons.

These are repeatedly used **development folds**, including the former validation labels. They are not an untouched test estimate. There are 127 named evaluations, including repeated controls and nearby settings; all positive and negative results are retained. Repeated folds overlap and are not independent samples.

## Ordered experiments

1. Nearest neighbors, logistic regression, joint logistic regression, shrinkage LDA and RBF SVM. Compare coordinate standardization with a single scale preserving rotation, plus small normalization/PCA controls.
2. Unlabeled structure: label spreading, direct K-means labels, and one cautious pseudo-labeling pass. The first graph attempt was aborted before scoring because isolated unlabeled components produce undefined probabilities; the recorded replacement uses a nearest-verified-neighbor fallback.
3. Preserve observed label combinations, or use smooth RBF regression boundaries. Neither beats Ani.
4. Test pseudo-label weights and passes. Remove Ani's PCA/GMM while retaining its joint logistic model and cosine neighbors. This gives the first improvement.
5. Downweight repeated reason texts and try balanced binary class weights. Neither beats Ani.
6. Change one setting at a time around the simplified recipe. A third pass helps; most other changes hurt.
7. Check more passes and neighbor choices, then freeze the three-pass recipe before later-year diagnostics. Five/ten passes and alternative neighbors do not improve it.

The final model does not combine all these experiments. The best pure logistic model scores 0.899876; independent-logistic self-training reaches 0.904947. The simplest tested alternatives do not explain a 0.05 leaderboard gap.
""")
code("""from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import display
PART1 = Path.cwd()
if not (PART1 / 'data/part1_train.csv').exists():
    PART1 = PART1 / 'part1'
assert (PART1 / 'data/part1_train.csv').exists()
OUT = PART1 / 'experiments/simple_methodical'
results = pd.read_csv(OUT / 'all_results.csv')
names = ['ani_train_only','dbscan_train_only','logistic_scalar_C1_t0.3',
         'selftrain_C1_w1_r1','joint_standard_knn0.3_r2','final_control_r3']
comparison = results[results.name.isin(names)].drop_duplicates('name').set_index('name')
display(comparison[['mean_f1','repeat_sd','macro_f1','delta_ani','repeat_wins_ani',
                    'fold_wins_ani','fold_ties_ani']].sort_values('mean_f1', ascending=False))
""")
md("""## Selected recipe

Standardize the embeddings using the permitted training feature pool. Fit one multinomial logistic regression (`C=0.3`, `tol=1e-4`, maximum 3,000 iterations). A two-reason trial contributes half an example to each reason; inverse reason-frequency weights balance the expanded targets. Blend 70% logistic scores with 30% scores from ten cosine neighbors, weighted by squared positive similarity. Neighbor targets divide their mass across a trial's verified or generated reasons.

For each of three passes, accept originally unlabeled trials only when the highest blended score is at least 0.8. Rebuild the augmented set from verified labels plus currently accepted rows. Finally select scores at least 0.3, with an argmax fallback. These are class-proportion scores, not calibrated independent binary probabilities.

There is no PCA, GMM, DBSCAN, graph or auxiliary-feature model in the selected recipe. Full fitting uses 235 verified labels and accepts 691, 774 and 807 unlabeled rows across the three passes. It never fits on the 312 test rows.
""")
code("""display(comparison[[f'f1_{r}' for r in ['Safety','Funding','Efficacy','Recruitment','Other']]])
display(pd.read_csv(OUT / 'verification/paired_trial_bootstrap.csv'))
""")
md("""The selected model wins 20 of 40 folds, ties 11 and loses 9 against Ani; it wins six of eight repeat averages. Its +0.004326 difference has a descriptive trial-bootstrap interval of approximately **[-0.000709, +0.009894]**, including zero. Each trial's eight predictions stay together, but the interval does not adjust for model selection or correlated training fits. It does not establish a leaderboard improvement.

“Other” remains the main weakness (F1 about 0.606). Gains mostly come from Safety and Recruitment; Funding and Efficacy do not improve.

## Checks after freezing the candidate

The original 157/78 split has already influenced historical development; it is not new validation. Chronological windows train only on known and originally unlabeled trials starting at or before 2003, 2005 or 2007. Later known trials are held out; later unknown features are excluded too. These supplement, rather than replace, the fixed benchmark.
""")
code("""diagnostics = pd.read_csv(OUT / 'verification/diagnostic_scores.csv')
display(diagnostics[['split','name','verified_fit','unlabeled_fit','held','sample_f1','macro_f1']])
display(diagnostics.pivot(index='split', columns='name', values='sample_f1'))
""")
md("""On the old 78-row validation set the three-pass candidate scores 0.884615 versus Ani's 0.891026. It ties Ani after 2003 and trails it after 2005 and 2007. The final window has only 37 known trials. These results support keeping the current reference; the CSV is an optional manual experiment, not a confirmed upgrade.

## Reproduction and files

The independent `simple_solution.py` implementation reproduces **all 40 selected-model fold probability arrays**, using `atol=1e-10, rtol=1e-8`. An independent refit of Ani using the original notebook recipe reproduces its first saved fold. Model reload gives exactly identical test scores. The candidate has 312 unique IDs in sample-submission order, with nonempty valid reasons.

`candidate/generated_labels.csv` distinguishes the 807 accepted fitting rows and their actual fitting targets from predictions for all 1,014 unlabeled trials. `candidate/part1_train_augmented.csv` preserves every original verified label. The PDF requires an unsupervised label-generation experiment: the evaluated K-means k=40 baseline has its own `candidate/kmeans_comparison_train_augmented.csv`; those labels are **not** the selected model's targets.

Each stage saves settings, input/fold checksums, scores and out-of-fold predictions. See [all results](experiments/simple_methodical/all_results.csv), the [candidate manifest](experiments/simple_methodical/candidate/manifest.json), and [checks](experiments/simple_methodical/verification/checks.json). Part 2 inputs, original data, old models, submissions and benchmark references were not replaced. Nothing was uploaded or committed/pushed.
""")
code("""manifest = json.loads((OUT / 'candidate/manifest.json').read_text())
display({k:manifest[k] for k in ['known_labels','final_self_training_counts','test_rows',
                               'independent_fold_replay_count','reload_exact','kaggle_score','versions']})
checks = pd.read_csv(OUT / 'candidate/independent_40_fold_replay.csv')
assert len(checks) == 40 and checks.max_abs_error.max() < 1e-8
display(checks.agg({'max_abs_error':['max']}))
""")
code("""# Figure source and standalone export.
import matplotlib.pyplot as plt
labels = ['DBSCAN reference','Ani reference','Simplified 2 passes','Simplified 3 passes']
models = ['dbscan_train_only','ani_train_only','joint_standard_knn0.3_r2','final_control_r3']
values = comparison.loc[models, 'mean_f1'].to_numpy()
fig, ax = plt.subplots(figsize=(7, 3))
ax.barh(labels[::-1], values[::-1], color=['#167d8d','#5ca4a9','#697386','#9ca3af'])
ax.set_xlim(.86, .94)
ax.set_xlabel('Mean sample F1 on fixed development folds')
for i, value in enumerate(values[::-1]):
    ax.text(value + .001, i, f'{value:.4f}', va='center')
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / 'comparison.png', dpi=180)
fig.savefig(OUT / 'comparison.pdf')
plt.show()
""")
md("""## Complete executable source

The next cells contain all new experiment, verification and final-model source. A small cell magic loads each into its own module. Ordinary `.py` files provide the same code for command-line use. Existing data and reference folds/predictions remain required.

The final cell defaults to displaying saved results. Set `RERUN_ALL = True` to rebuild all stages and checks from frozen settings; this takes several minutes. Use the repository environment or `requirements-simple.txt`.
""")
code("""import sys
import types
from IPython.core.magic import register_cell_magic
@register_cell_magic
def source_module(name, source):
    module = types.ModuleType(name.strip())
    module.__file__ = str(PART1 / (name.strip() + '.py'))
    sys.modules[name.strip()] = module
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    print(f'Loaded {name.strip()} ({len(source.splitlines())} source lines)')
""")
for name in ['simple_models', 'simple_benchmark', 'simple_solution', 'simple_verify', 'simple_prepare']:
    code(f'%%source_module {name}\n' + (ROOT / f'{name}.py').read_text())
code("""RERUN_ALL = False
if RERUN_ALL:
    import simple_benchmark, simple_verify, simple_prepare
    for path in sorted(OUT.glob('*/protocol.json')):
        protocol = json.loads(path.read_text())
        if 'specs' in protocol and not (path.parent / 'aborted.json').exists():
            simple_benchmark.run(path.parent.name, protocol['specs'])
    simple_verify.main()
    simple_prepare.main()
else:
    print('Saved, independently checked results. Set RERUN_ALL=True to refit.')
""")
md("""## Local sources

- `Midterm_Project.pdf`, pages 2, 4–5 and 6–7: task, features, temporal split, metrics and unsupervised-method requirement.
- `experiments/paired_recipe_cv/BENCHMARK.md` and saved split CSVs: unchanged evaluation contract.
- `paired_recipe_cv.ipynb` and its archived Ani recipe: reference and joint-logistic/cosine-neighbor method. This study simplifies that existing recipe; it does not claim to invent it.
- scikit-learn implementations of logistic regression, nearest neighbors, LDA, SVC, kernel ridge regression, K-means, DBSCAN and label spreading.

The higher class scores motivated this work; they are not evidence that these changes can reproduce them.
""")
notebook = nbf.v4.new_notebook(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3')))
nbf.write(notebook, ROOT / 'simple_methodical.ipynb')
print(ROOT / 'simple_methodical.ipynb')
