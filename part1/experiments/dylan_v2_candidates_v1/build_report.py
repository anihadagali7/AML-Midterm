"""Generate the comparison report from the independently checked results."""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
NAMES = {'anchored':'Anchored (submitted leader)', 'cand1':'Cand1: balanced LDA + graph labels',
         'cand2':'Cand2: LDA + cluster/neighbor features', 'cand4':'Cand4: LR/LDA/graph ensemble',
         'cand5':'Cand5: LR + graph labels + prior adjustment',
         'anchored_cand1':'Anchored + cand1', 'anchored_cand5':'Anchored + cand5',
         'anchored_cand1_cand5':'Anchored + cand1 + cand5'}


def table(frame, digits=5):
    return '<div class="scroll">' + frame.to_html(index=False, border=0, float_format=lambda x:f'{x:.{digits}f}') + '</div>'


def named(frame):
    result = frame.copy()
    result.model = result.model.map(NAMES)
    return result


def main():
    ours = pd.read_csv(HERE / 'strict_inductive_comparison.csv')
    native = pd.read_csv(HERE / 'native_dylan_comparison.csv')
    repeats = pd.read_csv(HERE / 'paired_repeat_summary.csv')
    intervals = pd.read_csv(HERE / 'paired_intervals.csv')
    verify = json.loads((HERE / 'verification.json').read_text())
    boundaries = json.loads((HERE / 'boundary_checks.json').read_text())
    assert verify['status'] == boundaries['status'] == 'passed'
    five_ours = pd.read_csv(HERE / 'five_seed/strict_inductive_comparison.csv')
    five_native = pd.read_csv(HERE / 'five_seed/native_dylan_comparison.csv')
    five_verify = json.loads((HERE / 'five_seed/verification.json').read_text())
    builder_checks = json.loads((HERE / 'five_seed/source_builder_checks.json').read_text())
    assert five_verify['status'] == builder_checks['status'] == 'passed'
    five_table = named(five_ours[['model','train','pooled','validation','chronology']].merge(five_native[['model','J']],on='model'))
    five_table.columns = ['Recipe','Our Train157','Our Pooled235','Our validation','Our chronology','Native five-seed J']
    five_index = five_ours.set_index('model')
    five_below = five_index.drop(index='anchored').lt(five_index.loc['anchored'],axis='columns').all().all()
    five_conclusion = ('Averaging five inner seeds leaves every candidate and blend below anchored on all four current-policy means.' if five_below else
                       'The table below records the complete five-seed result; interpret it separately from the published single-seed reproduction.')
    our_table = named(ours[['model','train','pooled','validation','chronology']])
    our_table.columns = ['Recipe', 'Train157 CV', 'Pooled235 CV', 'Validation78', 'Chronology98']
    native_table = named(native[['model','A','B','validation','J']])
    native_table.columns = ['Recipe', 'A: random CV', 'B: chronology', 'C: validation', 'J = ½A + ½B']
    pair = repeats[(repeats.access == 'strict_inductive') & (repeats.model != 'anchored')].copy()
    pair['W / T / L'] = pair.wins.astype(str)+' / '+pair.ties.astype(str)+' / '+pair.losses.astype(str)
    pair = named(pair[['model','kind','mean_delta','W / T / L']])
    pair.columns = ['Recipe','Pool','Mean difference vs anchored','Repeat wins / ties / losses']
    selected = intervals[((intervals.access == 'strict_inductive') & (intervals.metric == 'pooled') & intervals.model.isin(['cand1','cand2','anchored_cand1'])) |
                         ((intervals.access == 'native_dylan') & (intervals.metric == 'J') & intervals.model.isin(['cand2','anchored_cand1']))]
    selected = named(selected[['model','access','metric','delta','conditional_ci_low','conditional_ci_high']])
    selected.columns = ['Recipe','Evaluation','Metric','Difference vs anchored','Conditional 95% lower','Conditional 95% upper']
    definitions = pd.DataFrame([
        ['cand1','Balanced shrinkage LDA; Correct & Smooth graph pseudo-labels; temperature calibration; expected-F1 decoding. Dylan’s one-SE selection.'],
        ['cand2','Balanced LDA with K-means(8) soft features and kNN-vote(20) features; different fixed graph and pseudo-label weights. Highest native J among the four candidates.'],
        ['cand4','Equal LR, calibrated LDA and direct Correct & Smooth predictions; SLD prior correction (strength 100); expected-F1 decoding.'],
        ['cand5','Powerset LR C=0.1; Correct & Smooth pseudo-labels; SLD prior correction (strength 25); expected-F1 decoding. Dylan’s recommended robustness hedge.'],
        ['Blends','Three supplied equal-weight blends: anchored+cand1, anchored+cand5, and anchored+cand1+cand5. Average label scores, then the fixed 0.3 cutoff plus top-label fallback.']
    ], columns=['Model','Fixed definition'])
    failures = dict(model_fitting_failures=0, verification_failures=0,
                    negative_finding='All seven candidate/blend means are below anchored on all four current-policy checks; none meets the paired advancement rule.',
                    version_note='The repository environment differs slightly from supplied dependency versions; all 28 native candidate/blend aggregate checks nevertheless reproduce to floating-point tolerance.',
                    artifact_scope='Published seed0 benchmark reproduced; separately evaluated the fixed five-inner-seed averaging procedure on every context. Existing final CSV artifacts were not regenerated or submitted.',
                    five_seed_conclusion=five_conclusion,
                    five_seed_context='Supplementary check declared after seeing single-seed results; all candidates repeated without tuning or selection.')
    (HERE/'failure_notes.json').write_text(json.dumps(failures,indent=2)+'\n')
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Dylan v2 candidates on both benchmarks</title>
<style>
body{{font:16px/1.6 system-ui,sans-serif;margin:0;background:#f2f6f7;color:#19323c}}
main{{max-width:1180px;margin:auto;padding:42px 30px 72px;background:white}}
h1{{font-size:36px;line-height:1.2}}h2{{margin-top:36px;font-size:24px}}p{{max-width:1000px}}
.verdict{{background:#eaf6f2;border-left:5px solid #16816f;padding:12px 22px}}.note{{font-size:14px;color:#536b75}}
a{{color:#086f79}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:16px 0}}
th,td{{padding:10px 12px;border-bottom:1px solid #d7e1e6;text-align:right;vertical-align:top}}
th:first-child,td:first-child{{text-align:left}}th{{background:#edf3f5}}li{{margin:8px 0}}code{{font-size:13px}}
@media print{{body,main{{padding:8px;background:white}}}}
</style><main>
<p class="note">Part 1 · four fixed standalone candidates · three supplied fixed blends · anchored control</p>
<h1>How do Dylan’s new models score?</h1>
<div class="verdict"><p><strong>None improves on anchored under our current benchmark.</strong>
All four standalone candidates and all three blends score lower on Train157 CV, Pooled235 CV, original validation and chronology.
Several do score higher on Dylan’s J, because they improve his chronological test while giving up random-CV performance.
That is a difference in evaluation setup and model behavior, not an observed Kaggle gain.</p></div>

<h2>What was tested</h2>
{table(definitions)}
<p>These are executable v2 pipelines, not the earlier v1 proposal sketches. The package’s cand3 is identical to cand1 and was not duplicated.
The two files under its <code>reference/</code> directory are explicitly described as lacking the assignment’s unsupervised-label-filling step; they are documented in
<a href="candidate_definitions.md">the supplied candidate inventory</a>, but are not additional proposed candidates in this panel.</p>
<p>We freshly fit all four candidates and anchored on every evaluation context. The three blends reuse those per-context predictions with their supplied weights.
There are 84 contexts under our policy and 54 under Dylan’s setup: 138 contexts and 1,104 recipe evaluations including blends.
No classifier parameter, pseudo-label weight, graph setting, decoder, or blend weight was tuned.</p>

<h2>Our benchmark, with our fitting rules</h2>
{table(our_table)}
<p class="note">Sample-wise F1; higher is better. Each CV pool uses all eight saved repeats × five grouped folds.
Validation is the separate 157→78 fit. Chronology pools the three fixed windows, totaling 98 held-out rows.
The supplied pipelines require the documented inductive adapters below to satisfy these rules.</p>
<p>Cand2 is the strongest standalone model on the two CV pools, but it remains behind anchored by <strong>0.00876</strong> on Train157 and <strong>0.01777</strong> on Pooled235.
Anchored+cand1 is the closest blend on Pooled235, behind by <strong>0.00488</strong>.
The three-way blend is closest on Train157, but its Pooled235, validation and chronology scores all regress.</p>
<p>All four standalone candidates lose to anchored on <strong>every one of the eight Pooled235 repeats</strong>.
No candidate or blend has a positive mean difference in either pool, and none passes the requirement for at least five wins in both pools.
The smaller blend gaps are uncertain; the result is a failure to demonstrate improvement, not proof of inferiority on every possible future sample.</p>

<h2>Dylan’s original benchmark, reproduced</h2>
{table(native_table)}
<p>Dylan A is ten repeats of ungrouped, label-combination-stratified five-fold CV on 235 labels.
B averages the overlapping 2004, 2005 and 2006 chronological holdouts, while keeping all 1,014 masked training rows available.
J weights A and B equally. C is the original validation check.</p>
<p><strong>Cand2 is best on native J: 0.89879 versus anchored’s 0.88573.</strong>
Its gain comes from B (+0.04298), while A falls by 0.01688 and validation falls by 0.01923.
Cand1 and cand5 also trade lower A/C for higher B/J. Cand4 loses to anchored on all four native metrics.
Anchored is best on A and C across this complete panel.</p>
<p>All 28 supplied A/B/C/J checks for the four candidates and three blends reproduced, with maximum absolute difference
{verify['native_reproduction_max_abs_error']:.2g}. These are fresh measurements, not copied score cells.
See <a href="native_reproduction.csv">the exact reproduction check</a>.</p>

<h2>Five-seed averaging used by the proposed final CSVs</h2>
<p>Dylan’s published benchmark uses inner seed0, but his final candidate builder averages seeds 0–4.
After completing the primary comparison, we separately froze and tested that averaging procedure for every candidate on every context, with no settings changed and no candidates selected out.</p>
{table(five_table)}
<p><strong>{five_conclusion}</strong> The averages combine raw class-combination probabilities and source priors before applying the decoder and prior correction, just like Dylan’s final builder.
The native validation case matches that unchanged builder exactly for all four candidates, including its raw probabilities, headed probabilities and labels.
Anchored is deterministic here; its already-fitted control predictions were reused, not counted as fresh fits.</p>
<p class="note">These are five inner seeds per outer training split, not five new independent datasets or five extra outer-CV repeats.
The native five-seed J column is newly measured and is distinct from Dylan’s published single-seed J above.
The strict version still needs the same fitting-boundary adapters. No full-data submission CSV was regenerated.</p>
<p><a href="five_seed/protocol.json">Supplementary protocol</a> · <a href="five_seed/paired_repeat_summary.csv">Paired repeat results</a> ·
<a href="five_seed/paired_intervals.csv">Paired intervals</a> · <a href="five_seed/source_builder_checks.json">Source-builder verification</a></p>

<h2>Why the fitting rules need explicit labels</h2>
<p>The native runs use Dylan’s unchanged model source. By default, label-free preprocessing and graphs can use all 1,561 rows, including held-out and actual Kaggle test features.
SLD estimates the target label frequencies from the query batch. Anchored retains its original inductive access, matching Dylan’s supplied comparison.
Therefore his native table does not give every model the same feature access.</p>
<p>The strict table uses the same classifier and head settings, but enforces the repository’s fitting boundaries:</p>
<ul>
<li>Each case contains only eligible fitting rows plus its query rows. Test rows and excluded validation-source rows are absent. Query labels are masked before fitting.</li>
<li>Scaling, PCA, clustering, graph nodes and pseudo-labels use only the saved fitting populations. Group-matching masked rows are purged in grouped CV and chronology; chronology keeps only earlier-year fitting rows.</li>
<li>Temperature calibration refits the complete uncalibrated pipeline within grouped inner folds. The source instead keeps pseudo-labels fixed while cross-validating only the final classifier.</li>
<li>Correct & Smooth residuals use grouped inner folds with freshly fitted preprocessing. Synthetic kNN training features leave out the whole embedding group, including masked duplicates.</li>
<li>Cand2’s soft K-means feature scale uses only fitting rows. The source’s scale uses a median over every global row.</li>
<li>SLD estimates its target frequencies on the eligible masked training pool, following Dylan’s inductive convention. It does not adapt to the query batch.</li>
</ul>
<p>These are versioned evaluation adapters, not claims that the original submitted CSVs had these boundaries.
Because the tables differ in splits, population sizes, feature access and inner fitting rules, the change between tables cannot be attributed to one factor alone.
Native transductive behavior may be useful in a separately authorized setting; it has not beaten anchored under our current contract.</p>

<h2>Repeat stability and uncertainty</h2>
{table(pair)}
{table(selected)}
<p>The intervals resample 222 embedding groups 5,000 times with shared weights across models, repeats and overlapping cuts.
They are conditional on the fitted predictions; they do not include the earlier candidate search, new refitting randomness or an unknown future distribution.
In particular, cand2’s native J advantage has an interval crossing zero. These reused labels do not provide an untouched confirmation of the search-selected candidates.</p>

<h2>How new are the ideas?</h2>
<p>The fixed combinations and the Correct & Smooth implementation are concrete additions worth measuring.
LDA, neighbors, clustering, ensembling, prior adjustment and expected-F1 decoding are not all new to this workspace: earlier
<a href="../../research/new_directions/index.html">New Directions studies</a> already include expected-F1 compositional decoding and source-prior corrections.
Those older attempts used different recipes and comparators, so this run supplies the direct comparison with the current anchored leader.</p>

<h2>What these results establish</h2>
<p><strong>Keep anchored as the current choice.</strong> The new models are useful alternatives and diagnostic controls, but none earns promotion on our benchmark.
Their better native J scores are real, reproducible results under Dylan’s setup; they are not recorded Kaggle results.
No candidate or blend has a recorded Kaggle score in our ledger, and this study generated no new submission.</p>
<p>The primary comparison uses <strong>one inner seed, 0</strong>, matching Dylan’s evaluation harness; the supplementary comparison uses the final builder’s <strong>five-seed average</strong>.
Both were evaluated on the saved benchmark populations; existing final candidate CSVs were not regenerated.
The original scored anchored artifact remains 0.89636; no local score here replaces that recorded Kaggle number.</p>

<h2>Verification and replay</h2>
<p>All 1,104 prediction sets were independently rescored, including each expected-F1 decoder and fixed blend.
Anchored exactly reproduced the retained reference predictions in all 138 contexts.
All {verify['protected_source_hashes']} protected hashes passed. The strict run recorded {verify['nested_context_audits']:,} nested boundary checks with zero group overlaps.</p>
<p>The supplementary five-seed results add another 1,104 independently rescored prediction sets, with reused anchored controls and
{five_verify['nested_context_audits']:,} strict nested boundary checks. Its frozen inputs also passed verification.</p>
<p>Additional fresh-fit probes changed held-out labels, changed one query embedding, and changed inner-calibration held-out labels.
The relevant predictions and fitted temperatures remained identical where independence was required.
See <a href="boundary_checks.json">boundary checks</a> and <a href="verification.json">verification results</a>.</p>
<p>The original source snapshot is unchanged under <code>source/p1lib/</code>; strict adjustments live in
<a href="strict_adapter.py">strict_adapter.py</a>. <a href="protocol.json">The frozen protocol</a> records the complete settings, access rules, source hashes and negative-result policy.
Read <a href="README.md">README.md</a> for replay commands. Compact matched-control predictions are retained for this small fixed cross-protocol comparison;
no fitted models, transformed feature matrices or duplicated datasets are saved.</p>
<p>Files: <a href="strict_inductive_comparison.csv">our comparison</a>, <a href="native_dylan_comparison.csv">Dylan comparison</a>,
<a href="aggregate_metrics.csv">per-label metrics</a>, <a href="block_metrics.csv">repeat/window metrics</a>,
<a href="paired_repeat_summary.csv">paired repeat results</a>, <a href="paired_intervals.csv">paired intervals</a>,
<a href="failure_notes.json">negative findings and limitations</a>.</p>
</main></html>'''
    (HERE/'report.html').write_text(html)
    print(HERE/'report.html')


if __name__ == '__main__':
    main()
