"""Rescore the fixed retrospective panel; no model fitting or metric selection."""
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import precision_recall_fscore_support, f1_score

HERE = Path(__file__).resolve().parent
PART = HERE.parents[1]
LABELS = ['Safety', 'Funding', 'Efficacy', 'Recruitment', 'Other']
NBOOT, SEED = 5000, 20261107


def save(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, default=lambda v: v.item()) + '\n')


def sign(v):
    return 0 if abs(v) < 1e-12 else int(np.sign(v))


def ranking(panel, metric):
    correct = wrong = tied = kaggle_ties = 0
    for a, b in combinations(panel.index, 2):
        actual = sign(panel.loc[a, 'kaggle'] - panel.loc[b, 'kaggle'])
        predicted = sign(panel.loc[a, metric] - panel.loc[b, metric])
        if not actual:
            kaggle_ties += 1
        elif not predicted:
            tied += 1
        elif actual == predicted:
            correct += 1
        else:
            wrong += 1
    return dict(n=len(panel), spearman=float(spearmanr(panel.kaggle, panel[metric]).statistic),
                kendall_tau_b=float(kendalltau(panel.kaggle, panel[metric]).statistic),
                correctly_ordered=correct, reversed=wrong, benchmark_ties=tied,
                kaggle_ties_excluded=kaggle_ties, comparable_pairs=correct + wrong + tied,
                top_by_benchmark='|'.join(panel.index[panel[metric] >= panel[metric].max() - 1e-12]))


def main():
    comp = pd.read_csv(HERE / 'comparison.csv').set_index('model')
    models = list(comp.index)
    metrics = list(comp.columns.drop('kaggle'))
    rf = pd.read_csv(HERE / 'per_trial_f1.csv.gz')
    cases = json.loads((HERE / 'splits.json').read_text())
    full = pd.concat([pd.read_csv(PART / 'data/part1_train.csv'),
                      pd.read_csv(PART / 'data/part1_val.csv')], ignore_index=True).set_index('ID')
    gold = full[full[LABELS].ge(0).all(axis=1)]
    group_hash = pd.util.hash_pandas_object(gold[[f'emb_{i}' for i in range(150)]].round(10), index=False)
    group_pos = pd.Series(pd.factorize(group_hash)[0], index=gold.index)
    ngroup = group_pos.nunique()
    rng = np.random.default_rng(SEED)
    weights = rng.multinomial(ngroup, np.full(ngroup, 1 / ngroup), size=NBOOT)

    panels = {'all_six': comp, 'exclude_transductive_ani': comp.drop('original_ani'),
              'top_four': comp.loc[models[2:]]}
    panels.update({f'leave_out_{name}': comp.drop(name) for name in models})
    rankings = [dict(panel=name, metric=m, **ranking(panel, m))
                for name, panel in panels.items() for m in metrics]
    pd.DataFrame(rankings).to_csv(HERE / 'rank_agreement.csv', index=False)
    pairs = []
    for a, b in combinations(models, 2):
        for metric in metrics:
            pairs.append(dict(first=a, second=b, metric=metric,
                              kaggle_delta=comp.loc[b, 'kaggle'] - comp.loc[a, 'kaggle'],
                              benchmark_delta=comp.loc[b, metric] - comp.loc[a, metric],
                              kaggle_order=sign(comp.loc[b, 'kaggle'] - comp.loc[a, 'kaggle']),
                              benchmark_order=sign(comp.loc[b, metric] - comp.loc[a, metric])))
    pd.DataFrame(pairs).to_csv(HERE / 'pairwise_orderings.csv', index=False)

    # Each observed embedding group gets one shared bootstrap multiplicity across
    # every appearance in A/B, repeated CV, all metrics and all model recipes.
    # Fits stay fixed: these intervals do not include retraining/search uncertainty.
    def boot_component(frame):
        perrow = frame.groupby(['ID', 'model']).f1.mean().unstack('model')[models]
        counts = np.bincount(group_pos.loc[perrow.index], minlength=ngroup)
        sums = np.zeros((ngroup, len(models)))
        np.add.at(sums, group_pos.loc[perrow.index].to_numpy(), perrow.to_numpy())
        denominator = weights @ counts
        assert (denominator > 0).all()
        return (weights @ sums) / denominator[:, None]

    b = {}
    for metric, track, kind in [('dylan_A', 'dylan', 'A'), ('grouped_train', 'canonical', 'train'),
                               ('grouped_pooled', 'canonical', 'pooled'), ('validation', 'canonical', 'validation'),
                               ('chronology98', 'canonical', 'chronology')]:
        b[metric] = boot_component(rf[(rf.track == track) & (rf.kind == kind)])
    for metric, track in [('dylan_B', 'dylan'), ('strict_B', 'strict_B')]:
        bycut = [boot_component(rf[(rf.track == track) & (rf.cut == cut)]) for cut in (2004, 2005, 2006)]
        b[metric] = sum(bycut) / 3
        if metric == 'dylan_B':
            b['dylan_B2005'] = bycut[1]
    b['dylan_J'] = (b['dylan_A'] + b['dylan_B']) / 2
    b['aligned_J'] = (b['grouped_pooled'] + b['strict_B']) / 2
    b['current_J_diagnostic'] = (b['grouped_pooled'] + b['chronology98']) / 2
    intervals, deltas = [], []
    for metric in metrics:
        for i, model in enumerate(models):
            lo, hi = np.quantile(b[metric][:, i], [.025, .975])
            intervals.append(dict(model=model, metric=metric, estimate=comp.loc[model, metric],
                                  bootstrap_se=b[metric][:, i].std(ddof=1), ci_low=lo, ci_high=hi))
        for i, j in combinations(range(len(models)), 2):
            d = b[metric][:, j] - b[metric][:, i]
            lo, hi = np.quantile(d, [.025, .975])
            deltas.append(dict(first=models[i], second=models[j], metric=metric,
                               delta=comp.loc[models[j], metric] - comp.loc[models[i], metric],
                               bootstrap_se=d.std(ddof=1), ci_low=lo, ci_high=hi,
                               bootstrap_positive_fraction=(d > 1e-12).mean()))
    pd.DataFrame(intervals).to_csv(HERE / 'conditional_intervals.csv', index=False)
    pd.DataFrame(deltas).to_csv(HERE / 'paired_differences.csv', index=False)
    save('bootstrap_protocol.json', dict(replicates=NBOOT, seed=SEED, groups=int(ngroup), rows=len(gold),
         unit='Embedding group, shared multiplicities across models, repeats and overlapping cuts',
         estimates='Per-row average across repeated fits; B is equal mean of cut-specific weighted means',
         limitations='Conditional on fitted predictions. Not a test of Kaggle gaps; excludes retraining, search and unseen-distribution uncertainty. No bootstrap draws retained.'))

    # Preserve the native SE approximation for comparison, without treating A/B as independent.
    native_se = []
    for model in models:
        f = rf[rf.model == model]
        a = f[(f.track == 'dylan') & (f.kind == 'A')].groupby('ID').f1.mean()
        a_se = a.std(ddof=1) / np.sqrt(len(a))
        bs = [g.f1.std(ddof=1) / np.sqrt(len(g))
              for _, g in f[(f.track == 'dylan') & (f.kind == 'B')].groupby('cut')]
        native_se.append(dict(model=model, A_se=a_se, B_se=np.mean(bs),
                              J_se_independence_approx=.5 * np.hypot(a_se, np.mean(bs)),
                              J_se_paired_group_bootstrap=b['dylan_J'][:, models.index(model)].std(ddof=1)))
    pd.DataFrame(native_se).to_csv(HERE / 'native_se_comparison.csv', index=False)

    # Aggregate precision/recall/F1 within each complete OOF repeat (not folds).
    aggregate = []
    def model_metrics(chosen, model):
        ys, ps = [], []
        for case in chosen:
            with np.load(HERE / 'results' / f'{case["key"]}.npz') as z:
                ys.append(z['truth']); ps.append(z[model + '_predictions'])
        y, p = np.concatenate(ys), np.concatenate(ps)
        pr, re, f1, support = precision_recall_fscore_support(y, p, zero_division=0)
        return dict(sample_f1=f1_score(y, p, average='samples', zero_division=0), macro_f1=float(f1.mean()),
                    **{f'{key}_{label}': float(v[j]) for key, v in [('precision', pr), ('recall', re), ('f1', f1), ('support', support)] for j, label in enumerate(LABELS)})
    for model in models:
        for track, kind in [('dylan', 'A'), ('dylan', 'B'), ('strict_B', 'B'), ('canonical', 'train'),
                            ('canonical', 'pooled'), ('canonical', 'validation'), ('canonical', 'chronology')]:
            chosen = [c for c in cases if c['track'] == track and c['kind'] == kind]
            split_key = 'repeat' if kind in ['A', 'train', 'pooled'] else 'cut' if kind == 'B' else None
            chunks = [[c for c in chosen if c[split_key] == r] for r in sorted({c[split_key] for c in chosen})] if split_key else [chosen]
            summaries = pd.DataFrame([model_metrics(chunk, model) for chunk in chunks])
            aggregate.append(dict(model=model, track=track, kind=kind, blocks=len(chunks),
                                  repeat_sd_population=summaries.sample_f1.std(ddof=0) if split_key == 'repeat' else None,
                                  **summaries.mean().to_dict()))
    pd.DataFrame(aggregate).to_csv(HERE / 'aggregate_label_metrics.csv', index=False)
    repeat = pd.read_csv(HERE / 'repeat_scores.csv')
    repeat_comparisons = []
    for kind in ['A', 'train', 'pooled']:
        tab = repeat[repeat.kind == kind].pivot(index='repeat', columns='model', values='sample_f1')
        for model in models:
            d = tab[model] - tab.anchored
            repeat_comparisons.append(dict(model=model, reference='anchored', kind=kind, repeats=len(d),
                                           mean_delta=d.mean(), delta_sd_population=d.std(ddof=0),
                                           wins=(d > 1e-12).sum(), ties=(abs(d) <= 1e-12).sum(), losses=(d < -1e-12).sum()))
    pd.DataFrame(repeat_comparisons).to_csv(HERE / 'paired_repeat_summary.csv', index=False)

    audit = []
    for c in cases:
        audit.append(dict(case=c['key'], track=c['track'], kind=c['kind'], cut=c.get('cut'),
                          gold=len(c['train_gold']), masked=len(c['masked_ids']), query=len(c['query_ids']),
                          gold_group_overlap=c['gold_group_overlap'], masked_group_overlap=c['masked_group_overlap'],
                          masked_at_or_after_query_start=c['future_masked']))
    pd.DataFrame(audit).to_csv(HERE / 'boundary_audit.csv', index=False)
    A = [c for c in cases if c['track'] == 'dylan' and c['kind'] == 'A']
    B = [c for c in cases if c['track'] == 'dylan' and c['kind'] == 'B']
    save('boundary_summary.json', dict(
        dylan_A_folds=len(A), A_folds_with_gold_group_overlap=sum(c['gold_group_overlap'] > 0 for c in A),
        A_folds_with_masked_group_overlap=sum(c['masked_group_overlap'] > 0 for c in A),
        B_query_occurrences=sum(len(c['query_ids']) for c in B),
        B_unique_queries=len(set(v for c in B for v in c['query_ids'])),
        B_cuts=[row for row in audit if row['kind'] == 'B']))
    print(pd.DataFrame(rankings).query("panel == 'all_six'").to_string(index=False))


if __name__ == '__main__':
    main()
