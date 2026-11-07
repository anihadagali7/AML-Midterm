import ast
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score, f1_score

DATA = '../part2/'
PHASES = ['I', 'II', 'III']
REASONS = ['Safety', 'Funding', 'Efficacy', 'Recruitment', 'Other']

train = pd.read_csv(DATA + 'outcome_train.csv')
val = pd.read_csv(DATA + 'outcome_val.csv')
test = pd.read_csv(DATA + 'outcome_test.csv')
df = pd.concat([train.assign(split='train'), val.assign(split='val'), test.assign(split='test')], ignore_index=True)
i_tr, i_va, i_te = (np.flatnonzero(df.split == s) for s in ['train', 'val', 'test'])
y, phase, groups = df.label.values, df.phase.values, df.trial_id.values

print(f'train {len(i_tr)}, val {len(i_va)}, test {len(i_te)}')
display(df.iloc[np.r_[i_tr, i_va]].groupby(['split', 'phase']).label.agg(['count', 'mean']).round(3).unstack(0))
print('success rate by start year (train):')
train.groupby('start_year').label.mean().loc[2000:].round(2).to_frame().T

def parse_list(s):
    try:
        out = []
        for x in ast.literal_eval(s):
            out += ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [x]
        return [str(x) for x in out]
    except (ValueError, SyntaxError):
        return []


def semicolon_tokens(col):
    return df[col].fillna('').map(lambda s: ' '.join(sorted({t.strip() for t in s.split(';') if t.strip()})))


icd = df.icdcodes.fillna('[]').map(parse_list)
text = {
    'criteria': df.criteria.fillna(''),
    'disease': df.diseases_cleaned.fillna('[]').map(lambda s: ' '.join(parse_list(s))),
    'drug': df.drugs.fillna('[]').map(lambda s: ' '.join(parse_list(s))),
    # ICD codes at three granularities: chapter letter, 3-char category, 5-char subcategory
    'icd': icd.map(lambda L: ' '.join(sorted({c[:1] for c in L} | {c[:3] for c in L} | {c[:5] for c in L}))),
    'targets': semicolon_tokens('drug_targets'),
    'pathways': semicolon_tokens('drug_pathways'),
    'actions': semicolon_tokens('drug_actions'),
}
split_tokens = dict(tokenizer=str.split, lowercase=False, token_pattern=None)

F = {
    'criteria': TfidfVectorizer(min_df=5, ngram_range=(1, 2), sublinear_tf=True).fit_transform(text['criteria']),
    'crit_counts': CountVectorizer(min_df=5, ngram_range=(1, 2)).fit_transform(text['criteria']),
    'disease': TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True).fit_transform(text['disease']),
    'drug': TfidfVectorizer(min_df=2, sublinear_tf=True).fit_transform(text['drug']),
    'icd': TfidfVectorizer(min_df=2, **split_tokens).fit_transform(text['icd']),
    'targets': TfidfVectorizer(min_df=3, **split_tokens).fit_transform(text['targets']),
    'pathways': TfidfVectorizer(min_df=5, **split_tokens).fit_transform(text['pathways']),
    'actions': CountVectorizer(min_df=2, binary=True, **split_tokens).fit_transform(text['actions']),
    'meta': sp.csr_matrix(np.column_stack([
        df.matched_drugs.notna(), np.log1p(text['drug'].str.split().str.len()), np.log1p(text['criteria'].str.len()) / 10,
        df.phase_detail.str.contains('/'), df.phase_detail.eq('early phase 1')]).astype(float)),
}
PHASE_1HOT = sp.csr_matrix(pd.get_dummies(df.phase)[PHASES].values.astype(float))
CRIT_NORM = normalize(F['criteria'])
pd.Series({k: v.shape[1] for k, v in F.items()}, name='n_features').to_frame().T

def per_phase(fit, pred, make_model, X):
    p = np.zeros(len(pred))
    for ph in PHASES:
        a, b = fit[phase[fit] == ph], phase[pred] == ph
        p[b] = make_model().fit(X[a], y[a]).predict_proba(X[pred[b]])[:, 1]
    return p


def nb_per_phase(fit, pred):
    return per_phase(fit, pred, lambda: MultinomialNB(alpha=1.0), F['crit_counts'])


def baseline_lr(fit, pred):
    return per_phase(fit, pred, lambda: LogisticRegression(C=1.0, max_iter=3000), F['criteria'])


def pooled_lr(blocks, C):
    X = sp.hstack([F[b] for b in blocks] + [PHASE_1HOT]).tocsr()
    return lambda fit, pred: LogisticRegression(C=C, max_iter=3000).fit(X[fit], y[fit]).predict_proba(X[pred])[:, 1]


def pooled_nb(fit, pred):
    X = sp.hstack([F['crit_counts'], PHASE_1HOT]).tocsr()
    return MultinomialNB(alpha=1.0).fit(X[fit], y[fit]).predict_proba(X[pred])[:, 1]


def knn_criteria(fit, pred, k=50):
    p = np.zeros(len(pred))
    for ph in PHASES:
        a, b = fit[phase[fit] == ph], pred[phase[pred] == ph]
        S = (CRIT_NORM[b] @ CRIT_NORM[a].T).toarray()
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        w = np.take_along_axis(S, idx, 1)
        p[phase[pred] == ph] = ((w * y[a][idx]).sum(1) + 1) / (w.sum(1) + 2)
    return p


BASE_MODELS = {
    'criteria LR': pooled_lr(['criteria'], 2.0),
    'disease LR': pooled_lr(['disease'], 1.0),
    'ICD LR': pooled_lr(['icd'], 1.0),
    'drug-name LR': pooled_lr(['drug'], 1.0),
    'drug-knowledge LR': pooled_lr(['targets', 'pathways', 'actions'], 0.3),
    'metadata LR': pooled_lr(['meta'], 1.0),
    'criteria MNB': pooled_nb,
    'criteria kNN': knn_criteria,
}


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def stacked(fit, pred, return_meta=False):
    Z_fit, Z_pred = np.zeros((len(fit), len(BASE_MODELS))), np.zeros((len(pred), len(BASE_MODELS)))
    for j, fn in enumerate(BASE_MODELS.values()):
        for a, b in GroupKFold(5).split(fit, groups=groups[fit]):
            Z_fit[b, j] = fn(fit[a], fit[b])
        Z_pred[:, j] = fn(fit, pred)
    ph = lambda idx: pd.get_dummies(pd.Categorical(phase[idx], PHASES)).values.astype(float)
    meta = LogisticRegression(C=1.0, max_iter=3000).fit(np.hstack([logit(Z_fit), ph(fit)]), y[fit])
    p = meta.predict_proba(np.hstack([logit(Z_pred), ph(pred)]))[:, 1]
    return (p, meta, Z_pred) if return_meta else p


MODELS = {'Multinomial NB (per phase)': nb_per_phase, 'Baseline LR (per phase)': baseline_lr, 'Improved: stacked ensemble': stacked}

def phase_metrics(idx, p):
    rows = {}
    for ph in PHASES:
        m = phase[idx] == ph
        yt, pp = y[idx][m], p[m]
        rows[ph] = {'F1': f1_score(yt, pp >= 0.5), 'ROC-AUC': roc_auc_score(yt, pp),
                    'PR-AUC': average_precision_score(yt, pp), 'log loss': log_loss(yt, np.clip(pp, 1e-4, 1 - 1e-4))}
    out = pd.DataFrame(rows).T
    out.loc['mean'] = out.mean()
    return out


val_pred = {name: fn(i_tr, i_va) for name, fn in MODELS.items()}
val_table = pd.concat({name: phase_metrics(i_va, p) for name, p in val_pred.items()}, names=['model', 'phase'])
val_table.round(3)

i_all = np.r_[i_tr, i_va]
year = df.start_year.fillna(df.start_year.median()).values
t_fit, t_eval = i_all[year[i_all] <= 2009], i_all[year[i_all] >= 2010]
temporal = pd.DataFrame({name: phase_metrics(t_eval, fn(t_fit, t_eval)).loc['mean'] for name, fn in MODELS.items()}).T
print(f'fit n={len(t_fit)} (success rate {y[t_fit].mean():.3f}), eval n={len(t_eval)} (success rate {y[t_eval].mean():.3f})')
temporal.round(3)

reasons = pd.read_csv('part1_best_reason_labels.csv').set_index('ID')[REASONS + ['source']]
failed = df.iloc[i_va].assign(**{n: p for n, p in val_pred.items()})
failed = failed[failed.label == 0].join(reasons, on='trial_id')
print(f'failed val rows: {len(failed)} | matched to Part 1: {failed.source.notna().sum()} '
      f"({(failed.source == 'true').sum()} true labels, {(failed.source == 'predicted').sum()} predicted)")

rows = []
for ph in PHASES:
    g = failed[failed.phase == ph]
    groups_by_reason = {r: g[g[r] == 1] for r in REASONS} | {'No recorded reason': g[g.source.isna()]}
    for reason, s in groups_by_reason.items():
        rows.append({'phase': ph, 'reason': reason, 'n trials': len(s),
                     'n true labels': int((s.source == 'true').sum()), 'n predicted labels': int((s.source == 'predicted').sum()),
                     **{f'error: {n.split(":")[0].split(" (")[0]}': (s[n] >= 0.5).mean() if len(s) else np.nan
                        for n in ['Baseline LR (per phase)', 'Improved: stacked ensemble']}})
errors = pd.DataFrame(rows).set_index(['phase', 'reason'])
errors.round(3)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, ph in zip(axes, PHASES):
    e = errors.loc[ph]
    x = np.arange(len(e))
    ax.bar(x - 0.2, e['error: Baseline LR'], 0.4, label='Baseline LR')
    ax.bar(x + 0.2, e['error: Improved'], 0.4, label='Improved (stack)')
    ax.set_xticks(x, [f'{r}\n(n={n})' for r, n in zip(e.index, e['n trials'])], rotation=45, ha='right', fontsize=8)
    ax.set_title(f'Phase {ph}')
axes[0].set_ylabel('fraction of failed trials predicted successful')
axes[0].legend()
plt.tight_layout()
plt.savefig('part2_first_error_by_reason.png', dpi=150)

p_test, meta, Z_test = stacked(i_all, i_te, return_meta=True)
submission = pd.DataFrame({'ID': df.ID.values[i_te], 'label': np.clip(p_test, 1e-3, 1 - 1e-3)})
submission.to_csv('part2_first_submission.csv', index=False)

print('meta-model weights:', dict(zip(list(BASE_MODELS) + PHASES, meta.coef_[0].round(2))))
pd.DataFrame({'train+val success rate': pd.Series(y[i_all]).groupby(phase[i_all]).mean(),
              'mean predicted P(success) on test': submission.groupby(phase[i_te]).label.mean()}).round(3)