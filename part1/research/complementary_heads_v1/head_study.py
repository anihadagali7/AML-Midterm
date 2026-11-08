"""Frozen, nested evaluation of supervised error-diverse kernel heads."""
import os
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[variable] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import pandas as pd
import sklearn
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from scipy.special import expit
from sklearn.metrics import f1_score, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
P1 = HERE.parents[1]
ROOT = P1.parent
HELPER = P1/'research/anchored_transductive_v1'
sys.path[:0] = [str(HERE), str(P1), str(HELPER)]
from anchored_solution import AnchoredReasonClassifier, select_reasons
from complement import ComplementaryKernel, STRENGTHS, centered_logit, decide, fit_fusion
spec = importlib.util.spec_from_file_location('_complement_benchmark', HELPER/'run.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
LABELS, EMB = benchmark.LABELS, benchmark.FEATURES
CANDIDATES = [prefix+name for prefix in ['blend_', 'fusion_'] for name in STRENGTHS]
NAMES = ['parent', 'anchored', 'calibration'] + ['head_'+n for n in STRENGTHS] + CANDIDATES


def read(path): return json.loads(Path(path).read_text())
def save(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def idhash(ids): return hashlib.sha256('\n'.join(ids).encode()).hexdigest()
def loadnp(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}


def cases():
    result = benchmark.cases()
    for case in result:
        for key in ['fit', 'query']:
            case[key] = case[key][['ID', 'start_year']+EMB+LABELS].reset_index(drop=True)
        case['reference_key'] = 'strict_inductive__'+('grouped_' if case['pool'] != 'supplementary' else '')+case['key']
    return result


def reference(case):
    z = loadnp(HERE/'parent_inputs.npz')
    keep = z['case'] == case['reference_key']
    np.testing.assert_array_equal(z['ids'][keep], case['query'].ID)
    np.testing.assert_array_equal(z['truth'][keep], case['query'][LABELS])
    return z['anchored'][keep], z['cand2'][keep]


def inner_inputs(frame, seed):
    gold = frame[frame[LABELS].ge(0).all(axis=1)].reset_index(drop=True)
    gg, fg = benchmark.groups(gold), benchmark.groups(frame)
    _, inv = np.unique(gg, return_inverse=True)
    target = np.zeros((inv.max()+1, 5), int)
    np.maximum.at(target, inv, gold[LABELS].to_numpy(int))
    # Predeclared deterministic retry only for class coverage, never for scores.
    for chosen_seed in range(seed, seed+20):
        splitter = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=chosen_seed)
        proposed = list(splitter.split(np.zeros((len(target), 1)), target))
        if all((target[tr].sum(0) > 0).all() for tr, _ in proposed): break
    else: raise ValueError('No class-complete three-fold inner split')
    for fold, (_, te) in enumerate(proposed):
        positions = np.flatnonzero(np.isin(inv, te))
        held = gold.iloc[positions].reset_index(drop=True)
        part = frame[~np.isin(fg, gg[positions])].reset_index(drop=True)
        assert not set(benchmark.groups(part)) & set(benchmark.groups(held))
        assert set(part.ID).isdisjoint(held.ID)
        yield fold, positions, part, held, chosen_seed


def anchor_predict(fit, query_x):
    return AnchoredReasonClassifier().fit(fit[EMB].to_numpy(float), fit[LABELS].to_numpy(int)).predict_proba(query_x)


def split_audit(part, held, seed, fold):
    return dict(seed=seed, fold=fold, fitting_ids_sha256=idhash(part.ID),
                query_ids_sha256=idhash(held.ID), gold=int(part[LABELS].ge(0).all(axis=1).sum()),
                masked=int(part[LABELS].eq(-100).all(axis=1).sum()), query=len(held), group_overlap=0)


def crossfit_anchor(frame):
    n = int(frame[LABELS].ge(0).all(axis=1).sum())
    scores = np.zeros((n, 5)); coverage = np.zeros(n, int); audit = []
    for fold, positions, part, held, seed in inner_inputs(frame, 270302):
        scores[positions] = anchor_predict(part, held[EMB].to_numpy(float))
        coverage[positions] += 1
        audit.append(split_audit(part, held, seed, fold))
    assert (coverage == 1).all()
    return scores, audit


def fit_pair(fit, query_x):
    """No query IDs or labels are accepted by the model interface."""
    gold = fit[fit[LABELS].ge(0).all(axis=1)].reset_index(drop=True)
    y = gold[LABELS].to_numpy(int)
    oa = np.zeros_like(y, dtype=float)
    ob = {name: np.zeros_like(oa) for name in STRENGTHS}
    coverage = np.zeros(len(gold), int); audit = []
    for fold, positions, part, held, seed in inner_inputs(fit, 270301):
        hx = held[EMB].to_numpy(float)
        oa[positions] = anchor_predict(part, hx)
        training_anchor, subaudit = crossfit_anchor(part)
        kernel = ComplementaryKernel().fit(part[EMB].to_numpy(float), part[LABELS].to_numpy(int), training_anchor)
        for name, score in kernel.predict_logits(hx).items(): ob[name][positions] = score
        coverage[positions] += 1
        item = split_audit(part, held, seed, fold)
        item.update(error_training_subfolds=subaudit, optimization=kernel.audit)
        audit.append(item)
    assert (coverage == 1).all()
    a = anchor_predict(fit, query_x)
    kernel = ComplementaryKernel().fit(fit[EMB].to_numpy(float), fit[LABELS].to_numpy(int), oa)
    b = kernel.predict_logits(query_x)
    scores = {'anchored': expit(centered_logit(a))}
    scores['calibration'], cw = fit_fusion(oa, None, y, a, None)
    weights = {'calibration': cw.tolist()}
    for name in STRENGTHS:
        scores['head_'+name] = expit(b[name])
        scores['blend_'+name] = .8*scores['anchored']+.2*expit(b[name])
        scores['fusion_'+name], w = fit_fusion(oa, ob[name], y, a, b[name])
        weights[name] = w.tolist()
    return scores, dict(inner=audit, optimization=kernel.audit, fusion_weights=weights), a


def sources():
    paths = [HERE/name for name in ['head_study.py', 'complement.py', 'check_complement.py',
             'parent_inputs.npz', 'parent_reported_result.json', 'provenance.json']]
    paths += [P1/'anchored_solution.py', P1/'TESTING_POLICY.md', HELPER/'run.py', HELPER/'models.py']
    paths += list((P1/'data').glob('*.csv'))
    paths += [P1/'experiments/current_leaders_grouped_v1/snapshots'/f'{n}.json'
              for n in ['train3', 'train5', 'pooled3', 'pooled5', 'chronology']]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def register():
    path = HERE/'protocol.json'; assert not path.exists()
    assert len(cases()) == 84
    save(path, dict(version='complementary-heads-v1', registered_at=datetime.now(timezone.utc).isoformat(),
        base_main='766d16d3bc74248cb7d8f4c1e7240094ebd5a55e',
        parent='Anchored .89636 is the architecture parent. Scored .89935 anchored80/Cand2_20 is the advancement control.',
        hypothesis='A supervised nonlinear head explicitly discouraged from sharing anchored residuals may learn useful complementary evidence; a regularized readout may use both heads better than a fixed blend.',
        adaptation='Course forbids deep learning. Frozen anchored is internally LR/KNN/LDA, not a neural network. Train classical Nyström RBF logistic head by gradients; no jointly updated anchor, gradient-reversal network, hidden-layer removal, or learned deep representation. Replace the output combination with learned readouts of all ten label-indexed pre-decision logits.',
        features='Only150 supplied reason embeddings. Anchor unchanged including its pseudo-label generation. Kernel scaler and128 Nyström RBF landmarks fit eligible gold+masked inputs, gamma1/150, seed270303. Kernel label loss uses supplied gold only; no imitation of anchor pseudo-labels.',
        head_loss='Sum across5 labels of mean binary cross-entropy +.005/2*sum(nonintercept_weights^2)+1e-6/2*sum(intercept^2) +lambda/2*sum(squared centered Pearson error correlations). Errors use centered-cutoff anchored scores sigmoid(logit(a)-logit(.3))-gold and kernel sigmoid(logit)-gold. Each variance adds1e-8. Lambda0 ordinary control, .1 and .3 diverse variants. No gradient reversal of supervised loss. Diverse fits start from ordinary solution. L-BFGS-B maxiter1000 ftol1e-11 gtol1e-7.',
        head_caveat='Anchored decision-aligned scores are not calibrated probabilities. Penalized objective is nonconvex; success here would concern this fixed adaptation only. Lower error correlation can sacrifice accuracy.',
        candidates=CANDIDATES,
        fixed_blends='80% decision-aligned anchored probabilities +20% kernel probabilities; threshold.5+argmax. This is a registered new score-space blend, not identical to the prior raw-score80/20 Cand2 blend.',
        learned_fusion='Five independent offset logistic readouts: base=logit(anchor)-logit(.3); intercept+all5 anchor centered logits+all5 kernel logits, each clipped[-6,6]/3. Sum BCE +20/2*sum(coefficients^2), including intercept; zero correction recovers anchor. Threshold.5+argmax. Calibration-only control removes kernel features. No fitted threshold or cross-validated regularization search.',
        nested_access='Outer saved84 cases. Three group-stratified inner folds seed270301 create genuine held-out scores for BOTH heads. For each inner kernel training, another three-fold anchor crossfit seed270302 supplies training residuals. Every level refits scaler/PCA/GMM/pseudo-labels/anchor, purges masked copies of held groups and excludes held gold. Deterministically retry up to20 seeds only if a fitting fold lacks a class. Full outer kernel uses first-level crossfitted anchor residuals. Final fusion fits inner scores only.13 fresh anchor fits and12 kernel fits per outer case.',
        schedule='Focused predeclared complete80 grouped cases+validation78+3chronology cases for every arm, including standalone heads and calibration as diagnostics. Supplementary exception is declared regardless of CV success.',
        scoring='Complete OOF sample-F1 per saved repeat,8means/populationSD, paired deltas/wins/ties/losses versus both controls and matched lambda0, macro/per-label F1 precision recall support, Other subgroups, heldout residual correlation and fixes/breaks.',
        advancement='Positive mean and >=5/8 repeat wins in BOTH pools versus scored .89935 control. Also report versus anchored and lambda0. Supplementary regressions visible. Prepare final235 only if qualifies and neither supplementary headline declines>.005; choose maximum minimum pool delta, then sum. No automatic upload or merge.',
        limitations='All labels/splits are reused development evidence. No fresh independent holdout, no inference of Kaggle labels. Standalone supervised head is diagnostic; combined candidate retains anchor unsupervised missing-label generation.',
        access='Strict inductive: query embeddings only at prediction; no actual Kaggle inputs loaded during benchmarking. Original validation preserves no outer group purge, matching historical protocol. Supplied input files hashed for protection.',
        reference='Fresh outer anchor must match retained scores within1e-7 and decisions exactly. Cached strict five-seed Cand2 only scores .89935 benchmark control; never used in training or new candidates.',
        retention='Keep source/frozen settings, every summary and verification. Keep useful matched ordinary/diverse fused predictions and controls; discard unsuccessful remaining caches after verification. No fitted models.',
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__),
        sources=sources()))
    print('Registered six combined candidates, three standalone diagnostics and calibration across84cases.', flush=True)


def guard(): assert read(HERE/'protocol.json')['sources'] == sources(), 'Frozen source changed'


def evaluate(case, output):
    started = time.monotonic(); fit, query = case['fit'], case['query']
    a_ref, c_ref = reference(case)
    with threadpool_limits(limits=1):
        scores, audit, a = fit_pair(fit, query[EMB].to_numpy(float))
    np.testing.assert_allclose(a, a_ref, atol=1e-7, rtol=0)
    np.testing.assert_array_equal(select_reasons(a), select_reasons(a_ref))
    scores['parent'] = expit(centered_logit(.8*a_ref+.2*c_ref))
    S = np.stack([scores[name] for name in NAMES])
    P = np.stack([decide(score) for score in S])
    assert np.isfinite(S).all() and (P.sum(2)>0).all()
    dest = Path(output)/'predictions'/(case['key']+'.npz')
    np.savez_compressed(dest, names=np.array(NAMES), ids=query.ID.to_numpy(str),
                        truth=query[LABELS].to_numpy(int), scores=S, predictions=P)
    audit.update(key=case['key'], pool=case['pool'], seed=case['seed'],
        fit_ids_sha256=idhash(fit.ID), query_ids_sha256=idhash(query.ID),
        anchor_max_error=float(np.abs(a-a_ref).max()), sha256=sha(dest), seconds=time.monotonic()-started)
    save(dest.with_suffix('.json'), audit)
    return dict(key=case['key'], seconds=round(audit['seconds'], 1))


def metric_rows(y, p):
    pr, re, f, support = precision_recall_fscore_support(y, p, zero_division=0)
    return dict(sample_f1=float(f1_score(y, p, average='samples')), macro_f1=float(f.mean())), [
        dict(label=name, precision=float(pr[j]), recall=float(re[j]), f1=float(f[j]), support=int(support[j]))
        for j, name in enumerate(LABELS)]


def collect(output):
    guard(); output=Path(output); grouped={}; audits=[]
    for case in cases():
        path=output/'predictions'/(case['key']+'.npz'); z=loadnp(path); audit=read(path.with_suffix('.json'))
        assert audit['sha256']==sha(path); audits.append(audit)
        np.testing.assert_array_equal(z['ids'],case['query'].ID)
        np.testing.assert_array_equal(z['truth'],case['query'][LABELS])
        key=(case['pool'],case['seed']) if case['pool']!='supplementary' else (case['key'],-1)
        grouped.setdefault(key,[]).append(z)
        if case['key'].startswith('through_'):grouped.setdefault(('chronology98',-1),[]).append(z)
    rows=[]; labels=[]; subgroups=[]; diversity=[]
    for (pool,seed),packets in grouped.items():
        y=np.concatenate([z['truth'] for z in packets]); ids=np.concatenate([z['ids'] for z in packets])
        assert len(ids)==len(set(ids))
        if pool in ['train','pooled']:assert len(packets)==5
        all_scores=np.concatenate([z['scores'] for z in packets],axis=1)
        all_predictions=np.concatenate([z['predictions'] for z in packets],axis=1)
        a=all_predictions[NAMES.index('anchored')]; ae=all_scores[NAMES.index('anchored')]-y
        ae-=ae.mean(0)
        for j,name in enumerate(NAMES):
            p=all_predictions[j]; m, ls=metric_rows(y,p)
            base=dict(pool=pool,seed=seed,recipe=name,n=len(y))
            rows.append(dict(**base,**m)); labels.extend(dict(**base,**label) for label in ls)
            for group,mask in [('other_alone',(y[:,4]==1)&(y.sum(1)==1)),
                ('other_with_recruitment',(y[:,4]==1)&(y[:,3]==1)),('not_other',y[:,4]==0)]:
                subgroups.append(dict(**base,subgroup=group,support=int(mask.sum()),other_predicted=int(p[mask,4].sum())))
            e=all_scores[j]-y; e-=e.mean(0)
            corr=(ae*e).mean(0)/np.sqrt((np.square(ae).mean(0)+1e-8)*(np.square(e).mean(0)+1e-8))
            for k,label in enumerate(LABELS):
                diversity.append(dict(**base,label=label,error_correlation=float(corr[k]),
                    fixes=int(((a[:,k]!=y[:,k])&(p[:,k]==y[:,k])).sum()),
                    breaks=int(((a[:,k]==y[:,k])&(p[:,k]!=y[:,k])).sum()),
                    jointly_wrong=int(((a[:,k]!=y[:,k])&(p[:,k]!=y[:,k])).sum())))
    out=output/'summary';out.mkdir(exist_ok=True);d=pd.DataFrame(rows)
    d.to_csv(out/'repeat_scores.csv',index=False)
    pd.DataFrame(labels).to_csv(out/'per_label_scores.csv',index=False)
    pd.DataFrame(subgroups).to_csv(out/'other_subgroups.csv',index=False)
    pd.DataFrame(diversity).to_csv(out/'diversity.csv',index=False)
    summary=d.groupby(['pool','recipe']).agg(mean_sample_f1=('sample_f1','mean'),
        population_sd=('sample_f1',lambda x:x.std(ddof=0)),mean_macro_f1=('macro_f1','mean')).reset_index()
    summary.to_csv(out/'overview.csv',index=False)
    pairs=[]; deltas=[]; qualifies={name:True for name in CANDIDATES}
    comparisons=[(name,control) for name in CANDIDATES for control in ['parent','anchored']]
    comparisons += [(prefix+name,prefix+'ordinary') for prefix in ['blend_','fusion_','head_'] for name in ['diverse10','diverse30']]
    comparisons += [('fusion_'+name,'calibration') for name in STRENGTHS]
    for pool in ['train','pooled']:
        pivot=d[d.pool==pool].pivot(index='seed',columns='recipe',values='sample_f1')
        for name,control in comparisons:
            v=pivot[name]-pivot[control]; wins=int((v>1e-12).sum());ties=int((v.abs()<=1e-12).sum())
            passed=bool(v.mean()>1e-12 and wins>=5)
            if control=='parent': qualifies[name]&=passed
            pairs.append(dict(pool=pool,recipe=name,control=control,mean_delta=float(v.mean()),wins=wins,ties=ties,losses=8-wins-ties,passes=passed))
            deltas.extend(dict(pool=pool,recipe=name,control=control,seed=int(k),delta=float(vv)) for k,vv in v.items())
    pd.DataFrame(pairs).to_csv(out/'paired_summary.csv',index=False)
    pd.DataFrame(deltas).to_csv(out/'paired_repeat_deltas.csv',index=False)
    save(out/'qualification.json',qualifies)
    save(out/'fit_summary.json',dict(cases=len(audits),fresh_anchored_fits=84*13,kernel_fits=84*12,
        fusion_fits=84*4,cached_cand2_control_cases=84,total_fit_seconds=sum(a['seconds'] for a in audits),
        max_anchor_error=max(a['anchor_max_error'] for a in audits)))
    opt=[]
    for audit in audits:
        for stage,entries in [('outer',audit['optimization'])]+[(f"inner{v['fold']}",v['optimization']) for v in audit['inner']]:
            for name,values in entries.items():
                opt.append(dict(case=audit['key'],stage=stage,recipe=name,iterations=values['iterations'],
                    max_gradient=values['max_gradient'],objective=values['objective'],
                    training_mean_squared_error_correlation=values['training_mean_squared_error_correlation']))
    pd.DataFrame(opt).to_csv(out/'optimization.csv',index=False)
    print(summary[summary.pool.isin(['train','pooled','validation','chronology98'])].to_string(index=False),flush=True)
    print('QUALIFIED',qualifies,flush=True)


def run(output,workers):
    guard();output=Path(output);assert not output.exists(),'Choose a fresh output directory.'
    (output/'predictions').mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures=[executor.submit(evaluate,case,str(output)) for case in cases()]
        for i,future in enumerate(as_completed(futures),1):print(json.dumps(dict(done=i,total=84,**future.result())),flush=True)
    collect(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['register','run','collect'])
    parser.add_argument('--output',type=Path,default=HERE/'results');parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.command=='register':register()
    elif args.command=='run':run(args.output,args.workers)
    else:collect(args.output)
