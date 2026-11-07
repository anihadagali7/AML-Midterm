"""Reuse the frozen comparison machinery for a separate subset-averaging study."""
import json
import run_ani_combinations as task
import ani_bagging as models

r,b,e,pooled=task.r,task.b,task.e,task.pooled
PREVIOUS=task.OUT
OUT=r.ROOT/'experiments/ani_bagging_v1'
PRIOR_PROTOCOL=json.loads((PREVIOUS/'protocol.json').read_text())
ORIGINAL_REPORT=task.report


def sources():
    names=['ani_bagging.py','run_ani_bagging.py','ani_combinations.py','run_ani_combinations.py']
    return {n:b.file_hash(r.ROOT/n) for n in names}


def prior_contract():
    task.prior.guard()
    for name,h in PRIOR_PROTOCOL['source_sha256'].items():assert b.file_hash(r.ROOT/name)==h
    assert b.file_hash(PREVIOUS/'registry.json')==PRIOR_PROTOCOL['registry_sha256']
    assert json.loads((PREVIOUS/'verification.json').read_text())['passed']


def guard():
    prior_contract();p=json.loads((OUT/'protocol.json').read_text())
    assert p['source_sha256']==sources()
    assert p['registry_sha256']==b.file_hash(OUT/'registry.json')
    assert p['protected_sha256']==task.prior.protected()
    assert p['previous_confirmation_sha256']==b.file_hash(PREVIOUS/'confirmation.json')
    return p


def prepare():
    if (OUT/'protocol.json').exists():guard();print('Frozen subset-averaging protocol verified.');return
    prior_contract();rows=[]
    old=json.loads((PREVIOUS/'registry.json').read_text())
    for c in old:
        if c['name'] not in task.CONTROLS:continue
        config=dict(c['config'],model='ani_bagged',source_sha256=sources())
        rows.append(dict(c,config=config))
    for parent,short in [(task.PAIR,'75/25'),(task.TRIPLE,'70/15/15')]:
        original=next(c for c in rows if c['name']==parent)
        for full in [.5,0.]:
            label='half subset average' if full else 'full subset average'
            rows.append(dict(name=f'{short} + {label}',parent=parent,kind=label,
                config=dict(original['config'],bagging=dict(folds=5,seed=270118,full_weight=full)),rule='t0.5'))
    r.save(OUT/'registry.json',rows)
    p=dict(version='ani-subset-averaging-v1',created_at=b.now(),source_sha256=sources(),
        registry_sha256=b.file_hash(OUT/'registry.json'),protected_sha256=task.prior.protected(),
        previous_confirmation_sha256=b.file_hash(PREVIOUS/'confirmation.json'),
        hypothesis='Averaging Ani fits on different available gold subsets may reduce sensitivity to the small labeled sample. This changes fitting, not just clustering initialization.',
        recipes='Four additions: two ensemble parents x two Ani averages. Average five grouped 80%-gold Ani fits, either alone or mixed 50/50 with the full outer-training Ani fit. All native Ani settings and ensemble partner weights unchanged. Seed270118, five grouped folds. No bootstrap duplicate weighting.',
        boundaries='Subset members use only their own gold rows and original masked rows, removing both outer held-out groups and omitted subset groups. Omitted gold rows are not pseudo-labeled. All five members predict the same outer query; no query inputs fit preprocessing.',
        initial=PRIOR_PROTOCOL['initial'],controls=PRIOR_PROTOCOL['controls'],
        shortlist=PRIOR_PROTOCOL['shortlist'],confirmation=PRIOR_PROTOCOL['confirmation'],
        supplementary=PRIOR_PROTOCOL['supplementary'],control_supplementary=PRIOR_PROTOCOL['control_supplementary'],
        limitations='Follow-up motivated by failed combination round. Reuses development labels and saved splits; not independent confirmation or a Kaggle score.',
        submissions_created=False,full235_submission_refit=False,test_inputs_for_fit=False)
    r.save(OUT/'protocol.json',p);print('Frozen four subset-averaging recipes and three controls.',flush=True)


def report():
    ORIGINAL_REPORT()
    path=OUT/'report.html';body=path.read_text()
    body=body.replace('Eight additions, one at a time','Four subset-averaging additions, one at a time')
    body=body.replace('<h1>Combining improvements with the leading ensembles</h1>','<h1>Averaging Ani models trained on grouped gold subsets</h1>')
    body=body.replace('<h2>Complete initial screen</h2>',
        '<p>Each new Ani component averages five fits trained on approximately 80% of the available gold labels, with either 0% or 50% weight retained for the full fitting-set Ani model. Tri-training and SVM partner recipes and ensemble weights stay fixed. The preceding eight single-change combinations failed to improve both eight-repeat CV pools; those results remain in <a href="../ani_combination_v1/report.html">the preceding report</a>.</p><h2>Complete initial screen</h2>')
    path.write_text(body)


# Reuse the same comparison and verification functions without editing their
# frozen source or existing output. Their module globals select this new study.
task.OUT=OUT
task.models=models
task.sources=sources
task.guard=guard
task.prepare=prepare
task.report=report


if __name__=='__main__':task.main()
