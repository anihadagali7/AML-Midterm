"""Regenerate the final train-only full-fit candidate. Never reads validation labels."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np,pandas as pd
import experiment as e
from threadpoolctl import threadpool_limits
rank=json.loads(Path('experiments/fullsize-ranking.json').read_text())
selected=rank[0]
test=pd.read_csv('data/part1_test.csv');u=e.DATA.iloc[e.UNLAB].drop(columns=e.LABELS)
query=pd.concat([test,u],ignore_index=True)
with threadpool_limits(limits=1):p=e.predict(np.arange(len(e.Y)),query,selected['config'])
h=e.decisions(p,selected['rule'])
sub=pd.DataFrame({'ID':test.ID,'Predicted':[' '.join(l for l,v in zip(e.LABELS,row) if v) for row in h[:len(test)]]})
path=Path('predictions/part1_fullfit_candidate.csv');sub.to_csv(path,index=False)
probs=pd.DataFrame(p[:len(test)],columns=['p_'+l for l in e.LABELS]);probs.insert(0,'ID',test.ID);probs.to_csv('predictions/fullfit_test_probabilities.csv',index=False)
aug=e.DATA.copy();aug.loc[e.UNLAB,e.LABELS]=h[len(test):];aug['label_source']='original';aug.loc[e.UNLAB,'label_source']='fullfit_pseudo_label';aug.to_csv('predictions/part1_train_augmented_fullfit.csv',index=False)
up=pd.DataFrame(p[len(test):],columns=['p_'+l for l in e.LABELS]);up.insert(0,'ID',u.ID.to_numpy());up.to_csv('predictions/fullfit_unlabeled_probabilities.csv',index=False)
assert sub.ID.tolist()==pd.read_csv('data/part1_sample_submission.csv').ID.tolist()
assert sub.Predicted.map(lambda s:bool(s) and set(s.split())<=set(e.LABELS)).all()
assert aug[e.LABELS].isin([0,1]).all().all() and np.array_equal(aug.loc[e.LAB,e.LABELS],e.Y)
manifest=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected=selected,training_gold_rows=len(e.Y),unlabeled_rows_used=len(e.UNLAB),test_rows=len(test),prediction_strategy='Fit each member on all 157 labeled rows; unsupervised transformations and teacher clustering use the permitted training unlabeled pool. Average member probabilities with weights 0.50 / 0.25 / 0.25; threshold 0.35, fallback to highest probability if empty.',validation_status='Not evaluated again on the used external validation set. Earlier 15-fold-bagged models achieved best 0.850427; their scores do not measure this full-fit candidate.',kaggle_score=None,submission=str(path),submission_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),engine_sha256=hashlib.sha256(Path('experiment.py').read_bytes()).hexdigest(),claims='0.936306 is training-only leave-one-embedding-group-out development F1, not Kaggle F1. The winning-score objective remains unverified.')
e.save_json('experiments/final-candidate.json',manifest)
print(json.dumps(manifest,indent=2))
