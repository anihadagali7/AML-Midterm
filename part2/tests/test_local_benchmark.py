"""Meaningful protocol/scoring regression checks; no model training required."""
import tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark_protocol import DIRECTORY,initialize,resolve
from experiments import load_data,metrics
from local_benchmark import paired_bootstrap,summarize


class FrozenProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.universe,cls.protocol=initialize()

    def test_no_trial_or_test_set_leakage(self):
        test_trials=set(load_data()['test'].trial_id)
        for split in self.protocol['splits']:
            train,ev=resolve(self.universe,split)
            self.assertFalse(set(train.trial_id)&set(ev.trial_id))
            self.assertFalse((set(train.trial_id)|set(ev.trial_id))&test_trials)

    def test_cv_covers_each_trial_once_per_repeat(self):
        expected=set(self.universe.ID)
        for repeat in [0,1]:
            ids=[i for s in self.protocol['splits'] if s['kind']=='group_cv' and s['repeat']==repeat for i in s['eval_ids']]
            self.assertEqual(len(ids),len(expected));self.assertEqual(set(ids),expected)

    def test_temporal_windows_are_forward_and_disjoint(self):
        seen=set()
        for split in self.protocol['splits']:
            if split['kind']!='temporal':continue
            train,ev=resolve(self.universe,split)
            self.assertLess(train.start_year.max(),ev.start_year.min())
            self.assertFalse(seen&set(ev.ID));seen.update(ev.ID)

    def test_protocol_reuse_is_identical(self):
        path=DIRECTORY/'protocol.json';before=path.read_bytes()
        initialize();self.assertEqual(before,path.read_bytes())


class ScoringTests(unittest.TestCase):
    def test_equal_phase_loss_differs_from_row_weighted_loss(self):
        rows=[];prob=[]
        for phase,n,correct in [('I',2,.9),('II',4,.5),('III',6,.2)]:
            for j in range(n):
                y=j%2;rows.append({'phase':phase,'label':y});prob.append(correct if y else 1-correct)
        result=metrics(pd.DataFrame(rows),np.array(prob))
        expected=np.mean([-np.log(.9),-np.log(.5),-np.log(.2)])
        self.assertAlmostEqual(result['phase_mean_logloss'],expected)
        self.assertNotAlmostEqual(result['logloss'],expected)

    def test_trial_paired_bootstrap_known_improvement(self):
        protocol={'splits':[{'name':'fake','kind':'group_cv','repeat':0}]}
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);folder=out/'predictions/fake';folder.mkdir(parents=True)
            rows=[]
            for trial in range(40):
                for phase in ['I','II','III']:
                    y=trial%2
                    rows.append({'ID':f'{trial}_{phase}','trial_id':str(trial),'phase':phase,'label':y,'probability':.8 if y else .2})
            baseline=pd.DataFrame(rows);candidate=baseline.copy()
            candidate['probability']=np.where(candidate.label,.9,.1)
            baseline.to_csv(folder/'ani_stack.csv',index=False);candidate.to_csv(folder/'candidate.csv',index=False)
            got=paired_bootstrap(out,protocol,'candidate','group_cv',repetitions=100)
            expected=-np.log(.9)+np.log(.8)
            self.assertAlmostEqual(got['group_cv_delta'],expected)
            self.assertAlmostEqual(got['group_cv_delta_lower'],expected)
            self.assertAlmostEqual(got['group_cv_delta_upper'],expected)

    def test_partial_comparison_cannot_be_marked_complete(self):
        _,protocol=initialize()
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            pd.DataFrame([{'model':'ani_stack','split':'official','kind':'official','repeat':None,'phase_mean_logloss':.5}]).to_csv(out/'scores_official.csv',index=False)
            summarize(out,protocol,bootstrap=False)
            self.assertFalse(pd.read_csv(out/'summary.csv').complete_comparison.iloc[0])


if __name__=='__main__':unittest.main()
