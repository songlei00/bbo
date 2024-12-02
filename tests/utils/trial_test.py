import unittest

from bbo.utils.trial import (
    Metric, MetricDict,
    ParameterValue, ParameterDict,
    Trial, is_better_than, topk_trials
)
from bbo.utils.metric_config import Objective, ObjectiveMetricGoal


class MetricDictTest(unittest.TestCase):
    def test_create(self):
        md = MetricDict({'obj1': 10})
        self.assertIsInstance(md['obj1'], Metric)
        self.assertEqual(md['obj1'].value, 10)
        self.assertEqual(md['obj1'].std, None)


class ParameterDictTest(unittest.TestCase):
    def test_create(self):
        pd = ParameterDict({'x1': 10})
        pd['x2'] = 20
        pd.update({'x3': 30})
        pd.setdefault('x4', 40)
        self.assertEqual(len(pd), 4)
        for k, v in pd.items():
            self.assertIsInstance(v, ParameterValue)


class TrialTest(unittest.TestCase):
    def test_complete(self):
        param = {'x1': 10, 'x2': 20}
        trial = Trial(param)
        m = {'obj': 0}
        trial.complete(m)
        self.assertIsInstance(trial.parameters, ParameterDict)
        self.assertIsInstance(trial.metrics, MetricDict)


class UtilFuncTest(unittest.TestCase):
    def setUp(self):
        self.min_obj = Objective()
        self.min_obj.add_metric('obj1', ObjectiveMetricGoal.MINIMIZE, -5, 5)
        self.max_obj = Objective()
        self.max_obj.add_metric('obj1', ObjectiveMetricGoal.MAXIMIZE, -5, 5)
        pds = [ParameterDict({'x1': 1}) for _ in range(10)]
        metrics = [MetricDict({'obj1': i}) for i in range(10)]
        self.trials = [Trial(pd, metrics=m) for pd, m in zip(pds, metrics)]
        
    def test_is_better_than(self):
        ret = is_better_than(self.min_obj, self.trials[0], self.trials[1])
        self.assertTrue(ret)
        ret = is_better_than(self.max_obj, self.trials[0], self.trials[1])
        self.assertFalse(ret)

    def test_topk(self):
        k = 3
        topk = topk_trials(self.min_obj, self.trials, k)
        self.assertEqual(topk, self.trials[: k])
        topk = topk_trials(self.max_obj, self.trials, k)
        self.assertEqual(topk, self.trials[-k: ])