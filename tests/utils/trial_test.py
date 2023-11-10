import unittest

from bbo.utils.trial import (
    Metric,
    MetricDict,
    ParameterValue,
    ParameterDict,
    Trial,
)


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
