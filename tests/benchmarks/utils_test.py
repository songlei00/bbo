import unittest
import random

from bbo.utils.parameter_config import SearchSpace, ScaleType
from bbo.utils.metric_config import Objective, ObjectiveMetricGoal
from bbo.utils.problem_statement import ProblemStatement
from bbo.utils.trial import Trial, MetricDict
from bbo.benchmarks.analyzers.utils import trials2df, df2trials


class UtilsTest(unittest.TestCase):
    def setUp(self):
        sp = SearchSpace()
        sp.add_float_param('float', 0, 10)
        sp.add_float_param('float_linear', 0, 10, scale_type=ScaleType.LINEAR)
        sp.add_int_param('int', 1, 10)
        sp.add_discrete_param('discrete', [0, 2, 4, 6])
        sp.add_categorical_param('categorical', ['a', 'b', 'c'])
        obj = Objective()
        obj.add_metric('obj1', ObjectiveMetricGoal.MAXIMIZE)
        obj.add_metric('obj2', ObjectiveMetricGoal.MAXIMIZE)
        self.problem_statement = ProblemStatement(sp, obj)
        n = 10
        self.trials = [Trial(sp.sample()) for _ in range(n)]
        metrics = [MetricDict({
            'obj1': i + random.uniform(-0.5, 0.5),
            'obj2': -i + random.uniform(-0.5, 0.5)
        }) for i in range(n)]
        for t, m in zip(self.trials, metrics):
            t.complete(m)

    def test_trials2df(self):
        df = trials2df(self.trials)
        trials = df2trials(df, self.problem_statement)
        self.assertEqual(self.trials, trials)