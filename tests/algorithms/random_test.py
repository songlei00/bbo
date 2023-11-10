import unittest

from bbo.benchmarks.experimenters.synthetic.toy import dummy
from bbo.benchmarks.experimenters.numpy_experimenter import NumpyExperimenter
from bbo.algorithms.random import RandomDesigner
from bbo.utils.parameter_config import SearchSpace
from bbo.utils.metric_config import (
    ObjectiveMetricGoal,
    Objective,
)
from bbo.utils.problem_statement import ProblemStatement


class RandomDesignerTest(unittest.TestCase):
    def setUp(self):
        sp = SearchSpace()
        sp.add_float_param('float', 0, 10)
        sp.add_int_param('int', 0, 10)
        sp.add_categorical_param('categorical', ['a', 'b', 'c'])
        sp.add_discrete_param('discrete', [1, 2, 3])
        obj = Objective()
        obj.add_metric('obj', goal=ObjectiveMetricGoal.MAXIMIZE)
        problem_statement = ProblemStatement(sp, obj)

        self.experimenter = NumpyExperimenter(dummy, problem_statement)
        self.designer = RandomDesigner(problem_statement)

    def test_run(self):
        for _ in range(10):
            trials = self.designer.suggest()
            self.experimenter.evaluate(trials)
            self.designer.update(trials)
