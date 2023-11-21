import unittest

from bbo.benchmarks.experimenters.synthetic.toy import dummy
from bbo.benchmarks.experimenters.numpy_experimenter import NumpyExperimenter
from bbo.algorithms.random import RandomDesigner
from bbo.algorithms.local_search import LocalSearchDesigner
from bbo.algorithms.regularized_evolution import RegularizedEvolutionDesigner
from bbo.utils.parameter_config import SearchSpace
from bbo.utils.metric_config import (
    ObjectiveMetricGoal,
    Objective,
)
from bbo.utils.problem_statement import ProblemStatement


class DesignerTest(unittest.TestCase):
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
        self.designers = [
            RandomDesigner(problem_statement),
            LocalSearchDesigner(problem_statement),
            RegularizedEvolutionDesigner(problem_statement),
        ]

    def test_run(self):
        for designer in self.designers:
            for _ in range(10):
                trials = designer.suggest()
                self.experimenter.evaluate(trials)
                designer.update(trials)
