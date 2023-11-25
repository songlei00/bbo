import unittest

from bbo.benchmarks.experimenters.synthetic.toy import dummy
from bbo.benchmarks.experimenters.numpy_experimenter import NumpyExperimenter
from bbo.algorithms.random import RandomDesigner
from bbo.algorithms.local_search import LocalSearchDesigner
from bbo.algorithms.regularized_evolution import RegularizedEvolutionDesigner
from bbo.algorithms.grid import GridSearchDesigner
from bbo.algorithms.cmaes import CMAESDesigner
from bbo.utils.parameter_config import SearchSpace, ScaleType
from bbo.utils.metric_config import (
    ObjectiveMetricGoal,
    Objective,
)
from bbo.utils.problem_statement import ProblemStatement


class DesignerTest(unittest.TestCase):
    def _create_mix_problem(self):
        sp = SearchSpace()
        sp.add_float_param('float', 0, 10)
        sp.add_int_param('int', 0, 10)
        sp.add_categorical_param('categorical', ['a', 'b', 'c'])
        sp.add_discrete_param('discrete', [1, 2, 3])
        obj = Objective()
        obj.add_metric('obj', goal=ObjectiveMetricGoal.MAXIMIZE)
        problem_statement = ProblemStatement(sp, obj)
        experimenter = NumpyExperimenter(dummy, problem_statement)
        return problem_statement, experimenter

    def _create_continue_problem(self):
        sp = SearchSpace()
        sp.add_float_param('float1', 0, 5)
        sp.add_float_param('float2', 0, 10)
        sp.add_float_param('float3', 0, 5, scale_type=ScaleType.LINEAR)
        sp.add_float_param('float4', 0, 10, scale_type=ScaleType.LINEAR)
        obj = Objective()
        obj.add_metric('obj', goal=ObjectiveMetricGoal.MAXIMIZE)
        problem_statement = ProblemStatement(sp, obj)
        experimenter = NumpyExperimenter(dummy, problem_statement)
        return problem_statement, experimenter

    def test_mix_run(self):
        problem_statement, experimenter = self._create_mix_problem()
        self.designers = [
            RandomDesigner(problem_statement),
            LocalSearchDesigner(problem_statement),
            RegularizedEvolutionDesigner(problem_statement),
            GridSearchDesigner(problem_statement),
            GridSearchDesigner(problem_statement, shuffle=True),
        ]
        for designer in self.designers:
            for _ in range(50):
                trials = designer.suggest()
                experimenter.evaluate(trials)
                designer.update(trials)

    def test_continue_run(self):
        problem_statement, experimenter = self._create_continue_problem()
        self.designers = [
            CMAESDesigner(problem_statement),
        ]
        for designer in self.designers:
            for _ in range(50):
                trials = designer.suggest()
                experimenter.evaluate(trials)
                designer.update(trials)

