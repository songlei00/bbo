import numpy as np

from bbo.algorithms.bo import BODesigner
from bbo.benchmarks.experimenters.numpy_experimenter import NumpyExperimenter
from bbo.utils.parameter_config import SearchSpace, ScaleType
from bbo.utils.metric_config import Objective, ObjectiveMetricGoal
from bbo.utils.problem_statement import ProblemStatement


def func(x):
    return -np.sum(x*x, axis=-1, keepdims=True)

dim = 3
sp = SearchSpace()
for i in range(3):
    sp.add_float_param('float{}'.format(i), -5, 5, scale_type=ScaleType.LINEAR)
obj = Objective()
obj.add_metric('obj', ObjectiveMetricGoal.MAXIMIZE)
problem_statement = ProblemStatement(sp, obj)
exp = NumpyExperimenter(func, problem_statement)

designer = BODesigner(problem_statement)

for _ in range(50):
    trials = designer.suggest()
    exp.evaluate(trials)
    designer.update(trials)
