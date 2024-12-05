import numpy as np
from bbo.benchmarks.experimenters.synthetic.vizier_bbob import (
    Rastrigin, BuecheRastrigin, LinearSlope, AttractiveSector, StepEllipsoidal,
    RosenbrockRotated, Ellipsoidal, Discus, BentCigar, SharpRidge,
    DifferentPowers, Weierstrass, SchaffersF7, SchaffersF7IllConditioned, GriewankRosenbrock,
    Schwefel, Katsuura, Lunacek, Gallagher101Me, Gallagher21Me,
    NegativeSphere, NegativeMinDifference
)

from bbo.utils.parameter_config import SearchSpace, ScaleType
from bbo.utils.metric_config import Objective, ObjectiveMetricGoal
from bbo.utils.problem_statement import ProblemStatement
from bbo.benchmarks.experimenters.numpy_experimenter import NumpyExperimenter


funcs = {
    'Rastrigin': Rastrigin,
    'BuecheRastrigin': BuecheRastrigin,
    'LinearSlope': LinearSlope,
    'AttractiveSector': AttractiveSector,
    'StepEllipsoidal': StepEllipsoidal,
    'RosenbrockRotated': RosenbrockRotated,
    'Ellipsoidal': Ellipsoidal,
    'Discus': Discus,
    'BentCigar': BentCigar,
    'SharpRidge': SharpRidge,
    'DifferentPowers': DifferentPowers,
    'Weierstrass': Weierstrass,
    'SchaffersF7': SchaffersF7,
    'SchaffersF7IllConditioned': SchaffersF7IllConditioned,
    'GriewankRosenbrock': GriewankRosenbrock,
    'Schwefel': Schwefel,
    'Katsuura': Katsuura,
    'Lunacek': Lunacek,
    'Gallagher101Me': Gallagher101Me,
    'Gallagher21Me': Gallagher21Me,
    'NegativeSphere': NegativeSphere,
    'NegativeMinDifference': NegativeMinDifference
}

def bbob_problem(name, dim, lb, ub, seed=0):
    sp = SearchSpace()
    for i in range(dim):
        sp.add_float_param('float{}'.format(i), lb, ub, scale_type=ScaleType.LINEAR)
    obj = Objective()
    obj.add_metric('obj', goal=ObjectiveMetricGoal.MINIMIZE)
    problem_statement = ProblemStatement(sp, obj)
    func = funcs[name]
    def impl(X):
        Y = []
        for x in X:
            y = func(x.reshape(-1, 1), seed)
            Y.append(y)
        return np.array(Y).reshape(-1, 1)
    exp = NumpyExperimenter(impl, problem_statement)
    return problem_statement, exp
