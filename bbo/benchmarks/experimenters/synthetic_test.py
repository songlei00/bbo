import pytest
import numpy as np

from bbo.algorithms.converters.core import TrialToArrayConverter
from bbo.benchmarks.experimenters import synthetic


def assert_optimal(func: synthetic.SyntheticFunction):
    for optimal_point in func.optimal_points:
        assert np.allclose(func(np.atleast_2d(optimal_point)), func.optimal_value)


def assert_random_sample(func: synthetic.SyntheticFunction, n: int = 10):
    lb, ub = zip(*func.bounds)
    X = np.random.rand(n, func.dim) * (np.array(ub) - np.array(lb)) + np.array(lb)
    Y = func(X)
    if func.goal.is_maximize:
        assert np.all(Y <= func.optimal_value)
    else:
        assert np.all(Y >= func.optimal_value)


def assert_problem_statement(func: synthetic.SyntheticFunction):
    exp = synthetic.SyntheticExperimenter(func)
    converter = TrialToArrayConverter.from_study_config(exp.problem_statement(), scale=False)
    trials = converter.to_trials(np.asarray(func.optimal_points))
    exp.evaluate(trials)
    y = converter.to_labels(trials)
    assert np.allclose(y, func.optimal_value, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('func', [
    synthetic.Branin2D(),
    synthetic.Ackley(5),
    synthetic.Ackley(10)
])
def test_impl(func):
    assert_optimal(func)
    assert_random_sample(func)
    assert_problem_statement(func)

