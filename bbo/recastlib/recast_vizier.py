import logging
from typing import Optional, Sequence

from attrs import define, field, validators

from bbo.algorithms.base import Designer
from bbo.utils.problem_statement import ProblemStatement
from bbo.utils.parameter_config import ParameterType
from bbo.utils.metric_config import ObjectiveMetricGoal
from bbo.utils.trial import Trial, ParameterDict

logger = logging.getLogger(__file__)

try:
    from vizier import pyvizier as vz
    from vizier import algorithms as vza
    from vizier.algorithms import designers as vz_designers
    import jax
except ImportError as e:
    logger.warning('Import vizier error: {}'.format(e))


@define
class VizierDesigner(Designer):
    _problem_statement: ProblemStatement = field(
        validator=validators.instance_of(ProblemStatement)
    )
    _algorithm: str = field(validator=validators.in_([
        'vizier', 'eagle', 'cmaes'
    ]))
    _seed: int = field(default=0, validator=validators.instance_of(int), kw_only=True)

    _impl = field(init=False)

    def __attrs_post_init__(self):
        self._impl = self._algorithm_factory()

    def _create_problem(self):
        problem = vz.ProblemStatement()
        root = problem.search_space.root
        for name, pc in self._problem_statement.search_space.parameter_configs.items():
            if pc.type == ParameterType.DOUBLE:
                root.add_float_param(name, pc.bounds[0], pc.bounds[1])
            elif pc.type == ParameterType.INTEGER:
                root.add_int_param(name, pc.bounds[0], pc.bounds[1])
            elif pc.type == ParameterType.CATEGORICAL:
                root.add_categorical_param(name, pc.feasible_values)
            elif pc.type == ParameterType.DISCRETE:
                root.add_discrete_param(name, pc.feasible_values)
            else:
                raise NotImplementedError
        for name, m in self._problem_statement.objective.metric_informations.items():
            goal = vz.ObjectiveMetricGoal.MAXIMIZE if m.goal == ObjectiveMetricGoal.MAXIMIZE else vz.ObjectiveMetricGoal.MINIMIZE
            metric = vz.MetricInformation(name=name, goal=goal)
            problem.metric_information.append(metric)
        return problem

    def _algorithm_factory(self):
        problem = self._create_problem()
        rng = jax.random.PRNGKey(self._seed)
        if self._algorithm == 'vizier':
            impl = vz_designers.VizierGPBandit(problem, rng=rng)
        elif self._algorithm == 'eagle':
            impl = vz_designers.EagleStrategyDesigner(problem, seed=self._seed)
        elif self._algorithm == 'cmaes':
            impl = vz_designers.CMAESDesigner(problem, seed=self._seed)
        else:
            raise NotImplementedError
        return impl

    def _suggest(self, count: Optional[int]=None) -> Sequence[Trial]:
        vz_suggestions = self._impl.suggest()
        suggestioins = list()
        for vz_suggestion in vz_suggestions:
            parameters = ParameterDict()
            for name, param in vz_suggestion.parameters.items():
                parameters[name] = param.value
            trial = Trial(parameters)
            suggestioins.append(trial)
        return suggestioins

    def _update(self, completed: Sequence[Trial]) -> None:
        vz_completed = list()
        for c in completed:
            param_dict = {name: p.value for name, p in c.parameters.items()}
            m_dict = {name: m.value for name, m in c.metrics.items()}
            vz_c = vz.Trial(param_dict)
            vz_c.complete(vz.Measurement(metrics=m_dict))
            vz_completed.append(vz_c)
        self._impl.update(vza.CompletedTrials(vz_completed), vza.ActiveTrials())
