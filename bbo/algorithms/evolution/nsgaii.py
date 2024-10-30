from typing import Optional, Sequence

from attrs import define, field, validators
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2 as pymoo_NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem
from pymoo.core.evaluator import Evaluator

from bbo.algorithms.base import Designer
from bbo.utils.converters.converter import DefaultTrialConverter
from bbo.utils.problem_statement import ProblemStatement
from bbo.utils.trial import Trial
from bbo.utils.metric_config import ObjectiveMetricGoal


@define
class NSGAIIDesigner(Designer):
    _problem_statement: ProblemStatement = field(
        validator=validators.instance_of(ProblemStatement)
    )
    _pop_size: int = field(default=20)
    _n_offsprings: Optional[int] = field(default=None)

    # internal attributes
    _last_pop = field(default=None, init=False)

    def __attrs_post_init__(self):
        self._converter = DefaultTrialConverter.from_problem(self._problem_statement)
        self._impl = pymoo_NSGA2(
            pop_size=self._pop_size,
            n_offsprings=self._n_offsprings or self._pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(eta=15, prob=0.9),
            mutation=PM(eta=20),
        )
        n_var = self._problem_statement.search_space.num_parameters()
        n_obj = self._problem_statement.objective.num_metrics()
        lb, ub = [], []
        for spec in self._converter.output_spec.values():
            lb.append(spec.bounds[0])
            ub.append(spec.bounds[1])
        lb, ub = np.array(lb), np.array(ub)
        self._nsga_problem = Problem(n_var=n_var, n_obj=n_obj, n_constr=0, xl=lb, xu=ub)
        termination = NoTermination()
        self._impl.setup(self._nsga_problem, termination=termination)

    def suggest(self, count: Optional[int]=None) -> Sequence[Trial]:
        pop = self._impl.ask()
        X = pop.get('X')
        features = dict()
        for i, name in enumerate(self._converter.input_converter_dict):
            features[name] = X[:, i]
        trials = self._converter.to_trials(features)
        self._last_pop = pop
        return trials

    def update(self, completed: Sequence[Trial]) -> None:
        labels = self._converter.to_labels(completed)
        F = []
        metric_informations = self._problem_statement.objective.metric_informations
        for label, metric_info in zip(labels.values(), metric_informations.values()):
            if metric_info.goal == ObjectiveMetricGoal.MAXIMIZE:
                label = - label
            F.append(label)
        F = np.concatenate(F, axis=-1)
        static = StaticProblem(self._nsga_problem, F=F)
        Evaluator().eval(static, self._last_pop)
        self._impl.tell(infills=self._last_pop)

    def result(self):
        res = self._impl.result()
        return np.atleast_2d(res.X), np.atleast_2d(res.F)
    
    def curr_pop(self):
        pop = self._impl.pop
        return pop.get('X'), pop.get('F')