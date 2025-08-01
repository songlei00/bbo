import copy
from typing import List, Optional, Sequence, Union, Callable

from attrs import define, field, validators
import numpy as np
import torch

from bbo.algorithms.abstractions import Designer, CompletedTrials, ActiveTrials, Surrogate
from bbo.algorithms.designers import RandomDesigner, RegularizedEvolutionDesigner1
from bbo.algorithms.converters.core import TrialConverter
from bbo.algorithms.converters.torch_converter import TrialToTorchConverter
from bbo.algorithms.surrogates.gp.gp import GP
from bbo.algorithms.designers.bo.acquisitions import AcquisitionFunction, EI
from bbo.algorithms.optimizers.base import GradientFreeOptimizer
from bbo.algorithms.optimizers.designer_optimizer import DesignerAsOptimizer
from bbo.shared.trial import Trial
from bbo.shared.parameter_config import ParameterType
from bbo.shared.base_study_config import ProblemStatement, MetricInformation, ObjectiveMetricGoal
from bbo.utils import logging_utils

logger = logging_utils.get_logger(__name__)

def default_surrogate_factory(train_X: torch.Tensor, train_Y: torch.Tensor):
    return GP(train_X, train_Y)

def default_acquisition_optimizer_factory(
    seed_or_rng: Optional[Union[int, np.random.Generator]] = None
) -> GradientFreeOptimizer:
    return DesignerAsOptimizer(
        designer_factory=lambda ps: RegularizedEvolutionDesigner1(ps, seed_or_rng=seed_or_rng),
        batch_size=1,
        num_evaluations=100
    )


@define
class BODesigner(Designer):
    _problem_statement: ProblemStatement = field(validator=validators.instance_of(ProblemStatement))
    _num_init: int = field(default=10, converter=int, validator=validators.instance_of(int), kw_only=True)
    _device: str = field(default='cpu', kw_only=True)
    _np_seed_or_rng: Optional[Union[int, np.random.Generator]] = field(
        default=None,
        validator=validators.optional(validators.instance_of((np.random.Generator, int))),
        kw_only=True
    )

    # GP config
    _surrogate_factory: Callable[[torch.Tensor, torch.Tensor], Surrogate] = field(
        default=default_surrogate_factory,
        kw_only=True
    )

    # Acquisition config
    _acquisition_function_factory: Callable[[torch.Tensor], AcquisitionFunction] = field(
        default=lambda Y: EI(Y.max()),
        kw_only=True
    )
    _acquisition_optimizer_factory: Callable[[], GradientFreeOptimizer] = field(
        default=default_acquisition_optimizer_factory,
        kw_only=True
    )

    # Internal attributes
    _acquisition_problem: ProblemStatement = field(validator=validators.instance_of(ProblemStatement), init=False)
    _acquisition_optimizer: GradientFreeOptimizer = field(validator=validators.instance_of(GradientFreeOptimizer), init=False)
    _init_designer: Designer = field(validator=validators.instance_of(Designer), init=False)
    _converter: TrialConverter = field(validator=validators.instance_of(TrialConverter), init=False)
    _trials: List[Trial] = field(factory=list, init=False)
    _metadata_ns: str = field(default='bo', init=False)

    def __attrs_post_init__(self):
        if self._problem_statement.search_space.num_parameters(ParameterType.DOUBLE) != \
            self._problem_statement.search_space.num_parameters():
            raise ValueError("BO Designer only supports double parameters now")
        if not self._problem_statement.is_single_objective:
            raise ValueError("BO Designer only supports single objective now")
        
        self._np_rng = np.random.default_rng(self._np_seed_or_rng)
        self._init_designer = RandomDesigner(self._problem_statement, self._np_rng)
        self._converter = TrialToTorchConverter.from_study_config(
            self._problem_statement,
            flip_sign_for_minimization_metrics=True
        )
        if self._device.startswith('cuda') and not torch.cuda.is_available():
            logger.warning(f"Device {self._device} is not available. Use CPU")
        self._device = torch.device(self._device if torch.cuda.is_available() else 'cpu')

        # Acquisition config
        self._acquisition_problem = copy.deepcopy(self._problem_statement)
        self._acquisition_problem.metric_information = [
            MetricInformation('acquisition', ObjectiveMetricGoal.MAXIMIZE)
        ]
        self._acquisition_optimizer = self._acquisition_optimizer_factory(self._np_rng)

    def suggest(self, count: Optional[int] = None) -> Sequence[Trial]:
        count = count or 1
        if len(self._trials) < self._num_init:
            return self._init_designer.suggest(count)
        else:
            # Prepare training data
            X, Y = self._converter.to_xy(self._trials)
            X, Y = self._transform_X(X), self._transform_Y(Y)
            
            # Train surrogate model
            surrogate = self._surrogate_factory(X, Y).to(self._device)
            surrogate.train()

            # Acquisition optimization
            acqf_fn = self._acquisition_function_factory(Y)
            def score_fn(trials: Sequence[Trial]):
                X = self._converter.to_features(trials)
                X = self._transform_X(X).unsqueeze(-2)
                mu, var = surrogate.predict(X)
                acqf_values = acqf_fn(mu, var.sqrt())
                return {self._acquisition_problem.metric_information_item().name: acqf_values.detach().cpu()}

            best_trials = self._acquisition_optimizer.optimize(score_fn, self._acquisition_problem)
            return best_trials

    def update(self, completed_trials: CompletedTrials, active_trials: Optional[ActiveTrials] = None):
        self._trials.extend(completed_trials.trials)

    def _transform_X(self, X):
        return X.double.to(self._device)

    def _transform_Y(self, Y: torch.Tensor):
        scaled_Y = (Y - Y.mean(dim=0)) / (Y.std(dim=0) + 1e-6)
        return scaled_Y.to(self._device)