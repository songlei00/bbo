import logging
import enum
import copy
from typing import Sequence, Optional, List

from attrs import define, field, validators, evolve
import numpy as np
import torch
from torch import optim
from torch.quasirandom import SobolEngine
from torch.utils.data import TensorDataset, DataLoader
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
import gpytorch
from gpytorch.constraints import GreaterThan

from bbo.algorithms.base import Designer
from bbo.algorithms.sampling.random import RandomDesigner
from bbo.algorithms.evolution.pso import PSODesigner
from bbo.utils.converters.converter import SpecType, BaseTrialConverter
from bbo.utils.converters.torch_converter import GroupedFeatureTrialConverter
from bbo.utils.metric_config import ObjectiveMetricGoal
from bbo.utils.problem_statement import ProblemStatement, Objective
from bbo.utils.trial import Trial, is_better_than
from bbo.predictors.base import Predictor
from bbo.predictors.gp import GPPredictor
from bbo.predictors.gp.mean_factory import MeanFactory
from bbo.predictors.gp.kernel_factory import KernelFactory
from bbo.algorithms.bo.acqf_factory import AcqfFactory
from bbo.algorithms.evolution.nsgaii import NSGAIIDesigner
from bbo.algorithms.evolution.regularized_evolution import RegularizedEvolutionDesigner
from bbo.benchmarks.experimenters.torch_experimenter import TorchExperimenter
from bbo.utils.utils import timer_wrapper

logger = logging.getLogger(__name__)


class RunningStatus(enum.Enum):
    INIT = 'INIT'
    RUN = 'RUN'


@define
class BODesigner(Designer):
    _problem_statement: ProblemStatement = field(
        validator=validators.instance_of(ProblemStatement)
    )
    _n_init: int = field(default=10, kw_only=True)
    _q: int = field(default=1, kw_only=True)
    _device: str = field(default='cpu', kw_only=True)

    # surrogate model configuration
    _predictor: Predictor = field(factory=lambda: GPPredictor())
    _mean_factory: MeanFactory = field(default=MeanFactory('constant'), kw_only=True)
    _kernel_factory: KernelFactory = field(default=KernelFactory('matern52'), kw_only=True)

    # surrogate model optimization configuration
    _mll_optimizer: str = field(
        default='l-bfgs', kw_only=True,
        validator=validators.in_(['l-bfgs', 'adam'])
    )
    _mll_lr: Optional[float] = field(default=None, kw_only=True)
    _mll_epochs: Optional[int] = field(default=None, kw_only=True)

    # acquisition function configuration
    _acqf_factory: AcqfFactory = field(default=AcqfFactory('qlogEI'), kw_only=True)
    _acqf_optimizer_type: str = field(
        default='lbfgs', kw_only=True,
        validator=validators.in_(['random', 'adam', 'lbfgs', 'pso', 'nsgaii', 're'])
    )
    _lr: float = field(default=0.01, validator=validators.instance_of(float), converter=float, kw_only=True)
    _epochs: int = field(default=200, validator=validators.instance_of(int), kw_only=True)
    _num_restarts: int = field(default=3, validator=validators.instance_of(int), kw_only=True)
    _num_raw_samples: int = field(default=2048, validator=validators.instance_of(int), kw_only=True)

    # trust region
    _use_trust_region: bool = field(default=True, kw_only=True)
    _length_min: float = field(default=0.5**7, kw_only=True)
    _length_max: float = field(default=1.6, kw_only=True)
    _length_init: float = field(default=0.8, kw_only=True)
    _failtol: int | None = field(default=None, kw_only=True)
    _succtol: int = field(default=3, kw_only=True)
    
    # internal attributes
    _trials: List[Trial] = field(factory=list, init=False)
    _init_designer: Designer = field(init=False)
    _converter: BaseTrialConverter = field(init=False)
    _type2bounds = field(init=False)
    _type2num = field(init=False)
    _length: float = field(init=False)
    _failcnt: int = field(default=0, init=False)
    _succcnt: int = field(default=0, init=False)
    _best_suggestion: Trial = field(default=None, init=False)
    _running_status: RunningStatus = field(default=RunningStatus.INIT, init=False)

    def __attrs_post_init__(self):
        self._init_designer = RandomDesigner(self._problem_statement)
        self._converter = GroupedFeatureTrialConverter.from_problem(self._problem_statement)

        type2bounds = {k: {'lb': [], 'ub': []} for k in SpecType}
        for spec in self._converter.output_spec.values():
            if spec.type in type2bounds:
                type2bounds[spec.type]['lb'].append(spec.bounds[0])
                type2bounds[spec.type]['ub'].append(spec.bounds[1])
            else:
                raise NotImplementedError('Unsupported variable type for BO: {}'.format(spec.type))
        for k in type2bounds:
            type2bounds[k]['lb'] = torch.tensor(type2bounds[k]['lb'])
            type2bounds[k]['ub'] = torch.tensor(type2bounds[k]['ub'])
        self._type2bounds = type2bounds

        self._type2num = dict()
        for k in type2bounds:
            self._type2num[k] = len(type2bounds[k]['lb'])

        self._device = torch.device(self._device if torch.cuda.is_available() else 'cpu')
        self._failtol = self._failtol or max(4, self._problem_statement.search_space.num_parameters())
        self._length = self._length_init

    @timer_wrapper
    def create_model(self, train_X, train_Y):
        mean_module = self._mean_factory()
        covar_module = self._kernel_factory()
        model = SingleTaskGP(train_X, train_Y, covar_module=covar_module, mean_module=mean_module).to(self._device)
        model.likelihood.noise_covar.register_constraint('raw_noise', GreaterThan(1e-4))
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

        return mll, model

    @timer_wrapper
    def optimize_model(self, mll, model, train_X, train_Y):
        if self._mll_optimizer == 'l-bfgs':
            fit_gpytorch_mll(mll)
        elif self._mll_optimizer == 'adam':
            train_Y = train_Y.squeeze()
            train_dataset = TensorDataset(train_X, train_Y)
            train_dataloader = DataLoader(train_dataset, 32, True)
            optimizer = optim.Adam(model.parameters(), lr=self._mll_lr)
            model.train()
            model.likelihood.train()
            for _ in range(self._mll_epochs):
                for X, Y in train_dataloader:
                    model.set_train_data(inputs=X, targets=Y, strict=False)
                    optimizer.zero_grad()
                    output = model(X)
                    loss = - mll(output, Y)
                    loss.backward()
                    optimizer.step()
            model.set_train_data(inputs=train_X, targets=train_Y, strict=False)
            model.eval()
            model.likelihood.eval()
        else:
            raise NotImplementedError

    @timer_wrapper
    def optimize_acqf(self, model, acqf, train_X, train_Y) -> Sequence[Trial]:
        weights = model.covar_module.base_kernel.base_kernel.lengthscale.cpu().detach().numpy().ravel()
        weights = weights / weights.mean()
        weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))
        weights = torch.from_numpy(weights)
        if self._problem_statement.objective.item().goal == ObjectiveMetricGoal.MAXIMIZE:
            center_X = train_X[train_Y.argmax()]
        else:
            center_X = train_X[train_Y.argmin()]
        trust_region_lb = torch.clip(center_X - weights * self._length / 2.0, 0.0, 1.0)
        trust_region_ub = torch.clip(center_X + weights * self._length / 2.0, 0.0, 1.0)

        def create_acqf_problem():
            sp = copy.deepcopy(self._problem_statement.search_space)
            obj = Objective()
            if isinstance(self._acqf_factory.acqf_type, str):
                obj.add_metric(self._acqf_factory.acqf_type, ObjectiveMetricGoal.MAXIMIZE)
            else:
                for acqf_type in self._acqf_factory.acqf_type:
                    obj.add_metric(acqf_type, ObjectiveMetricGoal.MAXIMIZE)
            problem_statement = ProblemStatement(sp, obj)
            if self._use_trust_region:
                for i, (k, v) in enumerate(problem_statement.search_space.parameter_configs.items()):
                    lb = trust_region_lb[i].numpy().item() * (v.bounds[1] - v.bounds[0]) + v.bounds[0]
                    ub = trust_region_ub[i].numpy().item() * (v.bounds[1] - v.bounds[0]) + v.bounds[0]
                    problem_statement.search_space.parameter_configs[k] = evolve(
                        v, bounds=(lb, ub)
                    )
            def acqf_obj(x, acqf):
                if not isinstance(acqf, (tuple, list)):
                    acqf = [acqf]
                y = []
                for acqf_tmp in acqf:
                    y.append(acqf_tmp(x.unsqueeze(1).to(self._device)).unsqueeze(-1))
                y = torch.hstack(y)
                return y
            experimenter = TorchExperimenter(lambda x: acqf_obj(x, acqf), problem_statement)
            return problem_statement, experimenter

        if self._acqf_optimizer_type in ['random', 'adam', 'lbfgs']:
            if self._use_trust_region:
                lb, ub = trust_region_lb, trust_region_ub
            else:
                lb = torch.cat([bounds['lb'] for bounds in self._type2bounds.values()]).to(self._device)
                ub = torch.cat([bounds['ub'] for bounds in self._type2bounds.values()]).to(self._device)
            seed = np.random.randint(int(5e5))
            sobol = SobolEngine(len(lb), scramble=True, seed=seed) 
            Xraw = sobol.draw(self._num_raw_samples).to(self._device)
            Xraw = Xraw * (ub - lb) + lb
            Xraw = Xraw.unsqueeze(1)
            Yraw = acqf(Xraw)
            indices = torch.argsort(Yraw)[-self._num_restarts: ]
            cand_X = Xraw[indices]
            cand_X.requires_grad_(True)

            if self._acqf_optimizer_type == 'random':
                pass
            elif self._acqf_optimizer_type == 'adam':
                optimizer = torch.optim.Adam([cand_X], lr=self._lr)
                for i in range(self._epochs):
                    optimizer.zero_grad()
                    loss = - acqf(cand_X).mean()
                    loss.backward()
                    optimizer.step()
                    for j, (l, u) in enumerate(zip(lb, ub)):
                        cand_X.data[..., j].clamp_(l, u)
            elif self._acqf_optimizer_type == 'lbfgs':
                optimizer = torch.optim.LBFGS([cand_X], lr=self._lr)
                def closure():
                    optimizer.zero_grad()
                    loss = - acqf(cand_X).mean()
                    loss.backward()
                    return loss
                for i in range(self._epochs):
                    optimizer.step(closure)
                    for j, (l, u) in enumerate(zip(lb, ub)):
                        cand_X.data[..., j].clamp_(l, u)
            else:
                raise NotImplementedError('Unsupported acqf optimizer')

            cand_Y = acqf(cand_X)
            cand_X = cand_X[cand_Y.argmax()]
            next_X_np = cand_X.detach().to('cpu').numpy()
            grouped_features = dict()
            start_idx = 0
            for k, d in self._type2num.items():
                X = next_X_np[:, start_idx: start_idx+d]
                grouped_features[k.name] = X
                start_idx += d
            next_X = self._converter.to_trials(grouped_features)
        elif self._acqf_optimizer_type == 'pso':
            problem_statement, experimenter = create_acqf_problem()
            best_trial = None
            for _ in range(self._num_restarts):
                pso_designer = PSODesigner(problem_statement)
                for _ in range(self._epochs):
                    trials = pso_designer.suggest()
                    experimenter.evaluate(trials)
                    pso_designer.update(trials)
                cand_best_trial = pso_designer.result()[0]
                if best_trial is None or is_better_than(problem_statement.objective, cand_best_trial, best_trial):
                    best_trial = cand_best_trial
            next_X = [best_trial]
        elif self._acqf_optimizer_type == 'nsgaii':
            problem_statement, experimenter = create_acqf_problem()
            if problem_statement.objective.num_metrics() <= 1:
                logger.warning('NSGA-II is a multi-objective optimization algorithm, but only single objective is defined')
            nsgaii_designer = NSGAIIDesigner(problem_statement, pop_size=20, n_offsprings=None)
            
            for _ in range(self._epochs):
                trials = nsgaii_designer.suggest()
                experimenter.evaluate(trials)
                nsgaii_designer.update(trials)

            # generate next_X for batch BO setting
            pareto_trials = nsgaii_designer.result()
            pareto_trials = [evolve(i, metrics=None) for i in pareto_trials]
            pop_trials = nsgaii_designer.curr_pop()
            pop_trials = [evolve(i, metrics=None) for i in pop_trials]
            diff_trials = [x for x in pop_trials if x not in pareto_trials]
            next_X = []

            if len(pareto_trials) >= self._q:
                idx = np.random.choice(len(pareto_trials), self._q, replace=False)
                next_X.extend([pareto_trials[i] for i in idx])
            else:
                next_X.extend(pareto_trials)
                if len(diff_trials) > 0:
                    quota = min(len(diff_trials), self._q-len(pareto_trials))
                    idx = np.random.choice(len(diff_trials), quota, replace=False)
                    next_X.extend([diff_trials[i] for i in idx])
                quota = self._q - np.vstack(next_X).shape[0]
                if quota > 0:
                    trials = self._init_designer.suggest(quota)
                    next_X.extend(trials)
        elif self._acqf_optimizer_type == 're':
            problem_statement, experimenter = create_acqf_problem()
            best_trial = None
            for _ in range(self._num_restarts):
                re_designer = RegularizedEvolutionDesigner(problem_statement)
                for _ in range(self._epochs):
                    trials = re_designer.suggest()
                    experimenter.evaluate(trials)
                    re_designer.update(trials)
                cand_best_trial = re_designer.result()[0]
                if best_trial is None or is_better_than(problem_statement.objective, cand_best_trial, best_trial):
                    best_trial = cand_best_trial
            next_X = [best_trial]
        else:
            raise NotImplementedError

        return next_X

    @timer_wrapper
    def _suggest(self, count: Optional[int]=None) -> Sequence[Trial]:
        if len(self._trials) < self._n_init:
            self._running_status = RunningStatus.INIT
            next_X = self._init_designer.suggest(count)
        else:
            self._running_status = RunningStatus.RUN
            count = count or 1
            features, labels = self._converter.convert(self._trials)

            train_X = []
            for k in SpecType:
                train_X.append(torch.from_numpy(features[k.name]))
            train_X = torch.cat(train_X, dim=-1).to(self._device)

            train_Y = []
            metric_informations = self._problem_statement.objective.metric_informations
            for v, metric_info in zip(labels.values(), metric_informations.values()):
                if metric_info.goal == ObjectiveMetricGoal.MINIMIZE:
                    v = -v
                train_Y.append(v)
            train_Y = np.concatenate(train_Y, axis=-1)
            if train_Y.shape[-1] > 1:
                raise NotImplementedError('Unsupported for multiobjective BO')
            train_Y = (train_Y - train_Y.mean()) / (train_Y.std() + 1e-6)
            train_Y = torch.from_numpy(train_Y).double().to(self._device)
        
            mll, model = self.create_model(train_X, train_Y)
            self.optimize_model(mll, model, train_X, train_Y)
            acqf = self._acqf_factory(model, train_X, train_Y)
            next_X = self.optimize_acqf(model, acqf, train_X, train_Y)

        return next_X

    def _update(self, completed: Sequence[Trial]) -> None:
        self._trials.extend(completed)
        if self._running_status == RunningStatus.RUN:
            self._adjust_length(completed)

        for trial in completed:
            if self._best_suggestion is None or \
                is_better_than(self._problem_statement.objective, trial, self._best_suggestion):
                self._best_suggestion = trial

        if self._length < self._length_min:
            self._restart()

    def _restart(self):
        logger.info('epoch {}, restart'.format(self._base_epoch))
        self._trials.clear()
        self._failcnt = 0
        self._succcnt = 0
        self._length = self._length_init
        self._best_suggestion = None
        self._running_status = RunningStatus.INIT

    def _adjust_length(self, completed):
        curr_best_trial = None
        for trial in completed:
            if curr_best_trial is None or is_better_than(self._problem_statement, trial, curr_best_trial):
                curr_best_trial = trial
                
        if is_better_than(self._problem_statement.objective, curr_best_trial, self._best_suggestion):
            self._succcnt += 1
            self._failcnt = 0
        else:
            self._succcnt = 0
            self._failcnt += 1
        if self._succcnt == self._succtol:
            self._length = min([2.0*self._length, self._length_max])
        elif self._failcnt == self._failtol:
            self._length /= 2.0
            self._failcnt = 0