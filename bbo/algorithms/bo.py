import logging
from typing import List, Sequence, Optional, Union

from attrs import define, field, validators, evolve
import numpy as np
import torch
from torch import optim
import botorch
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
import gpytorch
from gpytorch.constraints import GreaterThan

from bbo.algorithms.base import Designer
from bbo.algorithms.sampling.random import RandomDesigner
from bbo.utils.converters.converter import SpecType
from bbo.utils.converters.torch_converter import GroupedFeatureTrialConverter
from bbo.utils.metric_config import ObjectiveMetricGoal
from bbo.utils.problem_statement import ProblemStatement, Objective
from bbo.utils.trial import Trial
from bbo.algorithms.bo_utils.mean_factory import mean_factory
from bbo.algorithms.bo_utils.kernel_factory import kernel_factory
from bbo.algorithms.bo_utils.acqf_factory import acqf_factory
from bbo.algorithms.evolution.nsgaii import NSGAIIDesigner
from bbo.algorithms.evolution.regularized_evolution import RegularizedEvolutionDesigner
from bbo.benchmarks.experimenters.torch_experimenter import TorchExperimenter

logger = logging.getLogger(__name__)


@define
class BODesigner(Designer):
    _problem_statement: ProblemStatement = field(
        validator=validators.instance_of(ProblemStatement)
    )
    _n_init: int = field(default=10, kw_only=True)
    _q: int = field(default=1, kw_only=True)
    _device: str = field(default='cpu', kw_only=True)

    # surrogate model configuration
    _mean_type: Optional[str] = field(
        default=None, kw_only=True,
        validator=validators.optional(validators.in_(['constant', 'mlp']))
    )
    _mean_config: Optional[dict] = field(default=None, kw_only=True)
    _kernel_type: Optional[str] = field(
        default=None, kw_only=True,
        validator=validators.optional(validators.in_(['matern52', 'mlp', 'kumar', 'mixed']))
    )
    _kernel_config: dict = field(factory=dict, kw_only=True)

    # surrogate model optimization configuration
    _mll_optimizer: str = field(
        default='l-bfgs', kw_only=True,
        validator=validators.in_(['l-bfgs', 'adam'])
    )
    _mll_lr: Optional[float] = field(default=None, kw_only=True)
    _mll_epochs: Optional[int] = field(default=None, kw_only=True)

    # acquisition function configuration
    _acqf_type: Union[str, List[str]] = field(
        default='qEI', kw_only=True,
        validator=validators.or_(
            validators.in_(['qEI', 'qUCB', 'qPI', 'qlogEI']),
            validators.deep_iterable(
                validators.in_(['qEI', 'qUCB', 'qPI', 'qlogEI'])
            ),
        )
    )
    _acqf_optimizer: str = field(
        default='l-bfgs', kw_only=True,
        validator=validators.in_(['l-bfgs', 'nsgaii', 're'])
    )
    _acqf_config: dict = field(factory=dict, kw_only=True)

    # internal attributes
    _trials: List[Trial] = field(factory=list, init=False)

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
        self._kernel_config['type2num'] = self._type2num
        
        self._device = torch.device(self._device if torch.cuda.is_available() else 'cpu')

    def create_model(self, train_X, train_Y):
        mean_module = mean_factory(self._mean_type, self._mean_config)
        covar_module = kernel_factory(self._kernel_type, self._kernel_config)
        logger.info('='*20)
        logger.info('mean_module: {}'.format(mean_module))
        logger.info('covar_module: {}'.format(covar_module))
        logger.info('='*20)
        model = SingleTaskGP(train_X, train_Y, covar_module=covar_module, mean_module=mean_module)
        model.likelihood.noise_covar.register_constraint('raw_noise', GreaterThan(1e-4))
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        model, mll = model.to(self._device), mll.to(self._device)

        return mll, model

    def optimize_model(self, mll, model, train_X, train_Y):
        if self._mll_optimizer == 'l-bfgs':
            fit_gpytorch_mll(mll)
        elif self._mll_optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self._mll_lr)
            model.train()
            model.likelihood.train()
            for _ in range(self._mll_epochs):
                optimizer.zero_grad()
                output = model(train_X)
                loss = - mll(output, train_Y.reshape(-1))
                loss.backward()
                optimizer.step()
            model.eval()
            model.likelihood.eval()
        else:
            raise NotImplementedError

    def create_acqf(self, model, train_X, train_Y):
        if isinstance(self._acqf_type, list):
            acqf = []
            for acqf_type in self._acqf_type:
                acqf_tmp = acqf_factory(acqf_type, model, train_X, train_Y)
                acqf.append(acqf_tmp)
        else:
            acqf = acqf_factory(self._acqf_type, model, train_X, train_Y)
            
        return acqf
    
    def optimize_acqf(self, acqf) -> Sequence[Trial]:
        if self._acqf_optimizer == 'l-bfgs':
            lb = torch.cat([bounds['lb'] for bounds in self._type2bounds.values()])
            ub = torch.cat([bounds['ub'] for bounds in self._type2bounds.values()])
            bounds = torch.vstack((lb, ub)).double().to(self._device)
            next_X_tensor, _ = botorch.optim.optimize.optimize_acqf(
                acqf, bounds=bounds, q=self._q, num_restarts=10, raw_samples=1024
            )
            grouped_features = dict()
            start_idx = 0
            for k, d in self._type2num.items():
                X = next_X_tensor[:, start_idx: start_idx+d]
                X = X.to('cpu').numpy()
                grouped_features[k.name] = X
                start_idx += d
            next_X = self._converter.to_trials(grouped_features)
        elif self._acqf_optimizer == 'nsgaii':
            sp = self._problem_statement.search_space
            obj = Objective()
            if isinstance(self._acqf_type, list):
                for name in self._acqf_type:
                    obj.add_metric(name, ObjectiveMetricGoal.MAXIMIZE)
            else:
                obj.add_metric(self._acqf_type, ObjectiveMetricGoal.MAXIMIZE)
            if obj.num_metrics() <= 1:
                logger.warning('NSGA-II is a multi-objective optimization algorithm, but only single objective is defined')
            nsgaii_problem_statement = ProblemStatement(sp, obj)
            nsgaii_designer = NSGAIIDesigner(
                nsgaii_problem_statement,
                pop_size=self._acqf_config.get('pop_size', 20),
                n_offsprings=self._acqf_config.get('n_offsprings', None),
            )
            def acqf_obj(x, acqf):
                if not isinstance(acqf, (tuple, list)):
                    acqf = [acqf]
                y = []
                for acqf_tmp in acqf:
                    y.append(acqf_tmp(x.unsqueeze(1)).unsqueeze(-1))
                y = torch.hstack(y)
                return y
            experimenter = TorchExperimenter(lambda x: acqf_obj(x, acqf), nsgaii_problem_statement)
            for _ in range(self._acqf_config.get('epochs', 200)):
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
        elif self._acqf_optimizer == 're':
            sp = self._problem_statement.search_space
            obj = Objective()
            obj.add_metric(self._acqf_type, ObjectiveMetricGoal.MAXIMIZE)
            re_problem_statement = ProblemStatement(sp, obj)
            re_designer = RegularizedEvolutionDesigner(re_problem_statement)
            experimenter = TorchExperimenter(
                lambda x: acqf(x.unsqueeze(1)).unsqueeze(-1), 
                re_problem_statement
            )
            for _ in range(self._acqf_config.get('epochs', 200)):
                trials = re_designer.suggest()
                experimenter.evaluate(trials)
                re_designer.update(trials)
            best_trials = re_designer.result()
            next_X = best_trials
        else:
            raise NotImplementedError

        return next_X

    def suggest(self, count: Optional[int]=None) -> Sequence[Trial]:
        if len(self._trials) < self._n_init:
            next_X = self._init_designer.suggest(count)
        else:
            count = count or 1
            features, labels = self._converter.convert(self._trials)

            train_X = []
            for k in SpecType:
                train_X.append(torch.from_numpy(features[k.name]).to(self._device))
            train_X = torch.cat(train_X, dim=-1).to(self._device)

            train_Y = []
            for _, v in labels.items():
                train_Y.append(v)
            train_Y = np.concatenate(train_Y, axis=-1)
            if train_Y.shape[-1] > 1:
                raise NotImplementedError('Unsupported for multiobjective BO')
            train_Y = (train_Y - train_Y.mean()) / (train_Y.std() + 1e-6)
            train_Y = torch.from_numpy(train_Y).double().to(self._device)
        
            mll, model = self.create_model(train_X, train_Y)
            self.optimize_model(mll, model, train_X, train_Y)
            acqf = self.create_acqf(model, train_X, train_Y)
            next_X = self.optimize_acqf(acqf)

        return next_X

    def update(self, completed: Sequence[Trial]) -> None:
        self._trials.extend(completed)
