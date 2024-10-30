import logging
from typing import List, Sequence, Optional, Union

from attrs import define, field, validators
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
from bbo.utils.converters.converter import SpecType, DefaultTrialConverter
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
            validators.deep_iterable(validators.in_(['qEI', 'qUCB', 'qPI', 'qlogEI'])),
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
        self._converter = DefaultTrialConverter.from_problem(self._problem_statement, merge_by_type=True)

        type2bounds = {
            SpecType.DOUBLE: {'lb': [], 'ub': []},
            SpecType.CATEGORICAL: {'lb': [], 'ub': []}
        }
        for spec in self._converter.output_spec.values():
            if spec.type in type2bounds:
                type2bounds[spec.type]['lb'].append(spec.bounds[0])
                type2bounds[spec.type]['ub'].append(spec.bounds[1])
            else:
                raise NotImplementedError('Unsupported variable type for BO: {}'.format(spec.type))
        for k in type2bounds:
            type2bounds[k]['lb'] = torch.tensor(type2bounds[k]['lb'])
            type2bounds[k]['ub'] = torch.tensor(type2bounds[k]['ub'])
            
        self._kernel_config['type2num'] = dict()
        for k in type2bounds:
            self._kernel_config['type2num'][k] = len(type2bounds[k]['lb'])
        
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
    
    def optimize_acqf(self, acqf):
        if self._acqf_optimizer == 'l-bfgs':
            bounds = torch.vstack((self._lb, self._ub)).double().to(self._device)
            next_X, _ = botorch.optim.optimize.optimize_acqf(
                acqf, bounds=bounds, q=self._q, num_restarts=10, raw_samples=1024
            )
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
            pareto_X, _ = nsgaii_designer.result()
            pop_X, _ = nsgaii_designer.curr_pop()
            diff_X = [x for x in pop_X if x not in pareto_X]
            diff_X = np.zeros((0, pareto_X.shape[-1])) if len(diff_X) == 0 else np.vstack(diff_X)

            if len(pareto_X) >= self._q:
                idx = np.random.choice(len(pareto_X), self._q, replace=False)
                next_X = torch.from_numpy(pareto_X[idx])
            else:
                next_X = [pareto_X]
                if len(diff_X) > 0:
                    quota = min(len(diff_X), self._q-len(pareto_X))
                    idx = np.random.choice(len(diff_X), quota, replace=False)
                    next_X.append(diff_X[idx])
                quota = self._q - np.vstack(next_X).shape[0]
                if quota > 0:
                    trials = self._init_designer.suggest(quota)
                    features = self._converter.to_features(trials)
                    random_X = []
                    for name in self._converter.input_converter_dict:
                        random_X.append(features[name])
                    next_X.append(np.hstack(random_X))
                next_X = torch.from_numpy(np.vstack(next_X))
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
            features = self._converter.to_features(best_trials)
            next_X = []
            for name in self._converter.input_converter_dict:
                next_X.append(features[name])
            next_X = torch.from_numpy(np.hstack(next_X))
        else:
            raise NotImplementedError

        return next_X

    def suggest(self, count: Optional[int]=None) -> Sequence[Trial]:
        if len(self._trials) < self._n_init:
            ret = self._init_designer.suggest(count)
        else:
            count = count or 1
            features, labels = self._converter.convert(self._trials)

            train_X = []
            for k in features:
                features[k] = torch.from_numpy(features[k]).to(self._device)
                train_X.append(features[k])
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

            double_next_X, cat_next_X, rest_next_X = torch.tensor_split(
                next_X, (self._kernel_config['type2num'][SpecType.DOUBLE], self._kernel_config['type2num'][SpecType.DOUBLE]+self._kernel_config['type2num'][SpecType.CATEGORICAL]), dim=-1
            )
            assert rest_next_X.shape[-1] == 0
            features = {
                SpecType.DOUBLE: double_next_X.to('cpu').numpy(),
                SpecType.CATEGORICAL: cat_next_X.to('cpu').numpy()
            }
            ret = self._converter.to_trials(features)

        return ret

    def update(self, completed: Sequence[Trial]) -> None:
        self._trials.extend(completed)
