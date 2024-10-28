from typing import List, Sequence, Optional, Union

from attrs import define, field, validators
import numpy as np
import torch
from torch import optim
import botorch
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement
import gpytorch
from gpytorch.constraints import GreaterThan

from bbo.algorithms.base import Designer
from bbo.algorithms.random import RandomDesigner
from bbo.utils.converters.converter import SpecType, DefaultTrialConverter
from bbo.utils.problem_statement import ProblemStatement
from bbo.utils.trial import Trial
from bbo.algorithms.bo_utils.mean_factory import mean_factory
from bbo.algorithms.bo_utils.kernel_factory import kernel_factory


@define
class BO(Designer):
    _problem_statement: ProblemStatement = field(
        validator=validators.instance_of(ProblemStatement)
    )
    _n_init: int = field(default=10, kw_only=True)
    _q: int = field(default=1, kw_only=True)
    _device: str = field(default='cpu', kw_only=True)

    # surrogate model configuration
    _mean_type: Optional[str] = field(default=None, kw_only=True)
    _mean_config: Optional[dict] = field(default=None, kw_only=True)
    _kernel_type: Optional[str] = field(default=None, kw_only=True)
    _kernel_config: Optional[dict] = field(default=None, kw_only=True)

    # surrogate model optimization configuration
    _mll_optimizer: str = field(default='l-bfgs', kw_only=True)
    _mll_lr: Optional[float] = field(default=None, kw_only=True)
    _mll_epochs: Optional[int] = field(default=None, kw_only=True)

    # acquisition function configuration
    _acqf_type: Union[str, List[str]] = field(default='EI', kw_only=True)
    _acqf_optimizer: str = field(default='l-bfgs', kw_only=True)

    # internal attributes
    _trials: List[Trial] = field(factory=list, init=False)

    def __attrs_post_init__(self):
        self._init_designer = RandomDesigner(self._problem_statement)
        self._converter = DefaultTrialConverter.from_problem(self._problem_statement, merge_by_type=True)
        lb, ub = [], []
        for spec in self._converter.output_spec.values():
            if spec.type == SpecType.DOUBLE:
                lb.append(spec.bounds[0])
                ub.append(spec.bounds[1])
            else:
                raise NotImplementedError('Unsupported variable type for BO')
        self._lb, self._ub = torch.tensor(lb), torch.tensor(ub)
        self._device = torch.device(self._device if torch.cuda.is_available() else 'cpu')

    def create_model(self, train_X, train_Y):
        mean_module = mean_factory(self._mean_type, self._mean_config)
        covar_module = kernel_factory(self._kernel_type, self._kernel_config)
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
        def get_acqf(acqf_type, model, train_X, train_Y):
            if acqf_type == 'EI':
                acqf = ExpectedImprovement(model, train_Y.max())
            else:
                raise NotImplementedError
            return acqf

        if isinstance(self._acqf_type, list):
            acqf = []
            for acqf_type in self._acqf_type:
                acqf_tmp = get_acqf(acqf_type, model, train_X, train_Y)
                acqf.append(acqf_tmp)
        else:
            acqf = get_acqf(self._acqf_type, model, train_X, train_Y)
            
        return acqf
    
    def optimize_acqf(self, acqf):
        bounds = torch.vstack((self._lb, self._ub)).double().to(self._device)
        if self._acqf_optimizer == 'l-bfgs':
            next_X, _ = botorch.optim.optimize.optimize_acqf(
                acqf, bounds=bounds, q=self._q, num_restarts=10, raw_samples=1024
            )
        else:
            raise NotImplementedError

        return next_X

    def suggest(self, count: Optional[int]=None) -> Sequence[Trial]:
        if len(self._trials) < self._n_init:
            ret = self._init_designer.suggest(count)
        else:
            count = count or 1
            features, labels = self._converter.convert(self._trials)

            train_X = features
            if not (SpecType.DOUBLE in train_X and len(train_X) == 1):
                raise NotImplementedError('Unsupported variable type for BO')
            train_X = train_X[SpecType.DOUBLE]

            train_Y = []
            for _, v in labels.items():
                train_Y.append(v)
            train_Y = np.concatenate(train_Y, axis=-1)
            if train_Y.shape[-1] > 1:
                raise NotImplementedError('Unsupported for multiobjective BO')
            train_Y = (train_Y - train_Y.mean()) / (train_Y.std() + 1e-6)

            train_X = torch.from_numpy(train_X).double().to(self._device)
            train_Y = torch.from_numpy(train_Y).double().to(self._device)
        
            mll, model = self.create_model(train_X, train_Y)
            self.optimize_model(mll, model, train_X, train_Y)
            acqf = self.create_acqf(model, train_X, train_Y)
            next_X = self.optimize_acqf(acqf)
            next_X = next_X.to('cpu').numpy()

            features = {SpecType.DOUBLE: next_X}
            ret = self._converter.to_trials(features)

        return ret

    def update(self, completed: Sequence[Trial]) -> None:
        self._trials.extend(completed)
