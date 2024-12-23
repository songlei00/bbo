import logging

from attrs import define, field, validators
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from tensorboardX import SummaryWriter

from bbo.predictors.base import Predictor
from bbo.predictors.gp.kernel_factory import KernelFactory
from bbo.predictors.gp.mean_factory import MeanFactory

logger = logging.getLogger(__name__)


class GP(ExactGP):
    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        mean_factory: MeanFactory,
        kernel_factory: KernelFactory,
        likelihood
    ):
        super(GP, self).__init__(train_X, train_Y, likelihood)
        self._mean_module = mean_factory()
        self._covar_module = kernel_factory()

    def forward(self, X):
        mean = self._mean_module(X)
        covar = self._covar_module(X)
        return MultivariateNormal(mean, covar)


@define
class GPPredictor(Predictor):
    _mean_factory: MeanFactory = field(default=MeanFactory('constant'), kw_only=True)
    _kernel_factory: KernelFactory = field(default=KernelFactory('matern52'), kw_only=True)

    # Training config
    _optimizer: str = field(default='adam', validator=validators.in_(['adam', 'lbfgs']), kw_only=True)
    _lr: float = field(default=0.01, converter=float, kw_only=True)
    _epochs: int = field(default=200, kw_only=True)
    _batch_size: int = field(default=32, kw_only=True)
    _device: str = field(default='cpu', kw_only=True)
    _logdir: str = field(default='bbo_log', kw_only=True)

    _model = field(init=False)
    _likelihood = field(init=False)
    _mll_fn = field(init=False)
    _mse_fn = field(init=False)
    _tb_writer = field(init=False)

    def __attrs_post_init__(self):
        self._tb_writer = SummaryWriter(self._logdir)
        self._device = torch.device(self._device if torch.cuda.is_available() else 'cpu')

    def fit(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        val_X: torch.Tensor = None,
        val_Y: torch.Tensor = None,
        val_interval: int = 50
    ):
        train_X, train_Y = train_X.to(self._device), train_Y.to(self._device)
        if val_X is not None and val_Y is not None:
            val_X, val_Y = val_X.to(self._device), val_Y.to(self._device)

        train_dataset = TensorDataset(train_X, train_Y)
        train_dataloader = DataLoader(train_dataset, self._batch_size, True)

        self._likelihood = GaussianLikelihood().to(self._device)
        self._model = GP(
            train_X, train_Y, self._mean_factory, self._kernel_factory, self._likelihood
        ).to(self._device)
        self._mll_fn = ExactMarginalLogLikelihood(self._likelihood, self._model)
        self._mse_fn = nn.MSELoss()
        if self._optimizer == 'adam':
            optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        elif self._optimizer == 'lbfgs':
            optimizer = torch.optim.LBFGS(self._model.parameters(), lr=self._lr)
        else:
            raise NotImplementedError
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self._epochs, eta_min=1e-7)

        for i in range(self._epochs):
            self._model.train()
            self._likelihood.train()
            loss_sum, cnt = 0, 0
            
            for X, Y in train_dataloader:
                self._model.set_train_data(inputs=X, targets=Y, strict=False)
                loss_list = []
                def closure():
                    optimizer.zero_grad()
                    output = self._model(X)
                    loss = -self._mll_fn(output, Y)
                    loss_list.append(loss.item())
                    loss.backward()
                    return loss
                optimizer.step(closure)
                loss_sum += sum(loss_list)
                cnt += len(loss_list)

            loss_mean = loss_sum / cnt
            scheduler.step()
            self._tb_writer.add_scalar('fit/train_mll', -loss_mean, global_step=i)
            
            if (i+1) % val_interval == 0 and val_X is not None and val_Y is not None:
                mll_mean, mse_mean = self.evaluate(val_X, val_Y, train_X, train_Y)
                logger.info(f'epoch: {i+1}, train_mll: {-loss_mean}, val_mll: {mll_mean}, val_mse: {mse_mean}')
                self._tb_writer.add_scalar('fit/val_mll', mll_mean, global_step=i)
                self._tb_writer.add_scalar('fit/val_mse', mse_mean, global_step=i)

        self._model.eval()
        self._likelihood.eval()

    def predict(self, test_X: torch.Tensor):
        self._model.eval()
        self._likelihood.eval()
        test_X = test_X.to(self._device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_Y = self._likelihood(self._model(test_X))
        return pred_Y
    
    def evaluate(
        self,
        test_X: torch.Tensor,
        test_Y: torch.Tensor,
        context_X: torch.Tensor = None,
        context_Y: torch.Tensor = None,
    ):
        test_X, test_Y = test_X.to(self._device), test_Y.to(self._device)
        if context_X is not None and context_Y is not None:
            context_X, context_Y = context_X.to(self._device), context_Y.to(self._device)
            self._model.set_train_data(inputs=context_X, targets=context_Y, strict=False)

        test_dataset = TensorDataset(test_X, test_Y)
        test_dataloader = DataLoader(test_dataset, self._batch_size, False)

        mll_sum, mse_sum, cnt = 0, 0, 0
        for X, Y in test_dataloader:
            pred_Y = self.predict(X)
            mll_sum += self._mll_fn(pred_Y, Y).item()
            mse_sum += self._mse_fn(pred_Y.mean, Y).item()
            cnt += 1

        return mll_sum / cnt, mse_sum / cnt