import os
import argparse
import logging
from functools import partial
import torch
from bbo.benchmarks.datasets import UCIDataset
from bbo.predictors.gp import GPPredictor, MeanFactory, KernelFactory

parser = argparse.ArgumentParser()
parser.add_argument('--predictor', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

logdir = os.path.join('predictor_log', args.dataset, args.optimizer+'_'+args.predictor)
if not os.path.exists(logdir):
    os.makedirs(logdir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=os.path.join(logdir, '{}-{}-log.log'.format(args.dataset, args.predictor)),
    filemode='w'
)
logger = logging.getLogger(__file__)

# load dataset
train_dataset = UCIDataset(args.dataset, train=True)
test_dataset = UCIDataset(args.dataset, train=False)
dims = train_dataset[0][0].shape[0]
logger.info('dims: {}, train size: {}, test size: {}'.format(dims, len(train_dataset), len(test_dataset)))

train_size = int(len(train_dataset) * 0.8)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset)-train_size])

# preprocess
def data_process(X, Y, min_X, max_X, mean_Y, std_Y):
    X = (X - min_X) / (max_X - min_X)
    Y = (Y - mean_Y) / (std_Y + 1e-6)
    return X.float(), Y.float()
train_X, train_Y = train_dataset[:]
val_X, val_Y = val_dataset[:]
train_Y, val_Y = train_Y.squeeze(), val_Y.squeeze()
min_X, max_X = train_X.min(dim=0).values, train_X.max(dim=0).values
mean_Y, std_Y = train_Y.mean(), train_Y.std()
mask = ~torch.isclose(min_X, max_X)
logger.info('X min: {}, X max: {}'.format(min_X, max_X))
logger.info('Y mean: {}, Y std: {}'.format(mean_Y, std_Y))

train_X, val_X = train_X[:, mask], val_X[:, mask]
min_X, max_X = min_X[mask], max_X[mask]
dims = train_X.shape[-1]
logger.info('After preprocessing, dims: {}, train size: {}, val size: {}'.format(dims, len(train_X), len(val_X)))

data_process = partial(data_process, min_X=min_X, max_X=max_X, mean_Y=mean_Y, std_Y=std_Y)
train_X, train_Y = data_process(train_X, train_Y)
val_X, val_Y = data_process(val_X, val_Y)

train_config = {
    'optimizer': args.optimizer,
    'lr': 0.01,
    'epochs': args.epochs,
    'batch_size': 32,
    'device': args.device,
    'logdir': logdir
}

if args.predictor == 'gp':
    predictor = GPPredictor(**train_config)
elif args.predictor == 'kumar_gp':
    predictor = GPPredictor(kernel_factory=KernelFactory('kumar'), **train_config)
elif args.predictor == 'mlp_gp':
    predictor = GPPredictor(kernel_factory=KernelFactory('mlp', hidden_features=[dims, 32, 16]), **train_config)
else:
    raise NotImplementedError

predictor.fit(train_X, train_Y, val_X, val_Y, val_interval=20)

# test dataset
test_X, test_Y = test_dataset[:]
test_Y = test_Y.squeeze()
test_X, test_Y = data_process(test_X, test_Y)
mll_mean, mse_mean = predictor.evaluate(test_X, test_Y, train_X, train_Y)
logger.info(f'test_dataset mll_mean: {mll_mean}, mse_mean: {mse_mean}')