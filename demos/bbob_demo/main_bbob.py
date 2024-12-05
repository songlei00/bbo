import logging
import argparse

from bbo.benchmarks.experimenters.synthetic.bbob import bbob_problem, funcs as bbob_funcs
from bbo.benchmarks.analyzers.plot_utils import PlotUtil
from bbo.benchmarks.analyzers.utils import trials2df
from bbo.algorithms import (
    RandomDesigner, GridSearchDesigner, LocalSearchDesigner,
    CMAESDesigner, NSGAIIDesigner, RegularizedEvolutionDesigner, PSODesigner,
    BODesigner, ChainDesigner
)
from bbo.algorithms.bo_utils import MeanFactory, KernelFactory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    # filename='bbo_log/log.log',
    # filemode='w'
)
logger = logging.getLogger(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('--algo', type=str, required=True)
parser.add_argument('--func', type=str, required=True)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

logger.info(args)

plot_util = PlotUtil(
    xy_fn=lambda r: (r.data.index, r.data['obj']),
    split_fn=lambda r: r.name.split('-')[0],
    group_fn=lambda r: r.name.split('-')[1],
    xlabel='Number of evaluations',
    ylabel='f(x)',
)
N = 100

problem_statement, exp = bbob_problem(args.func, 20, -5, 5, 0)
if args.algo == 'random':
    designer = RandomDesigner(problem_statement)
elif args.algo == 'grid':
    designer = GridSearchDesigner(problem_statement)
elif args.algo == 'ls':
    designer = LocalSearchDesigner(problem_statement)
elif args.algo == 're':
    designer = RegularizedEvolutionDesigner(problem_statement)
elif args.algo == 'pso':
    designer = PSODesigner(problem_statement)
elif args.algo == 'random_bo':
    designer = BODesigner(problem_statement, acqf_optimizer='random')
elif args.algo == 'random_kumar_bo':
    designer = BODesigner(problem_statement, acqf_optimizer='random')
else:
    raise NotImplementedError

for i in range(N):
    trials = designer.suggest()
    exp.evaluate(trials)
    designer.update(trials)

plot_util.add_result(args.func + '-' + args.algo + '-{}'.format(args.seed), trials2df(designer.trials))
plot_util.save_results()