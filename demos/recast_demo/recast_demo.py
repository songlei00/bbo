import logging
import argparse
import os

from absl import logging as absl_logging

from bbo.recastlib import VizierDesigner
from bbo.benchmarks.experimenters.synthetic.bbob import bbob_problem
from bbo.benchmarks.analyzers.plot_utils import PlotUtil
from bbo.utils.utils import trials2df

parser = argparse.ArgumentParser()
parser.add_argument('--algo', type=str, required=True)
parser.add_argument('--func', type=str, required=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--N', type=int, default=100)
args = parser.parse_args()

class Filter(logging.Filterer):
    def filter(self, record):
        return False

absl_logging.get_absl_logger().addFilter(Filter())
logging.getLogger().addFilter(Filter())

logdir = os.path.join('recast_log', args.func, args.algo)
if not os.path.exists(logdir):
    os.makedirs(logdir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=os.path.join(logdir, '{}-{}-{}-log.log'.format(args.func, args.algo, args.seed)),
    filemode='w'
)

logger = logging.getLogger(__file__)
logger.info(args)

plot_util = PlotUtil(
    xy_fn=lambda r: (r.data.index, r.data['obj']),
    split_fn=lambda r: r.name.split('-')[0],
    group_fn=lambda r: r.name.split('-')[1],
    xlabel='Number of evaluations',
    ylabel='f(x)',
    save_dir=logdir
)

problem_statement, exp = bbob_problem(args.func, 30, -5, 5, 0)
designer = VizierDesigner(problem_statement, algorithm=args.algo, seed=args.seed)

for i in range(args.N):
    logger.info('epoch {}'.format(i))
    trials = designer.suggest()
    exp.evaluate(trials)
    designer.update(trials)

plot_util.add_result(args.func + '-' + args.algo + '-{}'.format(args.seed), trials2df(designer.base_trials))
plot_util.save_results()