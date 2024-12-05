import logging
import argparse

from bbo.benchmarks.analyzers.plot_utils import PlotUtil

plot_util = PlotUtil(
    xy_fn=lambda r: (r.data.index, r.data['obj']),
    split_fn=lambda r: r.name.split('-')[0],
    group_fn=lambda r: r.name.split('-')[1],
    xlabel='Number of evaluations',
    ylabel='f(x)',
)
plot_util.load_results()
plot_util.plot()