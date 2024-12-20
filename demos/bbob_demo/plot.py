import numpy as np
from bbo.benchmarks.analyzers.plot_utils import PlotUtil

plot_util = PlotUtil(
    # xy_fn=lambda r: (r.data.index, r.data['obj'].apply(np.log)),
    xy_fn=lambda r: (r.data.index, r.data['obj'].apply(np.log).cummin()),
    split_fn=lambda r: r.name.split('-')[0],
    group_fn=lambda r: r.name.split('-')[1],
    xlabel='Number of evaluations',
    ylabel='f(x)',
    save_dir='bbob_log'
)
plot_util.load_results()
plot_util.plot()