from typing import Sequence
import logging

import matplotlib
params = {
    'lines.linewidth': 2,
    'legend.fontsize': 20,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
}
matplotlib.rcParams.update(params)
import matplotlib.pyplot as plt

from bbo.utils.trial import Trial
from bbo.benchmarks.analyzers.utils import trials2df

def plot_func(
    x_data, y_data,
    save_path: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None
):
    plt.figure(dpi=300)
    plt.plot(x_data, y_data)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.savefig(save_path, bbox_inches='tight')

def plot_trials(trials: Sequence[Trial]):
    df = trials2df(trials)
    for obj_name in trials[0].metrics:
        file_name = '{}.pdf'.format(obj_name)
        plot_func(df.index, df[obj_name], file_name, 'x', 'y', obj_name)
        logging.info('Plot results are saved to {}.'.format(file_name))
