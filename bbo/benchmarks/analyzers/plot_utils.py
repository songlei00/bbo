# https://github.com/openai/baselines/blob/master/baselines/common/plot_util.py

import logging
import os
from typing import Dict, List, Callable
from attrs import define, field, validators
from collections import defaultdict, namedtuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


Result = namedtuple('Result', ['name', 'data'])

def default_xy_fn(r: Result):
    return r.data['x'], r.data['y']

def default_split_fn(r: Result):
    return ''

def default_group_fn(r: Result):
    return r.name


@define
class PlotUtil:
    _xy_fn: Callable = field()
    _split_fn: Callable = field(default=default_split_fn, kw_only=True)
    _group_fn: Callable = field(default=default_group_fn, kw_only=True)
    _xlabel: str | None = field(
        validator=validators.optional(validators.instance_of(str)), default=None, kw_only=True
    )
    _ylabel: str | None = field(
        validator=validators.optional(validators.instance_of(str)), default=None, kw_only=True
    )
    _save_dir: str | None = field(
        validator=validators.optional(validators.instance_of(str)),
        default='bbo_log', kw_only=True
    )
    _legend_show: bool = field(default=True, validator=validators.instance_of(bool), kw_only=True)
    _ncols: int | None = field(
        default=None, validator=validators.optional(validators.instance_of(int)), kw_only=True
    )
    _shaded_err: bool = field(default=True, validator=validators.instance_of(bool), kw_only=True)
    _shaded_std: bool = field(default=True, validator=validators.instance_of(bool), kw_only=True)

    _allresults: List[Result] = field(factory=list, init=False)
    _colors: List[str] = field(factory=lambda: [
        'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal',  'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold',  'darkred', 'darkblue'
    ], init=False)
    _params: Dict = field(factory=lambda: {
        'lines.linewidth': 2,
        'legend.fontsize': 20,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
    }, init=False)

    def add_result(self, name: str, df: pd.DataFrame):
        self._allresults.append(Result(name, df))

    def save_results(self):
        save_dir = self._save_dir
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        for r in self._allresults:
            r.data.to_csv(os.path.join(save_dir, r.name + '.csv'))

    def load_results(self, filter_fn=lambda path: True):
        load_dir = self._save_dir
        for f_path, d_name, f_names in os.walk(load_dir):
            for f_name in f_names:
                path = os.path.join(f_path, f_name)
                if f_name.endswith('.csv') and filter_fn(path):
                    df = pd.read_csv(path, index_col=0)
                    self.add_result(f_name.rstrip('.csv'), df)

    def plot(self):
        rc_params = matplotlib.rcParams
        matplotlib.rcParams.update(self._params)

        assert len(self._allresults) > 0
        sk2r = defaultdict(list)
        for r in self._allresults:
            sk2r[self._split_fn(r)].append(r)

        ncols = int(self._ncols or np.ceil(np.sqrt(len(sk2r))))
        nrows = int(np.ceil(len(sk2r) / ncols))
        _, axarr = plt.subplots(nrows, ncols, squeeze=False, dpi=300, figsize=(10*ncols, 6*nrows))

        groups = list(set(self._group_fn(result) for result in self._allresults))
        groups.sort()
        g2l = dict()
        for i, sk in enumerate(sorted(sk2r.keys())):
            idx_row, idx_col = i // ncols, i % ncols
            ax = axarr[idx_row][idx_col]
            ax.set_title(sk)
            sresults = sk2r[sk]
            gresults = defaultdict(list)
            for r in sresults:
                group = self._group_fn(r)
                x, y = self._xy_fn(r)
                gresults[group].append((x, y))

            for j, group in enumerate(groups):
                xys = gresults[group]
                if not any(xys):
                    continue
                color = self._colors[j % len(self._colors)]
                x = xys[0][0]
                ys = [xy[1] for xy in xys]
                ymean = np.mean(ys, axis=0)
                ystd = np.std(ys, axis=0)
                ystderr = ystd / np.sqrt(len(ys))
                l, = ax.plot(x, ymean, color=color, label=group)
                g2l[group] = l
                if self._shaded_err:
                    ax.fill_between(x, ymean-ystderr, ymean+ystderr, color=color, alpha=.4)
                if self._shaded_std:
                    ax.fill_between(x, ymean-ystd, ymean+ystd, color=color, alpha=.2)
        
        if self._legend_show and any(g2l.keys()):
            axarr[0][-1].legend(g2l.values(), g2l.keys(), loc=2, bbox_to_anchor=(1, 1))
        if self._xlabel is not None:
            for ax in axarr[-1]:
                ax.set_xlabel(self._xlabel)
        if self._ylabel is not None:
            for ax in axarr[:, 0]:
                ax.set_ylabel(self._ylabel)
        plt.tight_layout()
        matplotlib.rcParams.update(rc_params)

        if self._save_dir is not None:
            if not os.path.exists(self._save_dir):
                os.makedirs(self._save_dir)
            plt.savefig(os.path.join(self._save_dir, 'plot_result.pdf'))