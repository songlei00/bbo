import logging
import os
from typing import Sequence, Dict, List
from attrs import define, field, validators

import matplotlib.pyplot as plt

from bbo.utils.problem_statement import ProblemStatement
from bbo.utils.metric_config import ObjectiveMetricGoal
from bbo.utils.trial import Trial
from bbo.benchmarks.analyzers.utils import trials2df

logger = logging.getLogger(__name__)


@define
class PlotUtil:
    _problem_statement: ProblemStatement = field(
        validator=validators.instance_of(ProblemStatement)
    )
    _save_dir: str | None = field(
        validator=validators.optional(validators.instance_of(str)), default=None
    )
    _xlabel: str | None = field(
        validator=validators.optional(validators.instance_of(str)), default=None, kw_only=True
    )
    _ylabel: str | None = field(
        validator=validators.optional(validators.instance_of(str)), default=None, kw_only=True
    )
    _title: str | None = field(
        validator=validators.optional(validators.instance_of(str)), default=None, kw_only=True
    )

    _label2trials: Dict[str, Sequence[Trial]] = field(factory=dict, init=False)
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

    def add_trials(self, trials: Sequence[Trial], label: str):
        self._label2trials[label] = trials

    def del_trials(self, label: str):
        self._label2trials.pop(label, None)

    def plot(self, besty=False):
        for m in self._problem_statement.objective.metrics:
            with plt.rc_context(self._params):
                _, ax = plt.subplots(dpi=300)
                for i, (label, trials) in enumerate(self._label2trials.items()):
                    df = trials2df(trials)
                    if besty:
                        if m.goal == ObjectiveMetricGoal.MAXIMIZE:
                            y = df[m.name].cummax()
                        else:
                            y = df[m.name].cummin()
                    else:
                        y = df[m.name]
                    ax.plot(df.index, y, label=label, color=self._colors[i%len(self._colors)])
                ax.set_xlabel(self._xlabel)
                ax.set_ylabel(self._ylabel)
                ax.set_title(self._title)
                ax.legend()
            if self._save_dir is not None:
                if besty:
                    file_name = os.path.join(self._save_dir, m.name + '_best.pdf')
                else:
                    file_name = os.path.join(self._save_dir, m.name + '.pdf')
                plt.savefig(file_name, bbox_inches='tight')
                logger.debug('Plot to {}'.format(file_name))
            else:
                plt.show()