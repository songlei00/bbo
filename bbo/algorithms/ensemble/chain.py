import logging
from typing import Sequence, Union, Optional

from attrs import define, field, validators

from bbo.algorithms.base import Designer
from bbo.utils.problem_statement import ProblemStatement
from bbo.utils.trial import Trial

logger = logging.getLogger(__name__)


@define
class ChainDesigner(Designer):
    _problem_statement: ProblemStatement = field(
        validator=validators.instance_of(ProblemStatement)
    )
    _cand_designers: Sequence[Designer] = field(
        validator=validators.deep_iterable(validators.instance_of(Designer))
    )
    _cand_quota: Union[Sequence[int], int] = field(
        default=1, kw_only=True,
        validator=validators.or_(
            validators.instance_of(int),
            validators.deep_iterable(validators.instance_of(int))
        )
    )
    _curr_idx: int = field(default=0, init=False)
    _curr_quota: int = field(init=False)

    def __attrs_post_init__(self):
        if isinstance(self._cand_quota, int):
            self._cand_quota = [self._cand_quota] * len(self._cand_designers)
        self._curr_quota = self._cand_quota[0]

    def _suggest(self, count: Optional[int]=None) -> Sequence[Trial]:
        return self._cand_designers[self._curr_idx].suggest(count)

    def _update(self, completed: Sequence[Trial]) -> None:
        for designer in self._cand_designers:
            designer.update(completed)
        self._curr_quota -= 1
        if self._curr_quota == 0:
            self._curr_idx = (self._curr_idx + 1) % len(self._cand_designers)
            self._curr_quota = self._cand_quota[self._curr_idx]