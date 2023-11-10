from typing import List

from attrs import define, field, validators

from bbo.utils.parameter_config import SearchSpace
from bbo.utils.metric_config import Objective
from bbo.utils.metadata import Metadata


@define
class ProblemStatement:
    search_space: SearchSpace = field(
        validator=validators.instance_of(SearchSpace),
    )
    objective: Objective = field(
        validator=validators.instance_of(Objective),
    )
    metadata: Metadata = field(
        kw_only=True,
        factory=Metadata,
        validator=validators.instance_of(Metadata),
    )