from typing import Union, List

from attrs import define, field, validators
from botorch.acquisition import (
    qExpectedImprovement,
    qUpperConfidenceBound,
    qProbabilityOfImprovement,
    qLogExpectedImprovement
)


@define
class AcqfFactory:
    _acqf_type: Union[str, List[str]] = field(
        default='qlogEI',
        validator=validators.or_(
            validators.in_(['qEI', 'qUCB', 'qPI', 'qlogEI']),
            validators.deep_iterable(
                validators.in_(['qEI', 'qUCB', 'qPI', 'qlogEI'])
            ),
        )
    )

    @staticmethod
    def _create_acqf(acqf_type, model, train_X, train_Y):
        if acqf_type == 'qEI':
            acqf = qExpectedImprovement(model, train_Y.max())
        elif acqf_type == 'qUCB':
            acqf = qUpperConfidenceBound(model, beta=0.18)
        elif acqf_type == 'qPI':
            acqf = qProbabilityOfImprovement(model, train_Y.max())
        elif acqf_type == 'qlogEI':
            acqf = qLogExpectedImprovement(model, train_Y.max())
        else:
            raise NotImplementedError
        return acqf

    def __call__(self, model, train_X, train_Y):
        if isinstance(self._acqf_type, str):
            return self._create_acqf(self._acqf_type, model, train_X, train_Y)
        else:
            acqf = []
            for acqf_type in self._acqf_type:
                acqf.append(self._create_acqf(acqf_type, model, train_X, train_Y))
            return acqf
        
    @property
    def acqf_type(self):
        return self._acqf_type
