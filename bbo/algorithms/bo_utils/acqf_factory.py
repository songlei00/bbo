from botorch.acquisition import (
    qExpectedImprovement,
    qUpperConfidenceBound,
    qProbabilityOfImprovement,
    qLogExpectedImprovement
)


def acqf_factory(acqf_type, model, train_X, train_Y):
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
