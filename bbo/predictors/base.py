import abc


class Predictor(abc.ABC):
    @abc.abstractmethod
    def fit(self):
        pass
    
    @abc.abstractmethod
    def predict(self):
        pass