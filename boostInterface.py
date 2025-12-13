from abc import abstractmethod
import numpy as np


class BoostInterface:
    @abstractmethod
    def initilize(self, choose_scenario: str):
        pass

    @abstractmethod
    def prediction_value(self):
        return np.array([])

    @abstractmethod
    def predict_fine_tune(self):
        return np.array([])

    @abstractmethod
    def parameter_tuning(self, param_grid, scoring, cv):
        pass
