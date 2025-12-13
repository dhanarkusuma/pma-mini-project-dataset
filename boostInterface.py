from abc import  abstractmethod

class BoostInterface:

    @abstractmethod
    def initilize(self,  choose_scenario:str):
        pass

    @abstractmethod
    def prediction_value(self):
        pass

    @abstractmethod
    def parameter_tuning(self, param_grid, scoring, cv):
        pass
