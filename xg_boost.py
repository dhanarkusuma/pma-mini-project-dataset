# using library sklearn.ensemble import AdaBoostClassifier
# input parameter adjusted to adaboost classifier
import boostInterface
import data_processing as dp

class XGBoost(boostInterface.BoostInterface):
    def __init__(self, data: dp.DataProcesssing):
        self.data = data

    def initialize(self, choose_scenario: str):
        pass

    def prediction_value(self):
        pass
    
