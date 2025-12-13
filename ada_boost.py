from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

import data_processing as dp
import boostInterface
import numpy as np


class Adaboost(boostInterface.BoostInterface):
    def __init__(self, data: dp.DataProcesssing):
        self.data = data
        self.scenario = None

    def initialize(self, dict_args):
        self.scenario = dict_args["scenario"]
        self.n_estimators = dict_args["n_estimators"]
        self.learning_rate = dict_args["learning_rate"]
        self.loss = dict_args["loss"]
        self.max_depth = dict_args["max_depth"]
        self.random_state = dict_args["random_state"]
        X = self.get_X(self.data.get_x_train())

        estimator = DecisionTreeRegressor(
            max_depth=self.max_depth, random_state=self.random_state
        )
        self.model = AdaBoostRegressor(
            estimator=estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            loss=self.loss,
            random_state=42,
        )
        self.model.fit(X, self.data.get_y_train())

    def get_model(self):
        return self.model

    def get_X(self, X):
        match self.scenario:
            case "x1":
                return X[0]
            case "x2":
                return X[1]
            case "x1x2":
                return X
            case _:
                pass

    def prediction_value(self):
        if self.model == "None":
            return np.array([])
        X_test = self.data.get_x_test()
        X_test = self.get_X(X_test)
        return self.model.predict(X_test)
