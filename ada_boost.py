from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

import data_processing as dp
import boost_interface
import numpy as np


class Adaboost(boost_interface.BoostInterface):
    def __init__(self, data: dp.DataProcessing):
        self.data = data
        self.scenario = None
        self.model = AdaBoostRegressor()
        param_grid = {
            "n_estimators": [200, 400],
            "learning_rate": [0.05, 0.1],
        }

        self.model_tune = GridSearchCV(estimator=self.model, param_grid=param_grid)

    def initialize(self, dict_args):
        self.scenario = dict_args["scenario"]
        self.n_estimators = dict_args["n_estimator"]
        self.learning_rate = dict_args["learning_rate"]
        self.loss = dict_args["loss"]
        self.max_depth = dict_args["max_depth"]
        self.random_state = 42

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

        X = self.get_X(self.data.get_x_train())
        self.model.fit(X, self.data.get_y_train())

    def get_model(self):
        return self.model

    def get_X(self, X):
        match self.scenario:
            case "free":
                return X.drop(columns=["X2_lama_hujan", "X1_curah_hujan"])
            case "x1":
                return X.drop(columns=["X2_lama_hujan"])
            case "x2":
                return X.drop(columns=["X1_curah_hujan"])
            case "x1x2":
                return X
            case _:
                return X

    def prediction_value(self):
        if self.model == "None":
            return np.array([])
        X_test = self.get_X(self.data.get_x_test())
        return self.model.predict(X_test)

    def initialize_parameter_tunning(self, params):
        params_grid = {
            "n_estimators": params["n_estimator"],
            "learning_rate": params["learning_rate"],
        }

        estimator = DecisionTreeRegressor(
            max_depth=params["max_depth"][0], random_state=42
        )

        self.model = AdaBoostRegressor(
            estimator=estimator,
            n_estimators=params["n_estimator"][0],
            learning_rate=params["learning_rate"][0],
            loss=params["loss"][0],
            random_state=42,
        )

        model_tune = GridSearchCV(
            estimator=self.model,
            param_grid=params_grid,
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
            verbose=1,
        )

        X = self.get_X(self.data.get_x_train())
        model_tune.fit(X, self.data.get_y_train())
        self.model = model_tune.best_estimator_
