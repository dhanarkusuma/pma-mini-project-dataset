import boostInterface
import data_processing as dp
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

import numpy as np


class XGBoost(boostInterface.BoostInterface):
    def __init__(self, data: dp.DataProcessing):
        self.data = data
        self.scenario = None

    def initialize(self, dict_args):
        self.scenario = dict_args["scenario"]
        self.n_estimators = dict_args["n_estimators"]
        self.learning_rate = dict_args["learning_rate"]
        self.max_depth = dict_args["max_depth"]
        self.random_state = dict_args["random_state"]
        self.subsample = dict_args["subsample"]
        self.colsample_bytree = dict_args["colsample_bytree"]
        self.gamma = dict_args["gamma"]
        self.reg_alpha = dict_args["reg_alpha"]
        self.reg_lambda = dict_args["reg_lambda"]
        self.X = self.get_X(self.data.get_x_train())

        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,  # L1
            reg_lambda=self.reg_lambda,  # L2
            random_state=self.random_state,
        )
        self.model.fit(self.X, self.data.get_y_train())

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
                return X

    def prediction_value(self):
        if self.model == "None":
            return np.array([])
        X_test = self.data.get_x_test()
        X_test = self.get_X(X_test)
        return self.model.predict(X_test)

    def initialize_parameter_tunning(self, params):
        param_grid = {
            "n_estimators": params["n_estimator"],
            "learning_rate": params["learning_rate"],
            "max_depth": params["max_depth"],
            "subsample": params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "gamma": params["gamma"],
            "reg_alpha": [0.0, 0.1],
            "reg_lambda": [1.0, 5.0],
        }
        grid = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=5,
            verbose=1,
            n_jobs=-1,
        )
        grid.fit(self.X, self.data.get_y_train())
        self.model = grid.best_estimator_
