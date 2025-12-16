import boost_interface
import data_processing as dp
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

import numpy as np


class XGBoost(boost_interface.BoostInterface):
    def __init__(self, data: dp.DataProcessing):
        self.data = data
        self.scenario = None
        self.model = xgb.XGBRegressor()

        param_grid = {
            "n_estimators": [200, 400],
            "learning_rate": [0.05, 0.1],
        }
        self.model_tune = GridSearchCV(estimator=self.model, param_grid=param_grid)
        self.predict_ft = 0

    def initialize(self, dict_args):
        self.scenario = dict_args["scenario"]
        self.n_estimators = dict_args["n_estimator"]
        self.learning_rate = dict_args["learning_rate"]
        self.max_depth = dict_args["max_depth"]
        self.random_state = 42
        self.subsample = dict_args["subsample"]
        self.colsample_bytree = dict_args["colsample_bytree"]
        self.gamma = dict_args["gamma"]
        self.reg_alpha = dict_args["reg_alpha"]
        self.reg_lambda = dict_args["reg_lambda"]

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
        param_grid = {
            "n_estimators": params["n_estimator"],
            "learning_rate": params["learning_rate"],
            "max_depth": params["max_depth"],
            "subsample": params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "gamma": params["gamma"],
            "reg_alpha": params["reg_alpha"],
            "reg_lambda": params["reg_lambda"],
        }
        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=params["n_estimator"][0],
            learning_rate=params["learning_rate"][0],
            max_depth=params["max_depth"][0],
            subsample=params["subsample"][0],
            colsample_bytree=params["colsample_bytree"][0],
            gamma=params["gamma"][0],
            reg_alpha=params["reg_alpha"][0],  # L1
            reg_lambda=params["reg_lambda"][0],  # L2
            random_state=42,
        )

        model_tune = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=3,
            verbose=1,
            n_jobs=-1,
        )
        X = self.get_X(self.data.get_x_train())
        model_tune.fit(X, self.data.get_y_train())
        self.model = model_tune.best_estimator_
