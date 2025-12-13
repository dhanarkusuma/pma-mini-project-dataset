# using library sklearn.ensemble import AdaBoostClassifier
# input parameter adjusted to adaboost classifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostRegressor

import data_processing as dp
import boostInterface


class Adaboost(boostInterface.BoostInterface):
    def __init__(
        self,
        data: dp.DataProcesssing,
    ):
        self.data = data
        self.model = None

    def initialize(self, dict_args):
        X = self.data.get_x_train()
        n_estimators = dict_args["n_estimators"]
        learning_rate = dict_args["learning_rate"]

        match dict_args["scenario"]:
            case "x1":
                X = X[0]
            case "x2":
                X = X[1]
            case "x1x2":
                pass
            case "tunning":
                pass

        self.adaBoost = AdaBoostRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42,
        )
        pass

    def prediction_value(self):
        pass


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import AdaBoostRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error


# url ="https://raw.githubusercontent.com/mirsabayuprasetyo-bot/data_machine_learning/refs/heads/main/data_time_series.csv"
# df = pd.read_csv(url)

# display(df.head())

# x = df[['X1 (curah hujan, cc)']]
# y = df['Y (jumlah kasus)']

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

# adaBoost = AdaBoostRegressor(n_estimators=100, learning_rate=1.0, random_state=42)

# adaBoost.fit(x_train, y_train)


# y_pred = adaBoost.predict(x_test)

# print(y_pred)

# rmse = root_mean_squared_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)

# print(f"rmse : {rmse}")
# print(f"mae : {mae}")
# print(f"mse : {mse}")

