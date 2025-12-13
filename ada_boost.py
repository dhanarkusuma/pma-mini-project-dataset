# using library sklearn.ensemble import AdaBoostClassifier
# input parameter adjusted to adaboost classifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

import data_processing as dp
import boostInterface

class Adaboost(boostInterface.BoostInterface):
    def __init__(self, data: dp.DataProcesssing,):
        self.data = data


    def initialize(self, choose_scenario:str):
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