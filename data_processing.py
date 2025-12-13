# this class responsible for data processing tasks
# like creating lag value, data cleanign and so on
import pandas as pd


class DataProcesssing:
    def __init__(self, data):
        self.data = data

    def initialize_data(self):
        # get data from test.csv and train.csv put to class variable
        # split data untuk train dan test
        pass

    def get_x_train(self):
        return pd.DataFrame([])

    def get_y_train(self):
        return pd.DataFrame([])

    def get_x_test(self):
        return pd.DataFrame([])

    def get_y_test(self):
        return pd.DataFrame([])

