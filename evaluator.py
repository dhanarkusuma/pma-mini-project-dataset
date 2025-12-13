from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error, mean_absolute_percentage_error

class Evaluator:
    def __init__(self, prediction, test):
        self.prediction = prediction
        self.test = test

    def get_rmse(self):
        metric = root_mean_squared_error(self.test, self.prediction)
        return metric
    def get_mape(self):
        metric = mean_absolute_percentage_error(self.test, self.prediction)
        return metric
    def get_mae(self):
        metric = mean_absolute_error(self.test, self.prediction)
        return metric
    def get_r2_score(self):
        metric = r2_score(self.test, self.prediction)
        return metric
    def get_mse(self):
        metric = mean_squared_error(self.test, self.prediction)
        return metric
    