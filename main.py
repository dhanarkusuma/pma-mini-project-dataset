import data_processing as dp
import ada_boost as ab
import xg_boost as xgb
import evaluator as ev
import boostInterface as bi


class main:
    def __init__(self):
        self.value = "Hello, World!"

    def run(self):
        boost = bi.BoostInterface()

        scenarios = [
            {
                "scenario": "free",
                "n_estimator": 50,
                "learning_rate": 0.1,
                "max_depth": 3,
                "min_sample_split": 2,
                "loss": "linear",
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "gamma": 0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
            },
            {
                "scenario": "x1",
                "n_estimator": 50,
                "learning_rate": 0.1,
                "max_depth": 3,
                "min_sample_split": 2,
                "loss": "linear",
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "gamma": 0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
            },
            {
                "scenario": "x2",
                "n_estimator": 50,
                "learning_rate": 0.1,
                "max_depth": 3,
                "min_sample_split": 2,
                "loss": "linear",
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "gamma": 0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
            },
            {
                "scenario": "x1x2",
                "n_estimator": 50,
                "learning_rate": 0.1,
                "max_depth": 3,
                "min_sample_split": 2,
                "loss": "linear",
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "gamma": 0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
            },
        ]

        print("hasil tanpa parameter tuning untuk adaboost dan xgboost")
        data = dp.DataProcessing()
        data.initialize_data()

        for scenario in scenarios:
            boost = ab.Adaboost(data)
            boost.initialize(scenario)
            print(f"ada boost successfully trained for scenario {scenario}")
            self.evaluator = ev.Evaluator(boost.prediction_value(), data.get_y_test())
            # RMSE
            print(f"nilai rmse untuk ada boost : {self.evaluator.get_rmse()}")
            # MSE
            print(f"nilai mse untuk ada boost : {self.evaluator.get_mse()}")
            # MAE
            print(f"nilai mae untuk ada boost : {self.evaluator.get_mae()}")
            # MAPE
            print(f"nilai mape untuk ada boost : {self.evaluator.get_mape()}")
            # R2 Score
            print(f"nilai r2 score untuk ada boost : {self.evaluator.get_r2_score()}")
            print("--------------------------------------------------")
            print("--------------------------------------------------")
            boost = xgb.XGBoost(data)
            boost.initialize(scenario)
            print(f"xg boost successfully trained for scenario {scenario}")
            self.evaluator = ev.Evaluator(boost.prediction_value(), data.get_y_test())
            # RMS7E
            print(f"nilai rmse untuk xg boost : {self.evaluator.get_rmse()}")
            # MSE
            print(f"nilai mse untuk xg boost : {self.evaluator.get_mse()}")
            # MAE
            print(f"nilai mae untuk xg boost : {self.evaluator.get_mae()}")
            # MAPE
            print(f"nilai mape untuk xg boost : {self.evaluator.get_mape()}")
            # R2 Score
            print(f"nilai r2 score untuk xg boost : {self.evaluator.get_r2_score()}")
            print("--------------------------------------------------")
            print("--------------------------------------------------")
            print("\n\n")
        pass

        scenarios = [
            {
                "scenario": "free",
                "n_estimator": [50, 100, 200],
                "learning_rate": [0.1, 0.01, 0.05],
                "max_depth": [3, 5, 7],
                "min_sample_split": [2, 4, 6],
                "loss": ["linear", "square", "exponential"],
                "subsample": [1.0, 0.8, 0.6],
                "colsample_bytree": [1.0, 0.8, 0.6],
                "gamma": [0, 0.1, 0.3],
                "reg_alpha": [0.0, 0.1, 0.5],
                "reg_lambda": [1.0, 5.0, 10.0],
            },
            {
                "scenario": "x1",
                "n_estimator": [50, 100, 200],
                "learning_rate": [0.1, 0.01, 0.05],
                "max_depth": [3, 5, 7],
                "min_sample_split": [2, 4, 6],
                "loss": ["linear", "square", "exponential"],
                "subsample": [1.0, 0.8, 0.6],
                "colsample_bytree": [1.0, 0.8, 0.6],
                "gamma": [0, 0.1, 0.3],
                "reg_alpha": [0.0, 0.1, 0.5],
                "reg_lambda": [1.0, 5.0, 10.0],
            },
            {
                "scenario": "x2",
                "n_estimator": [50, 100, 200],
                "learning_rate": [0.1, 0.01, 0.05],
                "max_depth": [3, 5, 7],
                "min_sample_split": [2, 4, 6],
                "loss": ["linear", "square", "exponential"],
                "subsample": [1.0, 0.8, 0.6],
                "colsample_bytree": [1.0, 0.8, 0.6],
                "gamma": [0, 0.1, 0.3],
                "reg_alpha": [0.0, 0.1, 0.5],
                "reg_lambda": [1.0, 5.0, 10.0],
            },
            {
                "scenario": "x1x2",
                "n_estimator": [50, 100, 200],
                "learning_rate": [0.1, 0.01, 0.05],
                "max_depth": [3, 5, 7],
                "min_sample_split": [2, 4, 6],
                "loss": ["linear", "square", "exponential"],
                "subsample": [1.0, 0.8, 0.6],
                "colsample_bytree": [1.0, 0.8, 0.6],
                "gamma": [0, 0.1, 0.3],
                "reg_alpha": [0.0, 0.1, 0.5],
                "reg_lambda": [1.0, 5.0, 10.0],
            },
        ]
        print("hasil menggunakan parameter tuning menggunakan gridCV")

        data = dp.DataProcessing()
        data.initialize_data()

        for scenario in scenarios:
            boost = ab.Adaboost(data)
            boost.initialize_parameter_tunning(scenario)
            print(
                f"ada boost successfully trained for scenario {scenario} with parameter tuning using gridCV"
            )

            return
            self.evaluator = ev.Evaluator(boost.prediction_value(), data.get_y_test())
            # RMSE
            print(f"nilai rmse untuk ada boost : {self.evaluator.get_rmse()}")
            # MSE
            print(f"nilai mse untuk ada boost : {self.evaluator.get_mse()}")
            # MAE
            print(f"nilai mae untuk ada boost : {self.evaluator.get_mae()}")
            # MAPE
            print(f"nilai mape untuk ada boost : {self.evaluator.get_mape()}")
            # R2 Score
            print(f"nilai r2 score untuk ada boost : {self.evaluator.get_r2_score()}")
            print("--------------------------------------------------")
            print("--------------------------------------------------")
            boost = xgb.XGBoost(data)
            boost.initialize_parameter_tunning(scenario)
            print(
                f"ada boost successfully trained for scenario {scenario} with parameter tuning using gridCV"
            )
            self.evaluator = ev.Evaluator(boost.prediction_value(), data.get_y_test())
            # RMSE
            print(f"nilai rmse untuk ada boost : {self.evaluator.get_rmse()}")
            # MSE
            print(f"nilai mse untuk ada boost : {self.evaluator.get_mse()}")
            # MAE
            print(f"nilai mae untuk ada boost : {self.evaluator.get_mae()}")
            # MAPE
            print(f"nilai mape untuk ada boost : {self.evaluator.get_mape()}")
            # R2 Score
            print(f"nilai r2 score untuk ada boost : {self.evaluator.get_r2_score()}")


if __name__ == "__main__":
    app = main()
    app.run()

